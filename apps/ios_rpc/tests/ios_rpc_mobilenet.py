# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import os
import re
import sys

import coremltools
import numpy as np
import tvm
from mxnet import gluon
from PIL import Image
from tvm import relay, rpc
from tvm.contrib import coreml_runtime, graph_executor, utils, xcode
from tvm.contrib.download import download_testdata
from tvm.contrib.target import coreml as _coreml
from tvm.relay import transform
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.quantize.quantize import prerequisite_optimize

# Change target configuration, this is setting for iphone6s
# arch = "x86_64"
# sdk = "iphonesimulator"
arch = "arm64"
sdk = "iphoneos"
target_host = "llvm -mtriple=%s-apple-darwin" % arch

MODES = {"proxy": rpc.connect, "tracker": rpc.connect_tracker, "standalone": rpc.connect}


# override metal compiler to compile to iphone
@tvm.register_func("tvm_callback_metal_compile")
def compile_metal(src, target):
    return xcode.compile_metal(src, sdk=sdk)


def prepare_input():
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_name = "cat.png"
    synset_url = "".join(
        [
            "https://gist.githubusercontent.com/zhreshold/",
            "4d0b62f3d01426887599d4f7ede23ee5/raw/",
            "596b27d23537e5a1b5751d2b0481ef172f58b539/",
            "imagenet1000_clsid_to_human.txt",
        ]
    )
    synset_name = "imagenet1000_clsid_to_human.txt"
    img_path = download_testdata(img_url, "cat.png", module="data")
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        synset = eval(f.read())
        image = Image.open(img_path).resize((224, 224))

    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image.astype("float32"), synset


def get_model(model_name, data_shape):
    gluon_model = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
    mod, params = relay.frontend.from_mxnet(gluon_model, {"data": data_shape})
    # we want a probability so add a softmax operator
    func = mod["main"]
    func = relay.Function(
        func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs
    )

    return func, params


def test_mobilenet(host, port, key, mode):
    temp = utils.tempdir()
    image, synset = prepare_input()
    model, params = get_model("mobilenetv2_1.0", image.shape)

    def run(mod, target):
        with relay.build_config(opt_level=3):
            lib = relay.build(
                mod, target=tvm.target.Target(target, host=target_host), params=params
            )
        path_dso = temp.relpath("deploy.dylib")
        lib.export_library(path_dso, fcompile=xcode.create_dylib, arch=arch, sdk=sdk)

        # connect to the proxy
        if mode == "tracker":
            remote = MODES[mode](host, port).request(key)
        else:
            remote = MODES[mode](host, port, key=key)
        remote.upload(path_dso)

        if target == "metal":
            dev = remote.metal(0)
        else:
            dev = remote.cpu(0)
        lib = remote.load_module("deploy.dylib")
        m = graph_executor.GraphModule(lib["default"](dev))

        m.set_input("data", tvm.nd.array(image, dev))
        m.run()
        tvm_output = m.get_output(0)
        top1 = np.argmax(tvm_output.numpy()[0])
        print("TVM prediction top-1:", top1, synset[top1])

        # evaluate
        ftimer = m.module.time_evaluator("run", dev, number=3, repeat=10)
        prof_res = np.array(ftimer().results) * 1000
        print("%-19s (%s)" % ("%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))

    def annotate(func, compiler):
        """
        An annotator for Core ML.
        """
        # Bind free variables to the constant values.
        bind_dict = {}
        for arg in func.params:
            name = arg.name_hint
            if name in params:
                bind_dict[arg] = relay.const(params[name])

        func = relay.bind(func, bind_dict)

        # Annotate the entire graph for Core ML
        mod = tvm.IRModule()
        mod["main"] = func

        seq = tvm.transform.Sequential(
            [
                transform.SimplifyInference(),
                transform.FoldConstant(),
                transform.FoldScaleAxis(),
                transform.AnnotateTarget(compiler),
                transform.MergeCompilerRegions(),
                transform.PartitionGraph(),
            ]
        )

        with relay.build_config(opt_level=3):
            mod = seq(mod)

        return mod

    # CPU
    run(model, target_host)
    # Metal
    run(model, "metal")
    # CoreML
    run(annotate(model, "coremlcompiler"), target_host)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo app demonstrates how ios_rpc works.")
    parser.add_argument("--host", required=True, type=str, help="Adress of rpc server")
    parser.add_argument("--port", type=int, default=9090, help="rpc port (default: 9090)")
    parser.add_argument("--key", type=str, default="iphone", help="device key (default: iphone)")
    parser.add_argument(
        "--mode",
        type=str,
        default="tracker",
        help="type of RPC connection (default: tracker), possible values: {}".format(
            ", ".join(MODES.keys())
        ),
    )

    args = parser.parse_args()
    assert args.mode in MODES.keys()
    test_mobilenet(args.host, args.port, args.key, args.mode)
