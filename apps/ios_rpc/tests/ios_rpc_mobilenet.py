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

import tvm
from tvm import rpc, relay
from tvm.contrib.download import download_testdata
from tvm.relay.expr_functor import ExprMutator
from tvm.relay import transform
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.quantize.quantize import prerequisite_optimize
from tvm.contrib import utils, xcode, graph_runtime, coreml_runtime
from tvm.contrib.target import coreml as _coreml

import os
import re
import sys
import numpy as np
from mxnet import gluon
from PIL import Image
import coremltools

# Set to be address of tvm proxy.
proxy_host = os.environ["TVM_IOS_RPC_PROXY_HOST"]
# Set your desination via env variable.
# Should in format "platform=iOS,id=<the test device uuid>"
destination = os.environ["TVM_IOS_RPC_DESTINATION"]

if not re.match(r"^platform=.*,id=.*$", destination):
    print("Bad format: {}".format(destination))
    print("Example of expected string: platform=iOS,id=1234567890abcabcabcabc1234567890abcabcab")
    sys.exit(1)

proxy_port = 9090
key = "iphone"

# Change target configuration, this is setting for iphone6s
# arch = "x86_64"
# sdk = "iphonesimulator"
arch = "arm64"
sdk = "iphoneos"
target_host = "llvm -mtriple=%s-apple-darwin" % arch

# override metal compiler to compile to iphone
@tvm.register_func("tvm_callback_metal_compile")
def compile_metal(src):
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


def test_mobilenet():
    temp = utils.tempdir()
    image, synset = prepare_input()
    model, params = get_model("mobilenetv2_1.0", image.shape)

    def run(mod, target):
        with relay.build_config(opt_level=3):
            lib = relay.build(mod, target=target, target_host=target_host, params=params)
        path_dso = temp.relpath("deploy.dylib")
        lib.export_library(path_dso, xcode.create_dylib, arch=arch, sdk=sdk)
        xcode.codesign(path_dso)

        # Start RPC test server that contains the compiled library.
        xcode.popen_test_rpc(proxy_host, proxy_port, key, destination=destination, libs=[path_dso])

        # connect to the proxy
        remote = rpc.connect(proxy_host, proxy_port, key=key)

        if target == "metal":
            ctx = remote.metal(0)
        else:
            ctx = remote.cpu(0)
        lib = remote.load_module("deploy.dylib")
        m = graph_runtime.GraphModule(lib["default"](ctx))

        m.set_input("data", tvm.nd.array(image, ctx))
        m.run()
        tvm_output = m.get_output(0)
        top1 = np.argmax(tvm_output.asnumpy()[0])
        print("TVM prediction top-1:", top1, synset[top1])

        # evaluate
        ftimer = m.module.time_evaluator("run", ctx, number=3, repeat=10)
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
    test_mobilenet()
