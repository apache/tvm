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
"""BNNS pattern detection check"""

import pytest

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.contrib import utils, graph_executor
from tvm.contrib.download import download_testdata
from tvm.relay.op.contrib.bnns import partition_for_bnns

import numpy as np

pytest.importorskip("onnx")

bnns_is_absent = tvm.get_global_func("relay.ext.bnns", True) is None

TARGET = "llvm"
INPUT_SHAPE = [1, 3, 224, 224]

BASE_MODEL_URL = "https://github.com/onnx/models/raw/bd206494e8b6a27b25e5cf7199dbcdbfe9d05d1c/"
MODEL_URL_COLLECTION = {
    "BERT": "text/machine_comprehension/bert-squad/model/bertsquad-10.onnx",
    "MobileNet-v2": "vision/classification/mobilenet/model/mobilenetv2-7.onnx",
    "ResNet50-v1": "vision/classification/resnet/model/resnet50-v1-7.onnx",
    "ResNet50-v2": "vision/classification/resnet/model/resnet50-v2-7.onnx",
    "SqueezeNet-v1.1": "vision/classification/squeezenet/model/squeezenet1.1-7.onnx",
    "SqueezeNet-v1.0": "vision/classification/squeezenet/model/squeezenet1.0-7.onnx",
    "Inception-v1": "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-7.onnx",
    "Inception-v2": "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-7.onnx",
}


def get_onnx_input_name(model):
    inputs = [node.name for node in model.graph.input]
    initializer = [node.name for node in model.graph.initializer]

    inputs = list(set(inputs) - set(initializer))
    return inputs


def get_model_url(model_name):
    return BASE_MODEL_URL + MODEL_URL_COLLECTION[model_name]


def get_name_from_url(url):
    return url[url.rfind("/") + 1 :].strip()


def find_of_download(model_name):
    model_url = get_model_url(model_name)
    model_file_name = get_name_from_url(model_url)
    return download_testdata(model_url, model_file_name, module="models")


def get_model(model_name):
    model_path = find_of_download(model_name)
    onnx_model = onnx.load(model_path)
    input_names = get_onnx_input_name(onnx_model)
    input_dict = {}
    for name in input_names:
        input_dict[name] = INPUT_SHAPE  # TODO: hardcode
    mod, params = relay.frontend.from_onnx(onnx_model, input_dict, freeze_params=True)
    return mod, params, input_dict


def simplify_model(mod):
    """
    Simplify execution graph

    At least merge BatchNorm into convolution. For this purpose decompose BN primitive
    into simple operation which can be calculated as const expr and after that merged
    into nearest conv/dense primitive.
    """
    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.FoldConstant(),
            transform.SimplifyInference(),
            transform.FoldScaleAxis(),
        ]
    )
    return seq(mod)


def process(model_name):
    temp = utils.tempdir()
    model, params, input_dict = get_model(model_name)

    def run(mod, target, simplify=True, with_bnns=False):
        with tvm.transform.PassContext(opt_level=3):
            if simplify:
                mod = simplify_model(mod)
            if with_bnns:
                mod = partition_for_bnns(mod)
            graph_module = relay.build(mod, target=target, params=params)

        lib_name = "deploy.tar"
        path_dso = temp.relpath(lib_name)
        graph_module.export_library(path_dso)

        dev = tvm.cpu(0)
        loaded_lib = tvm.runtime.load_module(path_dso)

        module = graph_executor.GraphModule(loaded_lib["default"](dev))
        module.run()
        return module.get_output(0).numpy()

    res_llvm = run(model, TARGET, simplify=True, with_bnns=False)
    res_bnns = run(model, TARGET, simplify=True, with_bnns=True)

    tvm.testing.assert_allclose(
        res_llvm,
        res_bnns,
        atol=0.002,
        rtol=0.007,
    )


@pytest.mark.skip(reason="Manually disabled because of huge complexity")
@pytest.mark.skipif(bnns_is_absent, reason="BNNS runtime is absent")
@pytest.mark.parametrize("model_name", MODEL_URL_COLLECTION.keys())
def test_topology(model_name):
    process(model_name)
