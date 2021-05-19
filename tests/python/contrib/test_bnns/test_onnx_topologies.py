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

import onnx
import pytest
import itertools
import numpy as np

import tvm
import tvm.testing
from tvm import relay
from tvm.relay import transform
from tvm.contrib.download import download_testdata

from .infrastructure import (
    Device,
    skip_complexity_test,
    get_run_modes,
    check_test_parameters,
    build_and_run,
    verify
)


DTYPE = 'float32'
INPUT_SHAPE = [1, 3, 224, 224]

BASE_MODEL_URL = "https://github.com/onnx/models/raw/master/"
MODEL_URL_COLLECTION = {
    # "BERT": "text/machine_comprehension/bert-squad/model/bertsquad-10.onnx",
    "MobileNet-v2": "vision/classification/mobilenet/model/mobilenetv2-7.onnx",
    "ResNet50-v1": "vision/classification/resnet/model/resnet50-v1-7.onnx",
    "ResNet50-v2": "vision/classification/resnet/model/resnet50-v2-7.onnx",
    # "SqueezeNet-v1.1": "vision/classification/squeezenet/model/squeezenet1.1-7.onnx",
    # "SqueezeNet-v1.0": "vision/classification/squeezenet/model/squeezenet1.0-7.onnx",
    # "Inception-v1": "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-7.onnx",
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
    return url[url.rfind("/") + 1:].strip()


def find_of_download(model_name):
    model_url = get_model_url(model_name)
    model_file_name = get_name_from_url(model_url)
    return download_testdata(model_url, model_file_name, module="models")


def get_model(model_name):
    model_path = find_of_download(model_name)
    onnx_model = onnx.load(model_path)
    input_names = get_onnx_input_name(onnx_model)
    input_shape_dict = {}
    input_dict = {}
    for name in input_names:
        input_shape_dict[name] = INPUT_SHAPE  # TODO: hardcode
        input_dict[name] = tvm.nd.array(np.random.uniform(0, 127, INPUT_SHAPE).astype(DTYPE))
    mod, params = relay.frontend.from_onnx(onnx_model, input_shape_dict, freeze_params=True)
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


def process(mode, model_name):
    check_test_parameters(mode)

    device = Device(mode)
    model, params, input_dict = get_model(model_name)
    with tvm.transform.PassContext(opt_level=3):
        model = simplify_model(model)

    outputs = []
    for enable_bnns in [False, True]:
        outputs.append(
            build_and_run(
                model,
                input_dict,
                1,
                params,
                device,
                enable_bnns=enable_bnns,
            )[0]
        )

    verify(outputs, atol=0.002, rtol=0.007)


@pytest.mark.parametrize("model_name", MODEL_URL_COLLECTION.keys())
@pytest.mark.parametrize("mode", get_run_modes())
@skip_complexity_test
def test_topology(mode, model_name):
    process(mode, model_name)


def main():
    for mode, model_name in itertools.product(get_run_modes(), MODEL_URL_COLLECTION.keys()):
        test_topology(mode, model_name)


if __name__ == "__main__":
    main()
