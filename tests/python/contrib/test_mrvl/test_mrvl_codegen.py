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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, W0611, C0413

"""MRVL MLIP codegen tests"""

import sys
import os
import numpy as np
import pytest
import logging

logging.basicConfig(level=logging.DEBUG)
mylogger = logging.getLogger()

import tvm
import tvm.relay.testing
from tvm import testing
from tvm import relay

from test_mrvl.infrastructure import skip_json_codegen_test, skip_aot_runtime_test
from test_mrvl.infrastructure import aot_build_and_json_codegen, verify_json_codegen


def _get_single_random_data_inp(name, ishape, dtype):
    data = {}
    np.random.seed(0)

    if dtype == "uint8":
        low, high = 0, 255
    else:
        low, high = -127, 128
    data[name] = np.random.uniform(low, high, ishape).astype(dtype)
    return data


def _get_single_input_mxnet_model_and_data_inp(mxnet_model, input_info):
    try:
        from gluoncv import data
    except ImportError:
        pytest.skip("Missing Gluoncv Package")

    """Convert Mxnet graph to relay."""
    (ishape, dtype, layout, im_fname, short) = input_info
    # FIXME: can't use mxnet_model.input_names[0]
    name = "data"
    inputs = {name: ishape}
    mod, params = relay.frontend.from_mxnet(mxnet_model, inputs)

    # we pre-process input for the NN model
    # FIXME: to force data values of input_x to be inside range: [low=-127, high=128]
    input_x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
    data_inp = {}
    data_inp[name] = input_x
    return mod, params, data_inp


def _get_single_input_keras_model_and_random_data_inp(keras_model, input_info):
    """Convert Keras graph to relay."""
    (ishape, dtype, layout) = input_info
    name = keras_model.input_names[0]
    inputs = {}
    inputs[name] = ishape
    mod, params = relay.frontend.from_keras(keras_model, inputs, layout=layout)
    data_inp = _get_single_random_data_inp(name, ishape, dtype)
    return mod, params, data_inp


def _exec_unix_cmd(os_cmd, verbose_prefix=""):
    mylogger.info(f"Debug: {verbose_prefix}run cmd: {os_cmd}")
    os.system(os_cmd)


def _aot_json_codegen_and_fp16_run_for_network(
    model_name,
    mod,
    params,
    data_inp,
    model_verification_info={},
    json_codegen_only=False,
):
    """Helper function to build and run a network."""
    mylogger.info(f"\nDebug: in _aot_json_codegen_and_fp16_run_for_network:\n{mod.astext(False)}")
    if skip_json_codegen_test():
        return

    my_cwd = os.getcwd()
    mylogger.info(f"\nDebug: cwd: {my_cwd}")
    working_dir = f"{my_cwd}/test_mrvl_{model_name}/"
    _exec_unix_cmd(f"rm -rf {working_dir}")
    _exec_unix_cmd(f"mkdir -p {working_dir}")
    (
        nodes_json_filename,
        consts_json_filename,
        mod_mrvl_subgraph,
        mod_non_mrvl_subgraph,
        mrvl_layers_in_mrvl_subgraph,
        mrvl_layers_in_non_mrvl_subgraph,
    ) = aot_build_and_json_codegen(
        model_name,
        working_dir,
        mod,
        params,
    )
    mylogger.info(f"\nDebug: cwd: {model_verification_info}")
    verify_json_codegen(nodes_json_filename, model_verification_info=model_verification_info)

    if json_codegen_only:
        print("aot json codegen only")
        return

    # check whether a Mrvl distribution package has been installed
    if skip_aot_runtime_test():
        return

    # TODO(ccjoechou): add final code for fp16-fp32-mixed inf run and then
    #   uncomment following calls when they become available
    # mrvl_subgraph_runtime_model_binary = aot_runtime_gen(
    #     nodes_json_filename, consts_json_filename, aot_fp16_cmd_opts,
    # )
    # mrvl_subgraph_actual_fp16_output = aot_run(
    #     mrvl_subgraph_runtime_model_binary, aot_run_cmd_opts, inf_inp=data_inp,
    # )
    # mrvl_subgraph_golden_fp32_output = tvm_llvm_fp32_run(
    #     mod_mrvl_subgraph, mrvl_layers_in_mrvl_subgraph, inf_inp=data_inp,
    # )
    # verify_mrvl_subgraph_aot_inf_result(
    #     mrvl_subgraph_actual_fp16_output,
    #     mrvl_subgraph_golden_fp32_output,
    #     delta_config,
    # )
    # actual_inf_output = tvm_llvm_fp32_run(
    #     mod_non_mrvl_subgraph,
    #     mrvl_layers_in_non_mrvl_subgraph,
    #     inf_inp=mrvl_subgraph_actual_fp16_output,
    # )
    # verify_aot_inf_result(actual_inf_output, delta_config)


# TODO(ccjoechou): re-enable this test after a Mrvl BYOC bug can be resolved
@pytest.mark.skipif(True, reason="Skip test_relay_resnet18_aot_json_codegen() for now")
def test_relay_resnet18_aot_json_codegen():
    """Mrvl MLIP codegen (to JSON files) with ResNet18 model"""

    def get_model(dtype):
        model_name = "resnet18"
        ishape = (1, 3, 224, 224)
        layout = "NCHW"
        input_info = (ishape, layout)
        mod, params = relay.testing.resnet.get_workload(num_layers=18, batch_size=1)
        name = "data"
        data_inp = _get_single_random_data_inp(name, ishape, dtype)
        return model_name, mod, params, data_inp

    dtype = "float32"
    assert dtype == "float32"
    _aot_json_codegen_and_fp16_run_for_network(
        *get_model(dtype),
        model_verification_info={},
        json_codegen_only=True,
    )


def test_ssd_resnet50_aot_json_codegen():
    """Mrvl MLIP codegen (to JSON files) with SSD-ResNet50 model"""

    def get_model(dtype):
        try:
            from gluoncv import model_zoo
        except ImportError:
            pytest.skip("Missing Gluoncv Package")

        model_name = "ssd_512_resnet50_v1_voc"
        ssd_resnet50 = model_zoo.get_model(model_name, pretrained=True)
        short = 512
        ishape = (1, 3, short, short)
        layout = "NCHW"
        # we will use the street_small.jpg image as the raw input tensor
        im_fname = tvm.contrib.download.download_testdata(
            "https://github.com/dmlc/web-data/blob/main/"
            + "gluoncv/detection/street_small.jpg?raw=true",
            "street_small.jpg",
        )
        input_info = (ishape, dtype, layout, im_fname, short)
        (mod, params, data_inp) = _get_single_input_mxnet_model_and_data_inp(
            ssd_resnet50, input_info
        )
        return model_name, mod, params, data_inp

    # setup per-test verification info to be checked
    model_verification_info = {}
    model_verification_info["nodes_size"] = 104
    model_verification_info["heads_size"] = 18

    dtype = "float32"
    assert dtype == "float32"
    _aot_json_codegen_and_fp16_run_for_network(
        *get_model(dtype),
        model_verification_info=model_verification_info,
        json_codegen_only=True,
    )


# TODO(ccjoechou): re-enable this test after either (1) relay Keras frontend can also support
#   data_format = channels_first (currently relay Keras frontend supports
#   data_format = channels_last only); or (2) Mrvl BYOC backend can support NHWC as
#   the input data format (currently, Mrvl BYOC backend supports only NCHW format)
@pytest.mark.skipif(True, reason="Skip test_mobilenet_aot_json_codegen() for now")
def test_mobilenet_aot_json_codegen():
    """Mrvl MLIP codegen (to JSON files) with MobileNet model"""

    def get_model(dtype):
        try:
            from tensorflow.python.keras import backend_config
            from keras.applications import MobileNetV2

            backend_config.set_image_data_format("channels_first")
        except ImportError:
            pytest.skip("Missing keras module")

        mobilenet = MobileNetV2(
            include_top=True, weights="imagenet", input_shape=(3, 224, 224), classes=1000
        )
        model_name = "mobilenet"
        ishape = (1, 3, 224, 224)
        layout = "NCHW"
        input_info = (ishape, dtype, layout)
        mod, params, data_inp = _get_single_input_keras_model_and_random_data_inp(
            mobilenet, input_info
        )
        return model_name, mod, params, data_inp

    dtype = "float32"
    assert dtype == "float32"
    _aot_json_codegen_and_fp16_run_for_network(
        *get_model(dtype),
        model_verification_info={},
        json_codegen_only=True,
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        sys.exit(0)
    pytest.main([__file__])
