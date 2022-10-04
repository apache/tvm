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

"""Test mobilenet model with both graph and aot executor"""

import numpy as np
import pytest

import tvm.testing
from tvm import relay
from tvm.contrib.hexagon.session import Session
from tvm.relay.backend import Executor, Runtime

from .infrastructure import get_hexagon_target


def get_mobilenet():
    """Download and import mobilenet model with ONNX"""
    import onnx  # pylint: disable=import-outside-toplevel

    model_url = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx"  # pylint: disable=line-too-long
    model_path = tvm.contrib.download.download_testdata(
        model_url, "mobilenetv2-7.onnx", module="onnx"
    )
    return onnx.load(model_path)


@tvm.testing.requires_hexagon
def test_mobilenet(hexagon_session: Session):
    """Test mobilenet with graph executor"""
    dtype = "float32"
    onnx_model = get_mobilenet()

    target_llvm = tvm.target.Target("llvm")
    runtime = Runtime("cpp")
    executor = Executor("graph", {"link-params": True})

    data_in = np.random.rand(1, 3, 224, 224).astype(dtype=dtype)

    input_name = "input"
    shape_dict = {input_name: data_in.shape}
    relay_mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    inputs = {input_name: data_in}

    with tvm.transform.PassContext(opt_level=3):
        hexagon_lowered = tvm.relay.build(
            relay_mod,
            get_hexagon_target("v68"),
            runtime=runtime,
            executor=executor,
            params=params,
        )

        llvm_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            runtime=runtime,
            executor=executor,
            params=params,
        )

    graph_mod = hexagon_session.get_executor_from_factory(hexagon_lowered)
    graph_mod.set_input(**inputs)
    graph_mod.run()
    hexagon_output = graph_mod.get_output(0).numpy()

    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_graph_mod.set_input(**inputs)
    llvm_graph_mod.run()
    expected_output = llvm_graph_mod.get_output(0).numpy()

    tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("enable_usmp", [False, True])
@tvm.testing.requires_hexagon
def test_mobilenet_aot(hexagon_session: Session, aot_host_target, aot_target, enable_usmp):
    """Test mobilenet with aot executor"""
    if hexagon_session._launcher._serial_number == "simulator":
        pytest.skip(msg="Skip on simulator due to long runtime.")

    dtype = "float32"
    onnx_model = get_mobilenet()

    data_in = np.random.rand(1, 3, 224, 224).astype(dtype=dtype)

    input_name = "input"
    shape_dict = {input_name: data_in.shape}
    relay_mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    inputs = {input_name: data_in}

    target_llvm = tvm.target.Target("llvm")
    config = {"tir.usmp.enable": enable_usmp}
    with tvm.transform.PassContext(opt_level=3, config=config):
        hexagon_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(aot_target, host=aot_host_target),
            runtime=Runtime("cpp"),
            executor=Executor("aot", {"unpacked-api": False, "interface-api": "packed"}),
            params=params,
        )

    aot_mod = hexagon_session.get_executor_from_factory(hexagon_lowered)
    aot_mod.set_input(**inputs)
    aot_mod.run()
    hexagon_output = aot_mod.get_output(0).numpy()

    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            runtime=Runtime("cpp"),
            executor=Executor("graph", {"link-params": True}),
            params=params,
        )

    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_graph_mod.set_input(**inputs)
    llvm_graph_mod.run()
    expected_output = llvm_graph_mod.get_output(0).numpy()

    tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
