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

import os
import sys
import pytest
import numpy as np

import tvm.testing
from tvm import te
from tvm import relay
from tvm.relay.backend import Executor, Runtime

from .conftest import requires_hexagon_toolchain


@requires_hexagon_toolchain
def test_mobilenet(hexagon_session):
    import onnx

    dtype = "float32"
    model_url = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
    model_path = tvm.contrib.download.download_testdata(
        model_url, "mobilenetv2-7.onnx", module="onnx"
    )
    onnx_model = onnx.load(model_path)

    target_hexagon = tvm.target.hexagon("v68")
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
            tvm.target.Target(target_hexagon, host=target_hexagon),
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


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
