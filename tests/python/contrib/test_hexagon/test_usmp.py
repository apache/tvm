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

"""USMP tests"""

import numpy as np
import pytest
import tvm.testing
from tvm import relay
from tvm.contrib.hexagon.session import Session
from tvm.relay.backend import Executor, Runtime
from tvm.testing.usmp import is_tvm_backendallocworkspace_calls


@pytest.mark.parametrize("usmp_enabled", [False, True])
@tvm.testing.requires_hexagon
def test_conv2d(hexagon_session: Session, aot_host_target, aot_target, usmp_enabled):
    """Try conv2d on AOT target with usmp_enabled and check for TVMBackendAllocWorkspace calls"""
    dtype = "float32"
    input_shape = (1, 8, 8, 3)
    w1_shape = (5, 5, 3, 1)
    w2_shape = (5, 5, 1, 3)
    data = relay.var("data", relay.TensorType(input_shape, dtype))
    weight1 = relay.var("weight1", relay.TensorType(w1_shape, dtype))
    weight2 = relay.var("weight2", relay.TensorType(w2_shape, dtype))
    outpu1 = relay.nn.conv2d(
        data,
        weight1,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    output2 = relay.nn.conv2d(
        outpu1,
        weight2,
        padding=(2, 2),
        kernel_size=(5, 5),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="float32",
    )
    f = relay.Function([data, weight1, weight2], output2)
    relay_mod = tvm.IRModule.from_expr(f)
    relay_mod = relay.transform.InferType()(relay_mod)

    weight1_data = np.random.rand(w1_shape[0], w1_shape[1], w1_shape[2], w1_shape[3]).astype(
        dtype=dtype
    )
    weight2_data = np.random.rand(w2_shape[0], w2_shape[1], w2_shape[2], w2_shape[3]).astype(
        dtype=dtype
    )
    input_data = np.random.rand(
        input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    ).astype(dtype=dtype)

    params = {"weight1": weight1_data, "weight2": weight2_data}
    inputs = {"data": input_data}

    with tvm.transform.PassContext(opt_level=3, config={"tir.usmp.enable": usmp_enabled}):
        lowered = tvm.relay.build(
            relay_mod,
            params=params,
            target=tvm.target.Target(aot_target, host=aot_host_target),
            runtime=Runtime("cpp"),
            executor=Executor("aot", {"unpacked-api": False, "interface-api": "packed"}),
        )

    assert is_tvm_backendallocworkspace_calls(lowered.lib) != usmp_enabled

    aot_mod = hexagon_session.get_executor_from_factory(lowered)
    aot_mod.set_input(**inputs)
    aot_mod.run()
    hexagon_output = aot_mod.get_output(0).numpy()

    target_llvm = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            relay_mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            runtime=Runtime("cpp"),
            executor=Executor("aot"),
        )

    llvm_mod = tvm.runtime.executor.AotModule(llvm_lowered["default"](tvm.cpu(0)))
    llvm_mod.set_input(**params)
    llvm_mod.run(**inputs)
    expected_output = llvm_mod.get_output(0).numpy()

    tvm.testing.assert_allclose(hexagon_output, expected_output, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
