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
from tvm import te
import numpy as np
from tvm import relay
from tvm.contrib import graph_executor
from tvm.runtime.vm import VirtualMachine
from tvm.topi.nn.qnn import SQNN_DTYPE_TO_CODE


def dequantize_test_driver(in_dtype, quant_args, axis, in_data):
    shape = in_data.shape
    input_data = relay.var("input_data", shape=shape, dtype=in_dtype)
    input_zero_point = relay.const(quant_args["in_zero_point"])
    input_scale = relay.const(quant_args["in_scale"])
    dequantized_output = relay.qnn.dequantize(
        input_data,
        input_scale=input_scale,
        input_zero_point=input_zero_point,
        axis=axis,
    )
    mod = relay.Function(relay.analysis.free_vars(dequantized_output), dequantized_output)
    mod = tvm.IRModule.from_expr(mod)
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod, "llvm", params=None)
    rt_mod = graph_executor.create(graph, lib, device=tvm.cpu(0))
    rt_mod.set_input(input_data=in_data)
    rt_mod.set_input(**params)
    rt_mod.run()
    res = rt_mod.get_output(0).numpy()
    return res


def build_simulated_dequantize(input_data, scale, zp, dtype, axis=-1):
    sim_q = relay.qnn.simulated_dequantize(
        input_data,
        scale,
        zp,
        axis=axis,
        in_dtype=dtype,
    )
    mod = tvm.IRModule.from_expr(sim_q)
    with tvm.transform.PassContext(opt_level=3):
        vm_exec = relay.vm.compile(mod, "llvm", params=None)
    vm = VirtualMachine(vm_exec, tvm.cpu(0))
    return vm


def verify_simulated_dequantize_simple(dtype):
    data = np.random.uniform(low=-128, high=127, size=[2, 5]).astype(dtype)
    data_fp = data.astype("float32")
    scale_np = np.float32(0.5)
    zp_np = np.int32(127)
    dtype_np = np.int32(SQNN_DTYPE_TO_CODE[dtype])
    quant_args = {"in_zero_point": zp_np, "in_scale": scale_np}
    dq_out = dequantize_test_driver(
        in_dtype=dtype,
        quant_args=quant_args,
        axis=-1,
        in_data=data,
    )
    input_data = relay.var("input_data", shape=data.shape, dtype="float32")
    scale = relay.var("scale", shape=[])
    zp = relay.var("zp", shape=[], dtype="int32")
    dtype = relay.var("dtype", shape=[], dtype="int32")
    vm = build_simulated_dequantize(input_data, scale, zp, dtype)
    sim_dq_out = vm.invoke("main", input_data=data_fp, scale=scale_np, zp=zp_np, dtype=dtype_np)
    np.testing.assert_allclose(sim_dq_out.numpy(), dq_out, rtol=1e-5)


def test_simulated_dequantize():
    verify_simulated_dequantize_simple("uint8")
    verify_simulated_dequantize_simple("int8")
    verify_simulated_dequantize_simple("int32")


def test_dynamic_channels():
    # Compile simulated quantize once but support either per-channel or scalar params.
    data = np.random.uniform(low=-64, high=64, size=[2, 5]).astype("int8")
    data_fp = data.astype("float32")
    # Test scalar qnn params.
    scale_np = np.asarray([0.5]).astype("float32")
    zp_np = np.asarray([0]).astype("int32")
    dtype_np = np.int32(SQNN_DTYPE_TO_CODE["int8"])
    quant_args = {"in_zero_point": zp_np[0], "in_scale": scale_np[0]}
    dq_out = dequantize_test_driver(
        in_dtype="int8",
        quant_args=quant_args,
        axis=0,
        in_data=data,
    )
    # Create variables with undefined shape and run with scalar inputs.
    input_data = relay.var("input_data", shape=data.shape, dtype="float32")
    scale = relay.var("scale", shape=[relay.Any()], dtype="float32")
    zp = relay.var("zp", shape=[relay.Any()], dtype="int32")
    dtype = relay.var("dtype", shape=[], dtype="int32")
    vm = build_simulated_dequantize(input_data, scale, zp, dtype, axis=0)
    sim_dq_out = vm.invoke("main", input_data=data_fp, scale=scale_np, zp=zp_np, dtype=dtype_np)
    np.testing.assert_allclose(sim_dq_out.numpy(), dq_out, rtol=1e-5)

    # Now get the perchannel quantize output and compare without recompiling.
    scale_np = np.array([0.5, 0.25]).astype("float32")
    zp_np = np.array([127, 123]).astype("int32")

    # Get the reference quantize output.
    quant_args = {"in_zero_point": zp_np, "in_scale": scale_np}
    dq_out = dequantize_test_driver(
        in_dtype="int8",
        quant_args=quant_args,
        axis=0,
        in_data=data,
    )
    # Run the simulated quantize without recompiling and confirm results match.
    sim_dq_out = vm.invoke("main", input_data=data_fp, scale=scale_np, zp=zp_np, dtype=dtype_np)
    np.testing.assert_allclose(sim_dq_out.numpy(), dq_out, rtol=1e-5)


def test_dynamic_dtype():
    # Compile simulated quantize once but support any type of quantization.
    data = np.random.uniform(low=0, high=255, size=[2, 5]).astype("uint8")
    data_fp = data.astype("float32")
    # Test scalar uint8 to fp32.
    scale_np = np.asarray([0.5]).astype("float32")
    zp_np = np.asarray([127]).astype("int32")
    dtype_np = np.int32(SQNN_DTYPE_TO_CODE["uint8"])
    quant_args = {"in_zero_point": zp_np[0], "in_scale": scale_np[0]}
    dq_out = dequantize_test_driver(
        in_dtype="uint8",
        quant_args=quant_args,
        axis=-1,
        in_data=data,
    )
    # Create variables with undefined shape and run with scalar inputs.
    input_data = relay.var("input_data", shape=data.shape, dtype="float32")
    scale = relay.var("scale", shape=[relay.Any()], dtype="float32")
    zp = relay.var("zp", shape=[relay.Any()], dtype="int32")
    dtype = relay.var("dtype", shape=[], dtype="int32")
    vm = build_simulated_dequantize(input_data, scale, zp, dtype)
    sim_dq_out = vm.invoke("main", input_data=data_fp, scale=scale_np, zp=zp_np, dtype=dtype_np)
    np.testing.assert_allclose(sim_dq_out.numpy(), dq_out, rtol=1e-5)

    # Now test int8 to float32 compilation.
    data = np.random.uniform(low=0, high=255, size=[2, 5]).astype("int8")
    data_fp = data.astype("float32")
    # Get the reference quantize output.
    dq_out = dequantize_test_driver(
        in_dtype="int8",
        quant_args=quant_args,
        axis=-1,
        in_data=data,
    )
    # Run the simulated quantize without recompiling and confirm results match.
    dtype_np = np.int32(SQNN_DTYPE_TO_CODE["int8"])
    sim_dq_out = vm.invoke("main", input_data=data_fp, scale=scale_np, zp=zp_np, dtype=dtype_np)
    np.testing.assert_allclose(sim_dq_out.numpy(), dq_out, rtol=1e-5)


if __name__ == "__main__":
    test_simulated_dequantize()
    test_dynamic_channels()
    test_dynamic_dtype()
