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


def allclose_with_rounding(a, b):
    # Find number of mismatches in inputs.
    mismatch = a != b
    # Allow some rounding errors due to GPU fp32 arithmetic.
    assert np.sum(mismatch) <= 3


def quantize_test_driver(in_dtype, quant_args, axis, out_dtype, in_data):
    shape = in_data.shape
    input_data = relay.var("input_data", shape=shape, dtype=in_dtype)
    output_zero_point = relay.const(quant_args["out_zero_point"])
    output_scale = relay.const(quant_args["out_scale"])
    quantized_output = relay.qnn.quantize(
        input_data,
        output_scale=output_scale,
        output_zero_point=output_zero_point,
        axis=axis,
        out_dtype=out_dtype,
    )
    mod = relay.Function(relay.analysis.free_vars(quantized_output), quantized_output)
    mod = tvm.IRModule.from_expr(mod)
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod, "llvm", params=None)
    rt_mod = graph_executor.create(graph, lib, device=tvm.cpu(0))
    rt_mod.set_input(input_data=in_data)
    rt_mod.set_input(**params)
    rt_mod.run()
    res = rt_mod.get_output(0).numpy()
    return res


def build_simulated_quantize(input_data, scale, zp, dtype, axis=-1):
    sim_q = relay.qnn.simulated_quantize(
        input_data,
        scale,
        zp,
        axis=axis,
        out_dtype=dtype,
    )
    mod = tvm.IRModule.from_expr(sim_q)
    with tvm.transform.PassContext(opt_level=3):
        vm_exec = relay.vm.compile(mod, "llvm", params=None)
    vm = VirtualMachine(vm_exec, tvm.cpu(0))
    return vm


def verify_simulated_quantize_simple(dtype):
    data = np.random.uniform(low=-128, high=127, size=[2, 5]).astype("float32")
    scale_np = np.float32(0.5)
    zp_np = np.int32(127)
    dtype_np = np.int32(SQNN_DTYPE_TO_CODE[dtype])
    quant_args = {"out_zero_point": zp_np, "out_scale": scale_np}
    q_out = quantize_test_driver(
        in_dtype="float32",
        quant_args=quant_args,
        axis=-1,
        out_dtype=dtype,
        in_data=data,
    )
    input_data = relay.var("input_data", shape=data.shape, dtype="float32")
    scale = relay.var("scale", shape=[])
    zp = relay.var("zp", shape=[], dtype="int32")
    dtype = relay.var("dtype", shape=[], dtype="int32")
    vm = build_simulated_quantize(input_data, scale, zp, dtype)
    sim_q_out = vm.invoke("main", input_data=data, scale=scale_np, zp=zp_np, dtype=dtype_np)
    allclose_with_rounding(sim_q_out.numpy(), q_out)


def test_simulated_quantize():
    verify_simulated_quantize_simple("uint8")
    verify_simulated_quantize_simple("int8")
    verify_simulated_quantize_simple("int32")


def test_dynamic_channels():
    # Compile simulated quantize once but support either per-channel or scalar params.
    data = np.random.uniform(low=-64, high=64, size=[2, 5]).astype("float32")
    # Test scalar qnn params.
    scale_np = np.asarray([0.5]).astype("float32")
    zp_np = np.asarray([127]).astype("int32")
    dtype_np = np.int32(SQNN_DTYPE_TO_CODE["uint8"])
    quant_args = {"out_zero_point": zp_np[0], "out_scale": scale_np[0]}
    q_out = quantize_test_driver(
        in_dtype="float32",
        quant_args=quant_args,
        axis=0,
        out_dtype="uint8",
        in_data=data,
    )
    # Create variables with undefined shape and run with scalar inputs.
    input_data = relay.var("input_data", shape=data.shape, dtype="float32")
    scale = relay.var("scale", shape=[relay.Any()], dtype="float32")
    zp = relay.var("zp", shape=[relay.Any()], dtype="int32")
    dtype = relay.var("dtype", shape=[], dtype="int32")
    vm = build_simulated_quantize(input_data, scale, zp, dtype, axis=0)
    sim_q_out = vm.invoke("main", input_data=data, scale=scale_np, zp=zp_np, dtype=dtype_np)
    allclose_with_rounding(sim_q_out.numpy(), q_out)

    # Now get the perchannel quantize output and compare without recompiling.
    scale_np = np.array([0.5, 0.25]).astype("float32")
    zp_np = np.array([127, 123]).astype("int32")

    # Get the reference quantize output.
    quant_args = {"out_zero_point": zp_np, "out_scale": scale_np}
    q_out = quantize_test_driver(
        in_dtype="float32",
        quant_args=quant_args,
        axis=0,
        out_dtype="uint8",
        in_data=data,
    )
    # Run the simulated quantize without recompiling and confirm results match.
    sim_q_out = vm.invoke("main", input_data=data, scale=scale_np, zp=zp_np, dtype=dtype_np)
    allclose_with_rounding(sim_q_out.numpy(), q_out)


def test_dynamic_dtype():
    # Compile simulated quantize once but support any type of quantization.
    data = np.random.uniform(low=-64, high=64, size=[2, 5]).astype("float32")
    # Test scalar float32 to uint8.
    scale_np = np.asarray([0.5]).astype("float32")
    zp_np = np.asarray([127]).astype("int32")
    dtype_np = np.int32(SQNN_DTYPE_TO_CODE["uint8"])
    quant_args = {"out_zero_point": zp_np[0], "out_scale": scale_np[0]}
    q_out = quantize_test_driver(
        in_dtype="float32",
        quant_args=quant_args,
        axis=-1,
        out_dtype="uint8",
        in_data=data,
    )
    # Create variables with undefined shape and run with scalar inputs.
    input_data = relay.var("input_data", shape=data.shape, dtype="float32")
    scale = relay.var("scale", shape=[relay.Any()], dtype="float32")
    zp = relay.var("zp", shape=[relay.Any()], dtype="int32")
    dtype = relay.var("dtype", shape=[], dtype="int32")
    vm = build_simulated_quantize(input_data, scale, zp, dtype)
    sim_q_out = vm.invoke("main", input_data=data, scale=scale_np, zp=zp_np, dtype=dtype_np)
    allclose_with_rounding(sim_q_out.numpy(), q_out)

    # Now test float32 to int32 compilation.
    # Get the reference quantize output.
    q_out = quantize_test_driver(
        in_dtype="float32",
        quant_args=quant_args,
        axis=-1,
        out_dtype="int32",
        in_data=data,
    )
    # Run the simulated quantize without recompiling and confirm results match.
    dtype_np = np.int32(SQNN_DTYPE_TO_CODE["int32"])
    sim_q_out = vm.invoke("main", input_data=data, scale=scale_np, zp=zp_np, dtype=dtype_np)
    allclose_with_rounding(sim_q_out.numpy(), q_out)


if __name__ == "__main__":
    test_simulated_quantize()
    test_dynamic_channels()
    test_dynamic_dtype()
