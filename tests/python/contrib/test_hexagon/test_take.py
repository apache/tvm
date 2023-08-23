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

import numpy as np
from scipy import special

import tvm
import tvm.testing
from tvm import relax
from tvm.script import tir as T, relax as R
from tvm.contrib.hexagon import generate_take_op
from tvm.contrib.hexagon import hexagon_unary_ops

from .infrastructure import quantize_np


# Testing the structural and value correctness on replacing unary op with take op.


@tvm.script.ir_module
class Module_tanh:
    @R.function
    def main(
        input: R.Tensor((1, 2, 2, 2), dtype="uint8"),
    ) -> R.Tensor((1, 2, 2, 2), dtype="uint8"):
        out = R.call_tir(
            Module_tanh.tanh,
            (
                input,
                R.const(0.003186821002586215, "float32"),
                R.const(0, "int32"),
                R.const(0.002631544131858676, "float32"),
                R.const(0, "int32"),
            ),
            out_sinfo=R.Tensor((1, 2, 2, 2), dtype="uint8"),
        )
        return out

    @T.prim_func
    def tanh(
        rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
        rxplaceholder_1: T.Buffer((), "float32"),
        rxplaceholder_2: T.Buffer((), "int32"),
        rxplaceholder_3: T.Buffer((), "float32"),
        rxplaceholder_4: T.Buffer((), "int32"),
        compute: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
    ):
        T.func_attr({"tir.noalias": True})
        pass


@tvm.script.ir_module
class Module_sqrt:
    @R.function
    def main(
        input: R.Tensor((1, 2, 2, 2), dtype="uint8"),
    ) -> R.Tensor((1, 2, 2, 2), dtype="uint8"):
        out = R.call_tir(
            Module_sqrt.sqrt,
            (
                input,
                R.const(0.003186821002586215, "float32"),
                R.const(0, "int32"),
                R.const(0.003535157327728918, "float32"),
                R.const(0, "int32"),
            ),
            out_sinfo=R.Tensor((1, 2, 2, 2), dtype="uint8"),
        )
        return out

    @T.prim_func
    def sqrt(
        rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
        rxplaceholder_1: T.Buffer((), "float32"),
        rxplaceholder_2: T.Buffer((), "int32"),
        rxplaceholder_3: T.Buffer((), "float32"),
        rxplaceholder_4: T.Buffer((), "int32"),
        compute: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
    ):
        T.func_attr({"tir.noalias": True})
        pass


@tvm.script.ir_module
class Module_rsqrt:
    @R.function
    def main(
        input: R.Tensor((1, 2, 2, 2), dtype="uint8"),
    ) -> R.Tensor((1, 2, 2, 2), dtype="uint8"):
        out = R.call_tir(
            Module_rsqrt.rsqrt,
            (
                input,
                R.const(0.003186821002586215, "float32"),
                R.const(0, "int32"),
                R.const(0.008154160766635542, "float32"),
                R.const(0, "int32"),
            ),
            out_sinfo=R.Tensor((1, 2, 2, 2), dtype="uint8"),
        )
        return out

    @T.prim_func
    def rsqrt(
        rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
        rxplaceholder_1: T.Buffer((), "float32"),
        rxplaceholder_2: T.Buffer((), "int32"),
        rxplaceholder_3: T.Buffer((), "float32"),
        rxplaceholder_4: T.Buffer((), "int32"),
        compute: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
    ):
        T.func_attr({"tir.noalias": True})
        pass


@tvm.script.ir_module
class Module_exp:
    @R.function
    def main(
        input: R.Tensor((1, 2, 2, 2), dtype="uint8"),
    ) -> R.Tensor((1, 2, 2, 2), dtype="uint8"):
        out = R.call_tir(
            Module_exp.exp,
            (
                input,
                R.const(0.003186821002586215, "float32"),
                R.const(0, "int32"),
                R.const(0.008838622987079832, "float32"),
                R.const(0, "int32"),
            ),
            out_sinfo=R.Tensor((1, 2, 2, 2), dtype="uint8"),
        )
        return out

    @T.prim_func
    def exp(
        rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
        rxplaceholder_1: T.Buffer((), "float32"),
        rxplaceholder_2: T.Buffer((), "int32"),
        rxplaceholder_3: T.Buffer((), "float32"),
        rxplaceholder_4: T.Buffer((), "int32"),
        compute: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
    ):
        T.func_attr({"tir.noalias": True})
        pass


@tvm.script.ir_module
class Module_erf:
    @R.function
    def main(
        input: R.Tensor((1, 2, 2, 2), dtype="uint8"),
    ) -> R.Tensor((1, 2, 2, 2), dtype="uint8"):
        out = R.call_tir(
            Module_erf.erf,
            (
                input,
                R.const(0.003186821002586215, "float32"),
                R.const(0, "int32"),
                R.const(0.002939393251118067, "float32"),
                R.const(0, "int32"),
            ),
            out_sinfo=R.Tensor((1, 2, 2, 2), dtype="uint8"),
        )
        return out

    @T.prim_func
    def erf(
        rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
        rxplaceholder_1: T.Buffer((), "float32"),
        rxplaceholder_2: T.Buffer((), "int32"),
        rxplaceholder_3: T.Buffer((), "float32"),
        rxplaceholder_4: T.Buffer((), "int32"),
        compute: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
    ):
        T.func_attr({"tir.noalias": True})
        pass


@tvm.script.ir_module
class Module_sigmoid:
    @R.function
    def main(
        input: R.Tensor((1, 2, 2, 2), dtype="uint8"),
    ) -> R.Tensor((1, 2, 2, 2), dtype="uint8"):
        out = R.call_tir(
            Module_sigmoid.sigmoid,
            (
                input,
                R.const(0.003186821002586215, "float32"),
                R.const(0, "int32"),
                R.const(0.002631544131858676, "float32"),
                R.const(0, "int32"),
            ),
            out_sinfo=R.Tensor((1, 2, 2, 2), dtype="uint8"),
        )
        return out

    @T.prim_func
    def sigmoid(
        rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
        rxplaceholder_1: T.Buffer((), "float32"),
        rxplaceholder_2: T.Buffer((), "int32"),
        rxplaceholder_3: T.Buffer((), "float32"),
        rxplaceholder_4: T.Buffer((), "int32"),
        compute: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
    ):
        T.func_attr({"tir.noalias": True})
        pass


@tvm.script.ir_module
class Module_hardswish:
    @R.function
    def main(
        input: R.Tensor((1, 2, 2, 2), dtype="uint8"),
    ) -> R.Tensor((1, 2, 2, 2), dtype="uint8"):
        out = R.call_tir(
            Module_hardswish.hardswish,
            (
                input,
                R.const(0.003186821002586215, "float32"),
                R.const(0, "int32"),
                R.const(0.0020250332087720325, "float32"),
                R.const(0, "int32"),
            ),
            out_sinfo=R.Tensor((1, 2, 2, 2), dtype="uint8"),
        )
        return out

    @T.prim_func
    def hardswish(
        rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
        rxplaceholder_1: T.Buffer((), "float32"),
        rxplaceholder_2: T.Buffer((), "int32"),
        rxplaceholder_3: T.Buffer((), "float32"),
        rxplaceholder_4: T.Buffer((), "int32"),
        compute: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
    ):
        T.func_attr({"tir.noalias": True})
        pass


@tvm.script.ir_module
class Module_log:
    @R.function
    def main(
        input: R.Tensor((1, 2, 2, 2), dtype="uint8"),
    ) -> R.Tensor((1, 2, 2, 2), dtype="uint8"):
        out = R.call_tir(
            Module_log.log,
            (
                input,
                R.const(0.003186821002586215, "float32"),
                R.const(0, "int32"),
                R.const(0.0057414634248614226, "float32"),
                R.const(255, "int32"),
            ),
            out_sinfo=R.Tensor((1, 2, 2, 2), dtype="uint8"),
        )
        return out

    @T.prim_func
    def log(
        rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
        rxplaceholder_1: T.Buffer((), "float32"),
        rxplaceholder_2: T.Buffer((), "int32"),
        rxplaceholder_3: T.Buffer((), "float32"),
        rxplaceholder_4: T.Buffer((), "int32"),
        compute: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
    ):
        T.func_attr({"tir.noalias": True})
        pass


@tvm.script.ir_module
class Module_abs:
    @R.function
    def main(
        input: R.Tensor((1, 2, 2, 2), dtype="uint8"),
    ) -> R.Tensor((1, 2, 2, 2), dtype="uint8"):
        out = R.call_tir(
            Module_abs.abs,
            (
                input,
                R.const(0.003186821002586215, "float32"),
                R.const(0, "int32"),
                R.const(0.0031868210196078434, "float32"),
                R.const(0, "int32"),
            ),
            out_sinfo=R.Tensor((1, 2, 2, 2), dtype="uint8"),
        )
        return out

    @T.prim_func
    def abs(
        rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
        rxplaceholder_1: T.Buffer((), "float32"),
        rxplaceholder_2: T.Buffer((), "int32"),
        rxplaceholder_3: T.Buffer((), "float32"),
        rxplaceholder_4: T.Buffer((), "int32"),
        compute: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "uint8"),
    ):
        T.func_attr({"tir.noalias": True})
        pass


# data = np.random.random([1, 2, 2, 2]).astype("float32") : Need to hadcode the data
# so that we can get the quantization parameters and use them as input to the main func
data = [
    [
        [[0.3034368, 0.60848576], [0.29697746, 0.67340654]],
        [[0.656068, 0.23129226], [0.42117321, 0.81263936]],
    ]
]
dtype = "uint8"

# Quantizing input : scale is returned as float64 and zp is returned as int32
inp_quant, inp_scale, inp_zero_point = quantize_np(data, dtype)
inp_quant = tvm.nd.array(inp_quant.astype(np.uint8))


# Test the implementations value output with numpy data. First the IR is runn through pass
# to replace unary op with take op. Followed by value testing.
def test_value():
    ops = ["tanh", "sqrt", "rsqrt", "exp", "erf", "sigmoid", "hardswish", "log", "abs"]

    atol_val = 2
    for op_name in ops:
        if op_name == "tanh":
            op_val = np.tanh(data)
            before = Module_tanh
        elif op_name == "sqrt":
            op_val = np.sqrt(data)
            before = Module_sqrt
        elif op_name == "rsqrt":
            op_val = 1 / np.sqrt(data)
            before = Module_rsqrt
        elif op_name == "exp":
            op_val = np.exp(data)
            before = Module_exp
        elif op_name == "erf":
            op_val = special.erf(data)
            before = Module_erf
        elif op_name == "sigmoid":
            op_val = 1 / (1 + np.exp(np.negative(data)))
            atol_val = 15
            before = Module_sigmoid
        elif op_name == "hardswish":
            op_val = hexagon_unary_ops.hardswish_func(data)
            before = Module_hardswish
        elif op_name == "log":
            op_val = np.log(data)
            before = Module_log
        elif op_name == "abs":
            op_val = np.abs(data)
            before = Module_abs

        # Quantizing output : scale is returned as float64 and zp is returned as int32
        out_quant, out_scale, out_zero_point = quantize_np(op_val, dtype)

        after = generate_take_op.PassReplaceWithTakeOpPrimFuncs()(before)
        target = tvm.target.Target("llvm", host="llvm")
        ex = relax.build(after, target, exec_mode="compiled")
        vm = relax.VirtualMachine(ex, tvm.cpu())
        res = vm["main"](inp_quant)

        tvm.testing.assert_allclose(res.numpy(), out_quant, atol=atol_val)
        print("Passed Value : ", op_name)


# Testing the structural implementation, if the unary op is replaced with take op.
def test_structural():
    Modules = [
        Module_tanh,
        Module_sqrt,
        Module_rsqrt,
        Module_exp,
        Module_erf,
        Module_sigmoid,
        Module_hardswish,
        Module_log,
        Module_abs,
    ]
    for mod in Modules:
        after = generate_take_op.PassReplaceWithTakeOpPrimFuncs()(mod)
        print(after)
        assert not tvm.ir.structural_equal(after["main"], mod["main"])
    print("Passed Structural")
