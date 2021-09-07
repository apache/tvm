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
"""Codegen for CMSIS-NN"""
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor


class GenerateTIR(ExprVisitor):
    """Generates TIR module containing TIR primfuncs corresponding to the Relay operators.
    Note: Relay operator to primfunc mapping may not be 1:1.
    """

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.tir_mod = None
        self.scale = 1.0 / 256

    def call_contains_op(self, call, op_name):
        if not isinstance(call.op, tvm.ir.op.Op):
            return False
        if call.op.name != op_name:
            return False
        return True

    def is_quantized_softmax(self, call):
        """Checks for the following relay sequence
        a = qnn.dequantize(in, scale, zero_point)
        b = nn.softmax(a)
        c = qnn.quantize(c, scale, zero_point)
        """
        if not self.call_contains_op(call, "qnn.quantize"):
            return False
        softmax_call = call.args[0]
        if not self.call_contains_op(softmax_call, "nn.softmax"):
            return False
        dequantize_call = softmax_call.args[0]
        if not self.call_contains_op(dequantize_call, "qnn.dequantize"):
            return False
        self.scale = dequantize_call.args[1].data.numpy().item(0)
        return True

    def emit_softmax_tir(self, call):
        """Generates TIR extern_call for softmax"""
        shape = call.checked_type.shape  # NHWC
        dtype = call.checked_type.dtype
        ir_builder = tvm.tir.ir_builder.create()
        in_buf = tvm.tir.decl_buffer(shape=shape, dtype=dtype)
        out_buf = tvm.tir.decl_buffer(shape=shape, dtype=dtype)

        trailing_dim = len(shape) - 1
        num_rows = 1
        for dim in range(trailing_dim):
            num_rows *= shape[dim]
        row_size = shape[trailing_dim]
        ir_builder.emit(
            tvm.tir.call_extern(
                dtype,
                "arm_softmax_s8",
                in_buf.data,
                num_rows,
                row_size,
                self.scale,
                out_buf.data,
            )
        )
        prim_func = tvm.tir.PrimFunc([in_buf, out_buf], ir_builder.get())
        prim_func = prim_func.with_attr("global_symbol", self.name)
        prim_func = prim_func.with_attr("tir.noalias", True)
        self.tir_mod = tvm.IRModule({self.name: prim_func})

    def visit_call(self, call):
        """Iterates over the relay operators within relay external function"""
        super().visit_call(call)
        if self.is_quantized_softmax(call):
            self.emit_softmax_tir(call)

    def generate_tir(self, func):
        self.visit(func)
        return self.tir_mod


def relay_to_tir(name, func):
    """Lower a Relay function to TIR for the CMSIS-NN target.

    The Relay function should only contain operations supported
    by the CMSIS-NN target. This is enforced by the graph partitioner
    for CMSIS-NN.

    Parameters
    ----------
    name: str
        Name of the external relay function
    func : tvm.relay.Function
        The Relay function to lower.

    Returns
    -------
    mod : tvm.IRModule
        The lowered TIR module.

    """
    return GenerateTIR(name).generate_tir(func)


@tvm.register_func("relay.ext.cmsisnn")
def cmsisnn_compiler(relay_func):
    """It compiles Relay's external function into equivalent TIR
    and subsequently converts that into 'c' code. During the 'c'
    code generation, it embeds CMSIS-NN APIs for the corresponding
    operators.
    """
    mod = tvm.IRModule()
    mod["main"] = relay_func
    mod = relay.transform.InferType()(mod)
    func_name = relay_func.attrs["global_symbol"]
    tir_mod = relay_to_tir(func_name, mod["main"])
    cmsisnn_runtime = tvm._ffi.get_global_func("runtime.module.cmsisnn.create")
    return cmsisnn_runtime(tir_mod)
