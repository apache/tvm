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
"""Pass to replace unary ops with Look Up Table and take op"""
import tvm
import tvm.testing
from tvm import relax
from tvm.contrib.hexagon import hexagon_unary_ops


def op_replace(call_node):
    """Checks if the op in the graph matched the list of unary ops which can be replaced"""
    def is_op(op_name: str, call_node: relax.Call) -> bool:
        if not isinstance(call_node, relax.Call):
            return False
        call_tir_op = tvm.ir.Op.get("relax.call_tir")
        if call_node.op != call_tir_op:
            return False
        global_var = call_node.args[0]
        return op_name in global_var.name_hint

    ops = ["tanh", "sqrt", "rsqrt", "exp", "erf", "sigmoid", "hardswish", "log", "abs"]
    for op in ops:
        if is_op(op, call_node):
            return True
    return False


@relax.expr_functor.mutator
class Tanh2TakeReplace(tvm.relax.PyExprMutator):
    """Pass which iterated over the nodes, checks for unary ops and replaces them with LUT and take op"""
    def __init__(self, mod: tvm.IRModule) -> None:
        super().__init__(mod)
        self.mod_ = mod

    def transform(self) -> tvm.IRModule:
        """Iterates over all the nodes"""
        for global_var, func in self.mod_.functions.items():
            # Skip non-relax functions
            if not isinstance(func, relax.Function):
                continue
            updated_func = self.visit_expr(func)
            self.builder_.normalize(updated_func)
            self.builder_.update_func(global_var, updated_func)
        # At the end of the transformation we return the updated IRModule from the BlockBuilder.
        return self.builder_.get()

    def visit_call_(self, call_node: relax.Call) -> relax.Call:
        if call_node.args[1][0].struct_info.dtype == "uint8":
            if op_replace(call_node):
                inp, inp_scale, inp_zp, out_scale, out_zp = list(call_node.args[1])
                # LUT node creation
                lut = hexagon_unary_ops.lut_generation(
                    inp_scale, inp_zp, out_scale, out_zp, call_node.args[0].name_hint
                )
                # Take operation node creation
                take_func = hexagon_unary_ops.generate_take_primfunc(inp, call_node.struct_info)
                take_func_gv = self.builder_.add_func(take_func, "take")
                take_node = relax.call_tir(
                    take_func_gv,
                    relax.expr.Tuple(
                        [call_node.args[1][0], relax.expr.Constant(tvm.nd.array(lut))]
                    ),
                    call_node.struct_info,
                )
                return take_node
        return call_node


@tvm.ir.transform.module_pass(opt_level=2, name="replace_unaryop_take")
class PassReplaceWithTakeOpPrimFuncs:
    def transform_module(self, mod, ctx):
        return Tanh2TakeReplace(mod).transform()
