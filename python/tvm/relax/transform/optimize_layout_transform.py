
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
# pylint: disable=invalid-name
"""Relax Optimize Layout Transform pass."""
import tvm
from tvm import IRModule, relax
from tvm.ir.transform import module_pass
from tvm.relax.analysis import remove_all_unused
from tvm.relax.expr_functor import mutator, PyExprMutator
from typing import Union

@mutator
class OptimizeLayoutTranformMutator(PyExprMutator):
    '''
    Mutator to iterate over relax functions to
    remove redundant transform layout operators
    introduced by AlterOpImpl pass.

    Parameters
    ----------
    mod: IRModule
        The ir module

    '''

    def __init__(self, mod: IRModule) -> None:
        super().__init__(mod)
        self.mod_ = mod
        self.patterns = [
            [
                "relax.layout_transform",
                "relax.layout_transform"
            ]
        ]


    # Matches the call_node against the pattern layout_transform -> layout_transform.
    # Based on the pattern matching, returns the updated arguments for call_node.
    def update_args(self, call_node) -> Union[None, relax.Tuple]:
        # Helper function to check if the called TIR function is matching
        # the relax operator name.
        def check_op_type(call_node: relax.Call, op_name: str) -> bool:
            if not isinstance(call_node, relax.Call):
                return False
            if call_node.op != tvm.ir.Op.get(op_name):
                return False
            return True

        new_call_args = []

        # Update args of call_node be checking the pattern 
        for arg in call_node.args[1]:
            is_pattern_match = False
            if not isinstance(arg, relax.expr.Var):
                new_call_args.append(arg)
                continue

            for pattern in self.patterns:
                is_pattern_found = True
                value = arg
                for pat in pattern:
                    if value == None:
                        break
                    value = self.lookup_binding(value)
                    if not check_op_type(value, pat):
                        is_pattern_found = False
                        break
                    value = value.args[0]

                # Check if the shape of value matches the shape of arg to replace
                if value != None and list(value.struct_info.shape) != list(arg.struct_info.shape):
                    is_pattern_found = False
                if is_pattern_found:
                    arg_to_update = value
                    break

            if is_pattern_found:
                new_call_args.append(arg_to_update)
            else:
                new_call_args.append(arg)

        return new_call_args

    def transform(self) -> IRModule:
        # Iterate over all the functions in the IRModule
        for global_var, func in self.mod_.functions.items():
            # Skip non-relax functions
            if not isinstance(func, relax.Function):
                continue
            # Skip primitive functions
            if "Primitive" in func.attrs.keys() and func.attrs["Primitive"] != 0:
                continue
            # Update the non-primitive Relax function
            updated_func = self.visit_expr(func)
            # Remove any unused bindings in the updated function
            updated_func = remove_all_unused(updated_func)
            self.builder_.update_func(global_var, updated_func)

        # At the end of the transformation we return the updated IRModule from the BlockBuilder.
        return self.builder_.get()

    # We only need to override Call node mutator.
    # Find out about pattern of interest and if the pattern matches, then
    # create a new call_node with updated args.
    def visit_call_(self, call_node: relax.Call) -> relax.Call:
        # Check if the call node matches our expected pattern
        if call_node.op != tvm.ir.Op.get("relax.call_tir"):
            return call_node

        new_args = self.update_args(call_node)

        # Check if new_args are different from original args
        if new_args == list(call_node.args[1]):
            return call_node

        # Construct a call to the primitive function
        return relax.call_tir(call_node.args[0], new_args, call_node.struct_info)


@module_pass(opt_level=0, name="OptimizeLayoutTransform")
class OptimizeLayoutTransform:
    """The wrapper for the optimization of layout transform pass."""

    def transform_module(self, mod, ctx):
        return OptimizeLayoutTranformMutator(mod).transform()
