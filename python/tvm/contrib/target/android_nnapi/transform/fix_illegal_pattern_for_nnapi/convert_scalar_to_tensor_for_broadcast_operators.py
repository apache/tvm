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
"""Convert scalar arguments to a broadcasting operator to its tensor equivalent
for Android NNAPI conversion."""
import tvm.relay


ZEROS_OP = tvm.relay.op.get("zeros")
ADD_OP = tvm.relay.op.get("add")
SUBTRACT_OP = tvm.relay.op.get("subtract")
MULTIPLY_OP = tvm.relay.op.get("multiply")
DIVIDE_OP = tvm.relay.op.get("divide")


class ConvertScalarToTensorForBroadcastOperators(tvm.relay.ExprMutator):
    """Convert scalar arguments to a broadcasting operator to its tensor equivalent
    for Android NNAPI conversion."""

    def __init__(self):
        super().__init__()
        self._call_op_stack = []

    def __call__(self, expr):
        return self.visit(expr)

    def visit_call(self, call):
        self._call_op_stack.append(call)
        if self._parent_is_transform_target() and self._is_scalar(call):
            assert (
                isinstance(call.op, tvm.ir.Op) and call.op == zeros
            ), "Only tvm.relay.zeros are supported for \
                    tvm.relay.Call scalar to tensor transformation"
            self._call_op_stack.pop()
            return tvm.relay.zeros(shape=(1,), dtype=call.checked_type.dtype)

        ret = super().visit_call(call)
        self._call_op_stack.pop()
        return ret

    def visit_constant(self, const):
        if self._parent_is_transform_target() and self._is_scalar(const):
            return tvm.relay.Constant(
                tvm.nd.array(
                    const.data.asnumpy().reshape(
                        [
                            1,
                        ]
                    )
                )
            )
        return super().visit_constant(const)

    def visit_var(self, var):
        # due to the need to also transform the parameter dict,
        # we only transform scalar variables
        assert not self._parent_is_transform_target() or not self._is_scalar(
            var
        ), "Transforming variable scalar is not supported"
        return super().visit_var(var)

    def _parent_is_transform_target(self):
        if len(self._call_op_stack) == 0:
            return False

        last_call = self._call_op_stack[-1]
        if not isinstance(last_call, tvm.ir.Op):
            return False

        return last_call.op in {
            ADD_OP,
            SUBTRACT_OP,
            MULTIPLY_OP,
            DIVIDE_OP,
        }  # only these ops are supported for the fix for now

    def _is_scalar(self, node):
        return len(node.checked_type.shape) == 0
