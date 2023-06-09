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
"Adds pads so that each conv2d operator has only one consumer"

import tvm
from tvm import relay

from ..expr_functor import ExprMutator, Call
from .. import expr as _expr


class PadsWithMultipleConsumersReplicator(ExprMutator):
    """A pass to to handle the situation when nn.pad operator has
    more than one qnn.conv2d consumer.

             pad
           /     \
       Conv2D   Conv2D

    In this case, because of the peculiarities of pattern parsing,
    conv2d does not get into the composite for the NPU.
    Therefore, pads are added so that each has only one consumer.
    """

    def __init__(self):
        ExprMutator.__init__(self)
        self.hashes = set()

    def visit_call(self, call):
        if (
            isinstance(call.op, tvm.ir.Op)
            and isinstance(call.args[0], Call)
            and isinstance(call.args[0].op, tvm.ir.Op)
            and call.op == relay.op.get("qnn.conv2d")
            and call.args[0].op == relay.op.get("nn.pad")
        ):
            if tvm.ir.structural_hash(call.args[0]) not in self.hashes:
                self.hashes.add(tvm.ir.structural_hash(call.args[0]))
            else:
                used_pad = self.visit(call.args[0])
                used_pad_args = [self.visit(arg) for arg in used_pad.args]
                new_pad = Call(
                    used_pad.op, used_pad_args, used_pad.attrs, used_pad.type_args, used_pad.span
                )
                new_pad = self.visit(new_pad)
                new_conv2d_args = []
                for i, arg in enumerate(call.args):
                    if i == 0:
                        new_conv2d_args.append(self.visit(new_pad))
                    else:
                        new_conv2d_args.append(self.visit(arg))
                new_conv2d_op = self.visit(call.op)
                expr__ = _expr.CallWithFields(
                    call,
                    new_conv2d_op,
                    new_conv2d_args,
                    call.attrs,
                    call.type_args,
                    None,
                    call.span,
                )
                return expr__

        new_args = [self.visit(arg) for arg in call.args]
        new_op = self.visit(call.op)
        expr__ = _expr.CallWithFields(
            call, new_op, new_args, call.attrs, call.type_args, None, call.span
        )
        return expr__


def replicate_pads(mod):
    """Traverses the Relay graph to replicate nn.pad operators if thay have
    multiple qnn.conv2d consumers. That making remove the situation when
    e.g. pad+conv2d corresponds qnn_conv2d_pattern, but can not be grouped
    because several conv2d use the same pad operation.

    Parameters
    ----------
    tvm.ir.IRModule
        The IRModule that gets generated from a relay frontend.

    Returns
    -------
    tvm.ir.IRModule
        The IRModule without nn.pad operators with multiple consumers.
    """
    replicator = PadsWithMultipleConsumersReplicator()
    for global_var, func in mod.functions.items():
        func = replicator.visit(func)
        mod.update_func(global_var, func)
    return mod
