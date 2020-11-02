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
"""Utilities that enable counting the number of layers in a graph."""
import tvm
from tvm import relay
from ..expr_functor import ExprVisitor


class LayerCounter(ExprVisitor):
    """A visitor pass that computes the deepest chain of specified ops in graph."""

    def __init__(self, valid_ops):
        self.depth_count = 0
        self.deepest_count = 0
        self.valid_ops = [relay.op.get(op) for op in valid_ops]
        super().__init__()

    def visit_call(self, call):
        if call.op in self.valid_ops:
            self.depth_count += 1
        current_count = self.depth_count
        self.deepest_count = max(self.deepest_count, current_count)
        for arg in call.args:
            self.visit(arg)
            self.depth_count = current_count

    def count(self):
        return self.deepest_count


def count_layers(expr, valid_ops):
    """Determine the number of layers of specified ops in a graph.
    This pass computes only the deepest chain of ops rather than the
    total number of ops in a graph. Thus, if there are two parallel
    convolutions (for example), they would be considered a single layer.

    Parameters
    ----------
    expr : tvm.relay.Expr, tvm.relay.Function, or tvm.ir.IRModule.
        The input expression.

    valid_ops: List[str]
        A list of the operations that should be included in the count.

    Returns
    -------
    layer_count : int
        The number of layers of the specified operations found in the graph.
    """
    if isinstance(expr, tvm.ir.IRModule):
        expr = expr["main"]
    count_pass = LayerCounter(valid_ops)
    count_pass.visit(expr)
    return count_pass.count()
