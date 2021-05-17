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
"""Relay type recasting pass"""
import tvm
from tvm import relay
from tvm.ir import IRModule
from .transform import InferType
from ..analysis import count_layers
from ..expr_functor import ExprMutator, Call


class RecastMutator(ExprMutator):
    """Cast operations to the target type."""

    def __init__(self, dtype, out_dtype, valid_ops, valid_op_count, skip_layers):
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.depth_count = 0
        self.valid_ops = [relay.op.get(op) for op in valid_ops]
        self.valid_op_count = valid_op_count
        self.skip_layers = skip_layers
        # Convert negative indices to positive ones.
        for i, layer in enumerate(skip_layers):
            if layer < 0:
                skip_layers[i] = self.valid_op_count + layer
        super().__init__()

    def visit_call(self, call):
        # Keep track of our current depth and layer count
        # so we can know whether to skip this layer or not.
        current_depth = self.depth_count
        current_layer = self.valid_op_count - current_depth - 1
        if call.op in self.valid_ops:
            self.depth_count += 1
        # Visit current call operation
        new_fn = self.visit(call.op)
        # Visit current arguments
        args = []
        for arg in call.args:
            args.append(self.visit(arg))
            self.depth_count = current_depth

        # Downcast this op if its the correct type and not skipped.
        if call.op in self.valid_ops and current_layer not in self.skip_layers:
            # Recast inputs to specified type.
            if call.op == relay.op.get("concatenate"):
                if len(call.args) != 1 or not isinstance(call.args[0], relay.expr.Tuple):
                    return Call(new_fn, args, call.attrs)

                tuple_args = [self.visit(arg) for arg in call.args[0].fields]
                new_args = list()
                for arg in tuple_args:
                    new_args.append(relay.cast(arg, dtype=self.dtype))
                new_args = [relay.expr.Tuple(new_args)]
            else:
                args = [self.visit(arg) for arg in call.args]
                new_args = list()
                for arg in args:
                    new_args.append(relay.cast(arg, dtype=self.dtype))

            # If out_dtype is in the attributes, we need to update it.
            orig_dtype = None
            if call.attrs is not None and "out_dtype" in call.attrs.keys():
                new_attr_dict = {}
                for attr in call.attrs.keys():
                    attr_value = call.attrs[attr]
                    if isinstance(attr_value, tvm.ir.container.Array):
                        attr_value = tuple(attr_value)
                    new_attr_dict[str(attr)] = attr_value
                new_attr_dict["out_dtype"] = self.out_dtype
                attr_type = str(call.attrs).split("(")[0]
                new_attrs = tvm.ir.make_node(attr_type, **new_attr_dict)
                if call.attrs["out_dtype"] != "":
                    orig_dtype = call.attrs["out_dtype"]
            else:
                new_attrs = call.attrs

            if orig_dtype is None:
                # Perform type inference to determine the original type.
                new_mod = IRModule.from_expr(call)
                new_mod = InferType()(new_mod)
                checked_arg = new_mod["main"].body
                orig_dtype = checked_arg.checked_type.dtype
            # Recast the output for compatibility with other graph operations.
            return relay.cast(Call(new_fn, new_args, new_attrs), orig_dtype)

        # Otherwise return the unchanged call.
        return Call(new_fn, args, call.attrs)


def recast(expr, dtype, out_dtype, ops=None, skip_layers=None):
    """Convert the types of operations in a graph to a new value.
    Note that this is primarily useful for testing performance of individual
    operations at the new datatype. In a real setting, this pass will
    almost certainly do a poor job converting from one datatype to another
    as it just applies hard casting. For example, when recasting from float
    to integer, many small values will simply be set to 0. Although this will
    allow autotuning and benchmarking to produce proper timings at the new
    data type, the output of the model will of course be heavily impacted.

    Parameters
    ---------
    expr: tvm.relay.Expr, tvm.relay.Function, or tvm.ir.IRModule
        The original function that will have its type changed.
    dtype: str
        The target type to cast to.
    out_dtype: str
        The output type to cast to.
    ops: List[str]
        A list of operations that should have their type changed,
        others will be left as is.
    skip_layers: List[int]
        A list of integers indicating operations that should
        not have their type changed, counted starting with the
        first valid operation encountered. Negative indices are
        allowed and indicate starting at the last layer.
    Returns
    -------
    output_expr : tvm.relay.Expr, tvm.relay.Function, or tvm.ir.IRModule
        The graph after recasting to the specified datatype.
    """
    return_mod = False
    if isinstance(expr, tvm.ir.IRModule):
        expr = expr["main"]
        return_mod = True
    if ops is None:
        ops = ["nn.conv2d"]
    if skip_layers is None:
        skip_layers = []
    layer_depth = count_layers(expr, ops)
    recast_pass = RecastMutator(dtype, out_dtype, ops, layer_depth, skip_layers)
    expr = recast_pass.visit(expr)
    if return_mod:
        return tvm.IRModule.from_expr(expr)
    return expr
