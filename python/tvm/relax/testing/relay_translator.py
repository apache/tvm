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
# pylint: disable=unused-argument, invalid-name, no-else-return,
# pylint: disable=too-many-nested-blocks, unused-variable
"""Relay to Relax translator."""

from typing import Any, Dict, List, Optional, Sequence

import tvm
from tvm import relax, relay
from tvm.ir.module import IRModule
from tvm.ir.instrument import PassInstrument
from tvm.relax.testing import nn
from tvm.relay.backend.te_compiler import select_implementation
from tvm.runtime import NDArray
from tvm.target import Target
from tvm.meta_schedule.relay_integration import _autotvm_silencer


def from_relay(
    func: relay.Function,
    target: Target,
    relay_params: Optional[Dict[str, NDArray]] = None,
    *,
    opt_level: int = 3,
    pass_config: Optional[Dict[str, Any]] = None,
    instruments: Optional[Sequence[PassInstrument]] = None,
    disabled_pass: Optional[List[str]] = None,
    translate_op_with_tir: Optional[Dict[str, tvm.tir.PrimFunc]] = None,
    append_op_attrs: bool = False,
) -> IRModule:
    """Convert a Relay function into a Relax program.

    Parameters
    ----------
    func : relay.Function
        Relay function to be converted.

    target: Target
        The target to compile the model, used for selecting topi functions.

    relay_params: Optional[Dict[str, NDArray]]
        Parameters to bind.

    opt_level: int
        The optimization level.

    pass_config: Optional[Dict[str, Any]]
        Pass configuration.

    instruments : Optional[Sequence[PassInstrument]]
        The list of pass instrument implementations to be passed onto relay
        while calling relay passes

    disabled_pass: Optional[List[str]]
        Passes to disable.

    translate_op_with_tir: Optional[Dict[str, tvm.tir.PrimFunc]]
        Dict that maps op names to user-defined PrimFuncs.
        Takes relay operator names and forces them to user-defined PrimFuncs during translation.

    append_op_attrs: bool
        Append relay op attrs to generated prim_funcs

    Returns
    -------
    mod : tvm.IRModule
        The Relax IRModule for compilation
    """
    # A map to store the mapping of Relay Expr to its corresponding Relax var
    var_map = {}
    # The output of the function
    output_var = None

    if not isinstance(target, Target):
        target = Target(target)
    if disabled_pass is None:
        disabled_pass = []
    if pass_config is None:
        pass_config = {
            "relay.FuseOps.max_depth": 1,  # Disable relay fusion
            "relay.backend.use_meta_schedule": True,
            "relay.backend.use_meta_schedule_dispatch": True,
        }

    if relay_params:
        func = relay.build_module.bind_params_by_name(func, relay_params)

    params = []
    tir_var_map: Dict[tvm.tir.Var, tvm.tir.PrimExpr] = dict()

    def convert_shape(shape: List[tvm.tir.PrimExpr]) -> List[tvm.tir.PrimExpr]:
        """Convert the relay shape to relax shape by changing Any dim to symbolic dim"""
        ret = []
        for dim in shape:
            if isinstance(dim, tvm.tir.IntImm):
                ret.append(tvm.tir.IntImm("int64", int(dim)))
            elif isinstance(dim, tvm.tir.Any):
                ret.append(tvm.tir.Var("d", "int64"))
            else:
                ret.append(dim)
        return ret

    def _copy_undefined_var_in_shape(sinfo: relax.TensorStructInfo):
        def _visit_expr(e: tvm.tir.PrimExpr):
            if isinstance(e, tvm.tir.Var) and e not in tir_var_map:
                new_var = tvm.tir.Var(e.name, e.dtype)
                tir_var_map[e] = new_var

        assert isinstance(
            sinfo.shape, relax.ShapeExpr
        ), "arg with TensorStructInfo in Relay translator must have ShapeExpr shape"
        for shape_value in sinfo.shape.values:
            tvm.tir.stmt_functor.post_order_visit(shape_value, _visit_expr)

    def visit_func(node):
        nonlocal output_var
        if isinstance(node, relay.Var):
            if isinstance(node.type_annotation, relay.TensorType):
                var_map[node] = nn.Placeholder(
                    tuple(convert_shape(node.type_annotation.shape)),
                    node.type_annotation.dtype,
                    node.name_hint,
                )
                params.append(var_map[node])
            else:
                raise TypeError("The type of relay.Var to be translated must be of TensorType.")
        elif isinstance(node, relay.Call):
            args = node.args
            new_args = []
            te_inputs = []
            for arg in args:
                if arg in var_map:
                    arg_expr = var_map[arg]
                    if isinstance(arg_expr.struct_info, relax.TensorStructInfo):
                        _copy_undefined_var_in_shape(arg_expr.struct_info)
                        new_args.append(arg_expr)
                        te_inputs.append(tvm.relax.expr.te_tensor(arg_expr, tir_var_map))
                    elif isinstance(arg_expr.struct_info, relax.TupleStructInfo):
                        n_tensor = len(arg_expr.struct_info.fields)
                        bound_tuple = bb.lookup_binding(arg_expr)
                        if isinstance(bound_tuple, relax.Tuple):
                            assert len(bound_tuple) == n_tensor
                        for i in range(n_tensor):
                            if isinstance(bound_tuple, relax.Tuple):
                                item = bb.emit(bound_tuple[i])
                            else:
                                item = bb.emit(relax.TupleGetItem(arg_expr, i))

                            assert isinstance(item.struct_info, relax.TensorStructInfo), (
                                "Relay translator doesn't support Call "
                                "argument being nested Tensor tuple."
                            )
                            _copy_undefined_var_in_shape(item.struct_info)
                            new_args.append(item)
                            te_inputs.append(tvm.relax.expr.te_tensor(item, tir_var_map))
                    else:
                        raise TypeError(
                            f"CallTIR argument type being {type(arg_expr.checked_type)} is not "
                            "supported."
                        )

            op_name = node.op.name
            attrs = node.attrs
            out_type = node.checked_type

            op_attrs_map = {}
            if append_op_attrs:
                func_attr_map = {"op_name": op_name}
                if attrs:
                    for attr in attrs.keys():
                        func_attr_map[attr] = attrs[attr]

                op_attrs_map["op_attrs"] = func_attr_map

            if translate_op_with_tir and op_name in translate_op_with_tir:
                tir_gvar = bb.add_func(translate_op_with_tir[op_name], op_name)
                call = relax.call_tir(
                    tir_gvar, new_args, relax.TensorStructInfo(out_type.shape, out_type.dtype)
                )
                var = bb.emit(call)
            else:
                with target:
                    best_impl, outputs = select_implementation(
                        node.op,
                        attrs,
                        te_inputs,
                        out_type,
                        target,
                        use_autotvm=False,
                    )
                    compute_func = best_impl.compute
                    name_hint = op_name.split(".")[-1]
                    var = bb.emit_te(
                        compute_func,
                        attrs,
                        new_args,
                        node.checked_type,
                        primfunc_name_hint=name_hint,
                        primfunc_attrs=op_attrs_map,
                    )

            output_var = var
            var_map[node] = var
        elif isinstance(node, relay.Constant):
            # fill the shape and checked_type fields of the Constant
            new_constant = relax.Constant(node.data)
            var_map[node] = new_constant
        elif isinstance(node, relay.Tuple):
            new_fields = []
            for field in node.fields:
                if field in var_map:
                    new_fields.append(var_map[field])
                else:
                    raise RuntimeError("field is not in var_map.")
            new_tuple = relax.Tuple(new_fields)
            new_tuple_var = relax.BlockBuilder.current().emit(new_tuple)
            var_map[node] = new_tuple_var
            output_var = new_tuple_var
        elif isinstance(node, relay.TupleGetItem):
            if node.tuple_value in var_map:
                new_tuple = var_map[node.tuple_value]
                new_tuple_get_item_node = relax.TupleGetItem(new_tuple, node.index)
                new_tuple_get_item_var = relax.BlockBuilder.current().emit(new_tuple_get_item_node)
                var_map[node] = new_tuple_get_item_var
                output_var = new_tuple_get_item_var
            else:
                raise RuntimeError("tuple is not in var_map")
        elif isinstance(node, relay.Function):
            cur_bb = relax.BlockBuilder.current()
            gv = cur_bb.emit_output(output_var)
            df_block = cur_bb._end_block()
            cur_bb._func._blocks.append(df_block)
            cur_bb.emit_func_output(gv, params)
        elif isinstance(node, tvm.ir.Op):
            pass
        else:
            raise TypeError("{} is not supported yet.".format(str(type(node))))

    # List of subset of relay->relay optimizations
    # See src/relay/backend/utils.cc::GetPassPrefix() for full list
    seq = tvm.get_global_func("relay.backend.GetPassPrefixSeq")(True, True)

    # Since optimization passes and OpStrategy are highly context-dependent,
    # we match the exact same context with `extract_task_from_relay()` env
    with target, _autotvm_silencer(), tvm.transform.PassContext(
        opt_level=opt_level,
        config=pass_config,
        disabled_pass=disabled_pass,
        instruments=instruments,
    ):
        mod = tvm.IRModule.from_expr(func)
        mod = seq(mod)
        bb = relax.BlockBuilder()
        with bb.function("main"):
            bb._begin_dataflow_block()
            relay.analysis.post_order_visit(mod["main"], visit_func)

    return bb.get()
