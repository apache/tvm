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
from tvm import relay
from tvm.testing import utils
from tvm.relay import transform, build_module
from tvm.relay.testing import run_opt_pass


def graph_split(expr, split_conf, params=None):
    """Splitting the graph into a list of subgraphs"""

    def get_dep_var(sub_var_dep):
        return [var for var in sub_var_dep[len(sub_var_dep) - 1]["ref_nodes"]]

    def parse_dependency(value, snode_dep, new_input_idx):
        new_args = []
        need_update = False
        for var in value.args:
            is_free_var = False
            for dep in snode_dep[:-1]:
                if var in dep["nodes"]:
                    # Mark the previous subgraph node as a dependency.
                    dep["nodes"][var] += 1
                    dep["ref_nodes"][var] = dep["nodes"][var]
                    # The var of this call is a free_var
                    is_free_var = True
            # if the var of this call is a free_var, recreate it and give it a fixed input name.
            if is_free_var:
                need_update = True
                new_args.append(relay.var(f"data_n_{new_input_idx}", var.checked_type))
                new_input_idx += 1
            else:
                new_args.append(var)
        # if the 'tvm.relay.expr.Call' has a free_var, recreate it with new name as 'data_n_*'.
        if need_update:
            value = tvm.relay.expr.Call(
                value.op, new_args, value.attrs, value.type_args, value.span
            )
        return value, snode_dep, new_input_idx

    def merge_constant_expr(constant_expr, expr):
        # merge constant express with a express
        if not isinstance(constant_expr.body, tvm.relay.expr.Let):
            return tvm.relay.expr.Let(constant_expr.var, constant_expr.value, expr)

        return tvm.relay.expr.Let(
            constant_expr.var, constant_expr.value, merge_constant_expr(constant_expr.body, expr)
        )

    def _recursion(anf, pipeline_mods, split_conf, constant_expr):
        # Enumurate all operators of compute graph, then split the compute graph into a group of
        # subgraph.
        nonlocal operator_index_map
        nonlocal new_input_idx
        nonlocal snode_dep
        cur_node_dep = snode_dep[len(snode_dep) - 1]
        if isinstance(anf, tvm.relay.Function):
            return tvm.relay.Function(
                anf.params,
                _recursion(anf.body, pipeline_mods, split_conf, constant_expr),
                anf.ret_type,
                anf.type_params,
                anf.attrs,
            )
        if isinstance(anf, tvm.relay.expr.Let):
            value = anf.value
            # record the constant expr to make sure all sugraphs can find correct constant.
            if isinstance(value, tvm.relay.expr.Constant):
                if not constant_expr:
                    constant_expr = tvm.relay.expr.Let(anf.var, value, anf.var)
                else:
                    constant_expr = tvm.relay.expr.Let(anf.var, value, constant_expr)
            if isinstance(value, tvm.relay.expr.Call):
                new_args = []
                # build current var list
                cur_node_dep["nodes"][anf.var] = 0
                # Get the dependency information of the nodes.
                value, snode_dep, new_input_idx = parse_dependency(value, snode_dep, new_input_idx)
                if isinstance(value.op, tvm.ir.Op):
                    if value.op.name in operator_index_map:
                        operator_index_map[value.op.name] += 1
                    else:
                        operator_index_map[value.op.name] = 0
                    split_operator_name = split_conf[0]["op_name"] if split_conf else ""
                    split_operator_index = split_conf[0]["op_index"] if split_conf else ""
                    # if a operator name and repeating count in the network match with the values
                    # of the 'split configuration', then this place is where we should do the
                    # graph splitting.
                    if (
                        split_conf
                        and split_operator_name in operator_index_map
                        and operator_index_map[split_operator_name] >= split_operator_index
                    ):
                        # Do graph splitting.
                        split_conf.pop(0)
                        snode_dep.append({"nodes": {}, "ref_nodes": {}})
                        ann = _recursion(
                            anf.body,
                            pipeline_mods,
                            split_conf,
                            constant_expr,
                        )
                        snode_dep.pop()
                        dep_vars = get_dep_var(snode_dep)
                        # When the nodes of the current subgraph are the depedency node of another
                        # subgraph, we need to set them as the output of current subgraph.
                        body = relay.Tuple(dep_vars) if len(dep_vars) > 1 else anf.var
                        # when the operator of current subgraph uses previous subgraph constant
                        # as the argument of a "relay.expr.call", such constant may become a free
                        # varaible if the constant does not exist in the current subgraph.
                        # merge the previous constant with current subgraph to avoid such issue.
                        if constant_expr:
                            ann = merge_constant_expr(constant_expr, ann)
                        ann = run_opt_pass(ann, transform.ToGraphNormalForm())
                        mod = tvm.IRModule.from_expr(ann)
                        pipeline_mods.insert(0, mod)
                        # Return the last node of the current subgraph.
                        return tvm.relay.expr.Let(anf.var, value, body)
            return tvm.relay.expr.Let(
                anf.var,
                value,
                _recursion(anf.body, pipeline_mods, split_conf, constant_expr),
            )
        else:
            return anf

    snode_dep = [{"nodes": {}, "ref_nodes": {}}]
    pipeline_mods = []
    operator_index_map = {}
    # Used to tracking new input which caused by graph splitting.
    new_input_idx = 0
    constant_expr = None
    subgraph_split_conf = split_conf.copy()
    # Binding the parameters.
    if params:
        expr = build_module.bind_params_by_name(expr, params)
    anf = run_opt_pass(expr, transform.ToANormalForm())
    anf = run_opt_pass(anf, transform.InferType())
    ann = _recursion(
        anf,
        pipeline_mods,
        subgraph_split_conf,
        constant_expr,
    )
    ann = run_opt_pass(ann.body, transform.ToGraphNormalForm())
    mod = tvm.IRModule.from_expr(ann)
    pipeline_mods.insert(0, mod)
    return pipeline_mods
