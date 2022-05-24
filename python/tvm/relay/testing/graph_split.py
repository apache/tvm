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
"""
Graph split Helper function.
========================
Helper definition for graph split.
"""
import tvm
from tvm import relay
from tvm.relay import transform, build_module
from tvm.relay.testing import run_opt_pass


def graph_split(function, split_config, parameters=None):
    """Splitting the graph into a list of subgraphs

    Parameters:
    -----------
    function: tvm.relay.function.Function
        The function in which the splitting happen.

    split_config: Array[Dict[str, str/integer]]
        The operator name and the number of the repeated operator name in the function.
        For example a network like following
        %1 = nn.add(%data,  meta[relay.Constant][0])
        %2 = nn.multiply(%1,  meta[relay.Constant][1])
        %3 = nn.add(%2,  meta[relay.Constant][2])
        %4 = nn.multiply(%3,  meta[relay.Constant][3])
        if a splliting solution want to split function as the second 'nn.add', like following

        #subgraph 1
        %1 = nn.add(%data,  meta[relay.Constant][0])
        %2 = nn.multiply(%1,  meta[relay.Constant][1])
        %3 = nn.add(%2,  meta[relay.Constant][2])

        #subgraph 2
        %1 = nn.multiply(%data,  meta[relay.Constant][0])

        the split_config should like following
        split_config = [{"operator_name":"add", "operator_index": 1}]

    params_data : Dict[str, NDArray]
        A map from parameter name to data. The subgraph will bind with the parameters.

    Returns:
    -------
    ret: Array[tvm.ir.module.IRModule]
        A list of IRModule.

    """

    def get_output_node_list(nodes_dependency):
        # Get a list of outputs nodes.

        # The node is a output node if it is used as an argument of a node of other subgraph.
        current_dependency = nodes_dependency[-1].items()
        return [node for node, other_use_count in current_dependency if other_use_count > 0]

    def check_dependency(call, other_subgraph_nodes):
        # checking whether the arguments of current call are comming from other subgraph.
        nonlocal new_input_idx
        need_update = False
        new_args = []
        for arg in call.args:
            arg_from_other_subgraph = False
            for subgraph_nodes in other_subgraph_nodes[:-1]:
                if arg in subgraph_nodes:
                    # Increasing the reference count for the other subgraph node.
                    subgraph_nodes[arg] += 1
                    arg_from_other_subgraph = True
            # when the arg of this call is from other subgraph, recreate it and give it
            # a fixed input name.
            if arg_from_other_subgraph:
                new_args.append(relay.var(f"data_n_{new_input_idx}", arg.checked_type))
            else:
                new_args.append(arg)
            need_update = need_update or arg_from_other_subgraph
            new_input_idx += 1 if arg_from_other_subgraph else 0
        # when the 'tvm.relay.expr.Call' has a arg from other subgraph, recreate it with
        # new name as 'data_n_*'.
        if need_update:
            call = tvm.relay.expr.Call(call.op, new_args, call.attrs, call.type_args, call.span)

        return call, other_subgraph_nodes

    def merge_constant_expr(constant_expr, expr):
        # merge constant express with a express
        if not isinstance(constant_expr.body, tvm.relay.expr.Let):
            return tvm.relay.expr.Let(constant_expr.var, constant_expr.value, expr)

        return tvm.relay.expr.Let(
            constant_expr.var, constant_expr.value, merge_constant_expr(constant_expr.body, expr)
        )

    def visit_and_split(anf, pipeline_mods, split_config, constant_expr):
        # Enumurate all operators of compute graph, then split the compute graph into a group of
        # subgraph.
        nonlocal operator_index_map
        nonlocal new_input_idx
        nonlocal other_subgraph_nodes
        cur_node_dep = other_subgraph_nodes[len(other_subgraph_nodes) - 1]
        if isinstance(anf, tvm.relay.Function):
            return tvm.relay.Function(
                anf.params,
                visit_and_split(anf.body, pipeline_mods, split_config, constant_expr),
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
                # Tacking the variable of current subgraph
                cur_node_dep[anf.var] = 0
                # Get the dependency information of the nodes.
                value, other_subgraph_nodes = check_dependency(value, other_subgraph_nodes)
                if isinstance(value.op, tvm.ir.Op):
                    if value.op.name in operator_index_map:
                        operator_index_map[value.op.name] += 1
                    else:
                        operator_index_map[value.op.name] = 0
                    split_operator_name = split_config[0]["op_name"] if split_config else ""
                    split_operator_index = split_config[0]["op_index"] if split_config else ""
                    # if an operator name and repeating count in the network match with the values
                    # of the 'split configuration', then this place is where we should do the
                    # graph splitting.
                    if (
                        split_config
                        and split_operator_name in operator_index_map
                        and operator_index_map[split_operator_name] >= split_operator_index
                    ):
                        # Do graph splitting.
                        split_config.pop(0)
                        other_subgraph_nodes.append({})
                        ann = visit_and_split(
                            anf.body,
                            pipeline_mods,
                            split_config,
                            constant_expr,
                        )
                        other_subgraph_nodes.pop()
                        output_vars = get_output_node_list(other_subgraph_nodes)
                        # When the nodes of the current subgraph are the depedency node of another
                        # subgraph, we need to set them as the output of current subgraph.
                        body = relay.Tuple(output_vars) if len(output_vars) > 1 else anf.var
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
                visit_and_split(anf.body, pipeline_mods, split_config, constant_expr),
            )

        return anf

    other_subgraph_nodes = [{}]
    pipeline_mods = []
    operator_index_map = {}
    new_input_idx = 0
    constant_expr = None
    subgraph_split_config = split_config.copy()
    # Binding the parameters.
    if parameters:
        function = build_module.bind_params_by_name(function, parameters)
    anf = run_opt_pass(function, transform.ToANormalForm())
    anf = run_opt_pass(anf, transform.InferType())
    ann = visit_and_split(
        anf,
        pipeline_mods,
        subgraph_split_config,
        constant_expr,
    )
    ann = run_opt_pass(ann.body, transform.ToGraphNormalForm())
    mod = tvm.IRModule.from_expr(ann)
    pipeline_mods.insert(0, mod)
    return pipeline_mods
