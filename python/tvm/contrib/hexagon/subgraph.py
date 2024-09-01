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
# pylint: disable=consider-using-with, unnecessary-ellipsis

"""This module returns the smallest IR test case producing the same error"""
import logging
import tvm
from tvm import relay
from tvm.relay.expr import Call
#logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)


# maps to store the nodes(callnodes) and their ids and vice-versa
id_to_node = {}
node_to_id = {}
NODEID = 0
def fvisit(expr):
    """function to create the mapping"""
    global NODEID # pylint: disable=W0603
    if isinstance(expr, relay.expr.Call):
        id_to_node[NODEID] = expr
        node_to_id[expr] = NODEID
        NODEID += 1


op_name_map = {}

def give_call_node(call_arg, start_node, new_id_to_node):
    """Function that returns appropriate callnode or a relay variable """
    if node_to_id[call_arg] < start_node:
        #append a temporary variable tensor of apprpriate shape if the id of the
        #argument node of the current node is outside the given range to the arguments list
        if call_arg.op.name not in op_name_map:
            op_name_map[call_arg.op.name] = 1
        else:
            op_name_map[call_arg.op.name] += 1
        return relay.var(
            call_arg.op.name + str(op_name_map[call_arg.op.name]),
            type_annotation=relay.transform.InferTypeLocal(call_arg),
        )

    # else just add the new node that must have already been created in previous iterations
    return new_id_to_node[node_to_id[call_arg]]


def produce_ir(start_node, end_node):
    """Producing the ir given a starting and ending node of a module(extracting subgraph)"""

    assert start_node <= end_node, "Start node cannot be greater than the end node"

    # temporary map used for storing newnodes created and their ids
    new_id_to_node = {}
    used_nodes = []
    # Traversing the module in the increasing order of ids
    for call_id in range(start_node, end_node + 1):
        arguments = []
        call_node = id_to_node[call_id]
        for call_arg in call_node.args:
            if isinstance(call_arg, relay.expr.Call):
                used_nodes.append(node_to_id[call_arg])
                arguments.append(give_call_node(call_arg, start_node, new_id_to_node))
            elif isinstance(call_arg, relay.expr.Tuple):
                for i in call_arg:
                    if isinstance(i, relay.expr.Call):
                        used_nodes.append(node_to_id[i])
                        arguments.append(give_call_node(i, start_node, new_id_to_node))
                    else:
                        arguments.append(i)
            else:
                arguments.append(call_arg)

        # creating the whole callnode with the corresponding op,
        #new arguments, and other attributes.
        temp = Call(
            call_node.op,
            arguments,
            call_node.attrs,
            call_node.type_args,
            call_node.span,
        )
        new_id_to_node[call_id] = temp

    op_name_map.clear()
    output_nodes = []
    # Finding the output nodes
    for i in range(start_node, end_node + 1):
        if i not in used_nodes:
            output_nodes.append(new_id_to_node[i])

    # creating the IR
    func = relay.Function(
        relay.analysis.free_vars(relay.expr.Tuple(output_nodes)), relay.expr.Tuple(output_nodes)
    )
    mod = tvm.IRModule.from_expr(func)
    return mod


def give_test_case(mod, target, func, start_node, end_node):
    """Recursive function to return the smallest test case(IR) that throws the 
    error(assuming there is only one error in the module)"""

    logging.debug(mod)
    # base case
    if start_node == end_node:
        return mod
    half = (end_node - start_node) // 2

    # getting the first and second halves of the module
    mod1 = produce_ir(start_node, start_node + half)
    mod1_stat = func(mod1, target)
    if not mod1_stat:
        return give_test_case(mod1, target, func, start_node, start_node + half)

    mod2 = produce_ir(start_node + half + 1, end_node)
    mod2_stat = func(mod2, target)
    if not mod2_stat:
        return give_test_case(mod2, target, func, start_node + half + 1, end_node)

    return mod


def smallest_ir(mod, target, func):
    """Function that is to be called to get the smallest ir"""

    start_node = 0
    end_node = int(give_count(mod) - 1)
# Applying the InferType pass on the module to populate the checked_type
#attribute of all the callnodes in the module

    # Travesing in a post dfs manner to create the mappings
    relay.analysis.post_order_visit(mod["main"], fvisit)
    if func(mod, target):
        return None
    return give_test_case(mod, target, func, start_node, end_node)


def give_count(mod):
    """fucntion to calculate the total number of nodes in a module"""

    op_freqs = relay.analysis.list_op_freqs(mod)
    return sum(op_freqs.values())
