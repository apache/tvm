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

# NOTE: We name this test file to start with test_graph_tuner
# to make it execute after zero_rank tensor test cases. This
# helps avoid topi arithmetic operator overloading issue:
# https://github.com/apache/tvm/issues/3240
# TODO: restore the file name after this issue is resolved.
import pytest

import tvm
from tvm import te

from tvm import autotvm, relay
from tvm.relay.testing import synthetic
from tvm.autotvm.graph_tuner.utils import (
    has_multiple_inputs,
    get_direct_ancestor,
    get_in_nodes,
    get_out_nodes,
    expr2graph,
    bind_inputs,
)
from tvm.autotvm.graph_tuner._base import OPT_OUT_OP
from tvm.autotvm.graph_tuner.utils.traverse_graph import _replace_device_with_tracing
from tvm.relay.expr import Call, TupleGetItem, Tuple, Var


def verify_has_multiple_inputs(node_list, node_idx, input_names, expected_result):
    out = has_multiple_inputs(node_list, node_idx, input_names, OPT_OUT_OP)
    assert out == expected_result, "Output mismatch: expecting checking %s to be %s but got %s." % (
        node_list[node_idx]["op"],
        str(expected_result),
        str(out),
    )


def test_has_multiple_inputs():
    data = relay.var("data")
    out1 = data * relay.expr.const(3.0)
    w0 = relay.var("w0")
    out2 = relay.nn.conv2d(data, w0)
    out = relay.add(out1, out2)
    net = relay.Function(relay.analysis.free_vars(out), out)
    net = bind_inputs(net, {"data": (1, 16, 224, 224), "w0": (16, 16, 1, 1)})
    target_ops = [relay.op.get("nn.conv2d")]
    node_list = []
    node_dict = {}
    expr2graph(net, target_ops, node_dict, node_list, tvm.target.Target("llvm"))
    input_names = ["data"]
    verify_has_multiple_inputs(node_list, 2, input_names, False)
    verify_has_multiple_inputs(node_list, 4, input_names, False)
    verify_has_multiple_inputs(node_list, 5, input_names, True)


def test_expr2graph():
    mod, _ = synthetic.get_workload()
    node_dict = {}
    node_list = []
    target_ops = [relay.op.get("nn.conv2d")]
    op_name_list = []

    def _count_node(node):
        if isinstance(node, Call):
            op_name_list.append(node.op)
        elif isinstance(node, (Var, TupleGetItem, Tuple)):
            op_name_list.append(None)

    relay.analysis.post_order_visit(mod["main"], _count_node)

    expr2graph(mod["main"], target_ops, node_dict, node_list, tvm.target.Target("llvm"))
    assert len(node_list) == len(op_name_list)
    for i, item in enumerate(zip(op_name_list, node_list)):
        op_name, node = item
        assert op_name == node["op"], "%dth Node operator mismatch: expecting %s but got %s" % (
            i,
            str(op_name),
            str(node["op"]),
        )


def test_get_direct_ancestor():
    data = relay.var("data")
    w0 = relay.var("w0")
    out1 = relay.nn.conv2d(data, w0)
    out2 = relay.add(out1, data * relay.expr.const(5.0))
    out3 = out2 + relay.expr.const(2.5)
    w1 = relay.var("w1")
    out = relay.nn.conv2d(out3, w1)
    net = relay.Function(relay.analysis.free_vars(out), out)
    net = bind_inputs(net, {"data": (1, 16, 224, 224), "w0": (16, 16, 1, 1), "w1": (16, 16, 1, 1)})
    target_ops = [relay.op.get("nn.conv2d")]
    node_list = []
    node_dict = {}
    expr2graph(net, target_ops, node_dict, node_list, tvm.target.Target("llvm"))
    visited_dict = {}
    input_names = ["data"]
    out = get_direct_ancestor(node_list, visited_dict, target_ops, 5, input_names)
    assert out == [0], "Output mismatch: expecting [0] but got %s." % str(out)

    # non-regression test
    out = relay.add(relay.log(data), relay.sqrt(data))
    net = relay.Function(relay.analysis.free_vars(out), out)
    net = bind_inputs(net, {"data": (1, 16, 224, 224)})
    node_list = []
    node_dict = {}
    expr2graph(net, target_ops, node_dict, node_list, tvm.target.Target("llvm"))
    out = get_direct_ancestor(node_list, visited_dict, target_ops, 3, input_names)
    assert out == [0], "Output mismatch: expecting [0] but got %s." % str(out)


def test_get_in_nodes():
    data = relay.var("data")
    w0 = relay.var("w0")
    out1 = relay.nn.conv2d(data, w0)
    out2 = relay.add(out1, data)
    out3 = out2 + relay.expr.const(2.5)
    w1 = relay.var("w1")
    out = relay.nn.conv2d(out3, w1)
    net = relay.Function(relay.analysis.free_vars(out), out)
    net = bind_inputs(net, {"data": (1, 16, 224, 224), "w0": (16, 16, 1, 1), "w1": (16, 16, 1, 1)})
    target_ops = [relay.op.get("nn.conv2d")]
    input_names = ["data"]
    node_list = []
    node_dict = {}
    expr2graph(net, target_ops, node_dict, node_list, tvm.target.Target("llvm"))
    out = get_in_nodes(node_list, target_ops, input_names)
    expected_out = {3: [0], 4: [3, 0], 7: [4]}
    diff_set = set(out) ^ set(expected_out)
    if len(diff_set) != 0:
        raise RuntimeError(
            "Output mismatch: expecting %s but got %s." % (str(expected_out), str(out))
        )


def test_get_out_nodes():
    in_nodes_dict = {8: [4], 4: [3, 0], 3: [0]}
    expected_out = {0: [3, 4], 3: [4], 4: [8], 8: []}
    out = get_out_nodes(in_nodes_dict)
    diff_set = set(out) ^ set(expected_out)
    if len(diff_set) != 0:
        raise RuntimeError(
            "Output mismatch: expecting %s but got %s." % (str(expected_out), str(out))
        )


def test_target_device_replacement():
    assert _replace_device_with_tracing("cuda") == "cuda -device=tracing"
    assert (
        _replace_device_with_tracing("cuda -device=some_device -libs=cudnn")
        == "cuda -device=tracing -libs=cudnn"
    )
    assert (
        _replace_device_with_tracing("llvm -device=arm_cpu -arg=xxx")
        == "llvm -device=tracing -arg=xxx"
    )
    assert _replace_device_with_tracing("llvm -device=arm_cpu") == "llvm -device=tracing"
    assert _replace_device_with_tracing("llvm -device=abc, def") == "llvm -device=tracing"


if __name__ == "__main__":
    test_has_multiple_inputs()
    test_expr2graph()
    test_get_direct_ancestor()
    test_get_in_nodes()
    test_get_out_nodes()
