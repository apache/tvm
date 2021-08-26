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
import pytest

pytest.importorskip("ethosu.vela")
from tvm import relay
from tvm.relay.testing import run_opt_pass
from tvm import te, topi
from tvm.relay.backend.contrib.ethosu.tir.scheduler import (
    tile_nd,
    schedule_pragmas,
    inline_no_ops,
    total_cascader,
    copy_constants,
    schedule_cache_reads,
)
from tvm.relay.backend.contrib.ethosu.tir.compiler import lower_to_te, extract_constants
from infra import AttachType, make_ethosu_conv2d


class TestTEGraph:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs


def test_tile_nd():
    input = te.placeholder((12, 12), dtype="uint8", name="input")
    out = topi.nn.relu(input)
    sch = te.create_schedule([out.op])
    outer_iters, inner_iters = tile_nd(sch, out, (3, 4))
    assert tuple(sch[out].leaf_iter_vars) == (*outer_iters, *inner_iters)


def test_schedule_pragmas():
    input = te.placeholder((12, 12), dtype="uint8", name="input")
    out = te.compute(
        (12, 12),
        lambda i, j: input[i, j],
        attrs={
            "op": "unity",
            "info": 1,
        },
    )
    sch = te.create_schedule([out.op])
    sch[out].split(out.op.axis[0], 3)
    schedule_pragmas(sch)
    iter_var = sch[out].leaf_iter_vars[1]
    assert list(sch[out].iter_var_attrs[iter_var].pragma_keys) == ["op", "info"]
    assert list(sch[out].iter_var_attrs[iter_var].pragma_values) == ["unity", 1]


def test_schedule_pragmas_for_const():
    input = te.placeholder((12, 12), dtype="uint8", name="input")
    const = te.compute((), lambda: 2)
    add = topi.add(input, const)
    sch = te.create_schedule([add.op])
    schedule_pragmas(sch)


def test_inline_no_ops():
    input = relay.var("input", shape=(12, 12), dtype="uint8")
    slice = relay.strided_slice(input, [0, 0], [6, 6])
    relu1 = relay.nn.relu(slice)
    reshape = relay.reshape(relu1, (36,))
    relu2 = relay.nn.relu(reshape)
    func = relay.Function(relay.analysis.free_vars(relu2), relu2)
    func = run_opt_pass(func, relay.transform.InferType())

    te_graph = lower_to_te(func)
    sch = te.create_schedule([te_graph.outputs[0].op])
    inline_no_ops(te_graph, sch)
    reshape_tensor = te_graph.outputs[0].op.input_tensors[0]
    slice_tensor = reshape_tensor.op.input_tensors[0].op.input_tensors[0]
    assert sch[reshape_tensor].attach_type == AttachType.kInline
    assert sch[slice_tensor].attach_type == AttachType.kInline


def test_total_cascader():
    input = te.placeholder((12, 12), dtype="uint8", name="input")
    relu1 = topi.nn.relu(input)
    relu2 = topi.nn.relu(relu1)
    relu3 = topi.nn.relu(relu2)
    sch = te.create_schedule([relu3.op])
    cascader = total_cascader((4, 4))
    cascader(TestTEGraph([input], [relu3]), {}, sch)
    assert sch[relu1].attach_type == AttachType.kScope
    assert sch[relu2].attach_type == AttachType.kScope
    assert sch[relu3].attach_type == AttachType.kGroupRoot
    # Check that the attaches are at the correct iter var
    assert sch[relu1].attach_ivar == sch[relu3].leaf_iter_vars[1]
    assert sch[relu2].attach_ivar == sch[relu3].leaf_iter_vars[1]


def test_copy_constants():
    ifm_a = relay.var("IFM_A", shape=(1, 26, 26, 32), dtype="int8")
    conv_a = make_ethosu_conv2d(ifm_a, 32, 8, (3, 3), (0, 0), (1, 1), (1, 1))
    conv_b = make_ethosu_conv2d(conv_a, 8, 4, (1, 1), (0, 0), (1, 1), (1, 1))
    func = relay.Function(relay.analysis.free_vars(conv_b), conv_b)
    func = run_opt_pass(func, relay.transform.InferType())

    func, const_dict = extract_constants(func)
    te_graph = lower_to_te(func)

    sch = te.create_schedule([te_graph.outputs[0].op])
    planner = copy_constants()
    planner(te_graph, const_dict, sch)
    assert len(sch.stages) == 21
    assert ".global" in sch.stages[5].op.name
    assert ".global" in sch.stages[7].op.name
    assert ".global" in sch.stages[15].op.name
    assert ".global" in sch.stages[17].op.name


def test_schedule_cache_reads():
    a = te.placeholder((12, 12), dtype="uint8", name="a")
    b = te.placeholder((12, 12), dtype="uint8", name="b")
    add = topi.add(a, b)
    sch = te.create_schedule([add.op])
    cr = sch.cache_read(b, "global", [add])
    schedule_cache_reads(sch)
    assert len(sch.stages) == 4
    assert len(sch[cr].leaf_iter_vars) == 1
    iv = sch[cr].leaf_iter_vars[0]
    assert list(sch[cr].iter_var_attrs[iv].pragma_keys) == ["op"]
    assert list(sch[cr].iter_var_attrs[iv].pragma_values) == ["ethosu_copy"]


if __name__ == "__main__":
    pytest.main([__file__])
