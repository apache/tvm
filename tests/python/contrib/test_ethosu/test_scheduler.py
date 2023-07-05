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
import tvm
from tvm.script import tir as T
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
    copy_luts,
)
from tvm.relay.backend.contrib.ethosu.tir.compiler import (
    lower_to_te,
    extract_constants,
    _lower_to_tir,
)
from .infra import (
    AttachType,
    make_ethosu_conv2d,
    make_ethosu_identity,
    make_ethosu_binary_elementwise,
)


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

    cached_func = lower_to_te(func)
    sch = te.create_schedule([cached_func.outputs[0].op])
    inline_no_ops(cached_func, sch)
    reshape_tensor = cached_func.outputs[0].op.input_tensors[0]
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
    cached_func = lower_to_te(func)

    sch = te.create_schedule([cached_func.outputs[0].op])
    planner = copy_constants()
    planner(cached_func, const_dict, sch)
    assert len(sch.stages) == 23
    assert ".global" in sch.stages[6].op.name
    assert ".global" in sch.stages[8].op.name
    assert ".global" in sch.stages[17].op.name
    assert ".global" in sch.stages[19].op.name


def test_no_copy_constants():
    ifm_a = relay.var("IFM_A", shape=(1, 26, 26, 32), dtype="int8")
    conv_a = make_ethosu_conv2d(ifm_a, 32, 8, (3, 3), (0, 0), (1, 1), (1, 1))
    conv_b = make_ethosu_conv2d(conv_a, 8, 4, (1, 1), (0, 0), (1, 1), (1, 1))
    func = relay.Function(relay.analysis.free_vars(conv_b), conv_b)
    func = run_opt_pass(func, relay.transform.InferType())

    func, _ = extract_constants(func)
    cached_func = lower_to_te(func)

    sch = te.create_schedule([cached_func.outputs[0].op])
    assert len(sch.stages) == 19
    ops_names = [x.op.name for x in sch.stages]
    assert all(".global" not in x for x in ops_names)


# This test makes sure that constants and LUTs have a correct storage scope
def test_copy_luts():
    ifm_shape = (1, 33, 33, 11)
    ifm = relay.var("IFM", shape=ifm_shape, dtype="int8")
    lut = relay.const([i for i in range(256)], dtype="int8")
    conv = make_ethosu_conv2d(
        ifm, ifm_shape[3], 8, (3, 3), (0, 0), (1, 1), (1, 1), lut=lut, activation="TANH"
    )
    identity = make_ethosu_identity(conv, lut=lut, activation="TANH")
    func = relay.Function(relay.analysis.free_vars(identity), identity)
    func = run_opt_pass(func, relay.transform.InferType())

    func, const_dict = extract_constants(func)
    te_graph = lower_to_te(func)

    sch = te.create_schedule([te_graph.outputs[0].op])
    copy_constants()(te_graph, const_dict, sch)
    copy_luts()(te_graph, const_dict, sch)
    assert len(sch.stages) == 17
    assert ".global" in sch.stages[6].op.name
    assert ".global" in sch.stages[8].op.name
    assert ".local" in sch.stages[10].op.name


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


# fmt: off
@tvm.script.ir_module
class DiamondGraphTir:
    @T.prim_func
    def main(input_placeholder: T.Buffer((1, 56, 56, 96), "int8"), input_ethosu_write: T.Buffer((1, 56, 56, 24), "int8")) -> None:
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        placeholder = T.Buffer([301056], dtype='int8', data=input_placeholder.data)
        ethosu_write = T.Buffer([75264], dtype='int8', data=input_ethosu_write.data)
        buffer1 = T.Buffer([2848], "uint8")
        buffer3 = T.Buffer([976], "uint8")
        p1_data = T.allocate([2848], "uint8", "global", annotations={"disable_lower_builtin":True})
        p1 = T.Buffer([2848], "uint8", data=p1_data)
        p2_data = T.allocate([976], "uint8", "global", annotations={"disable_lower_builtin":True})
        p2 = T.Buffer([976], "uint8", data=p2_data)
        p5_data = T.allocate([75264], "int8", "global", annotations={"disable_lower_builtin":True})
        p5 = T.Buffer([75264], "int8", data=p5_data)
        p6_data = T.allocate([75264], "int8", "global", annotations={"disable_lower_builtin":True})
        p6 = T.Buffer([75264], "int8", data=p6_data)
        T.evaluate(T.call_extern("ethosu_copy", buffer1[0], 2848, p1[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_copy", buffer3[0], 976, p2[0], dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 56, 56, 96, 56, 0, 56, placeholder[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 5376, 96, 1, "int8", 56, 56, 24, 56, 0, 56, p5[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 1344, 24, 1, 1, 1, 1, 1, 1, 1, p1[0], 2608, T.int8(-1), T.int8(-1), 12, p1[2608], 240, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_conv2d", "int8", 56, 56, 24, 56, 0, 56, p5[0], 0, 0, 0, T.float32(0.5), 10, "NHWC", 1344, 24, 1, "int8", 56, 56, 24, 56, 0, 56, p6[0], 0, 0, 0, T.float32(0.25), 14, "NHWC", 1344, 24, 1, 1, 1, 1, 1, 1, 1, p2[0], 736, T.int8(-1), T.int8(-1), 12, p2[736], 240, T.int8(-1), T.int8(-1), 0, 0, 0, 0, "NONE", 0, 0, "TFL", "NONE", 0, 0, 0, dtype="handle"))
        T.evaluate(T.call_extern("ethosu_binary_elementwise", "int8", 56, 56, 24, 56, 0, 56, p5[0], 0, 0, 0,T.float32(1), 0, "NHWC", 1344, 24, 1, "int8", 56, 56, 24, 56, 0, 56, p6[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1344, 24, 1, "int8", 56, 56, 24, 56, 0, 56, ethosu_write[0], 0, 0, 0, T.float32(1), 0, "NHWC", 1344, 24, 1, "ADD", 0, "NONE", 0, 0, "TFL", 0, 0, 0, 0, 0, 0, dtype="handle"))
    __tvm_meta__ = None
# fmt: on


def test_schedule_diamond_graph():
    ifm_a = relay.var("IFM_A", shape=(1, 56, 56, 96), dtype="int8")
    conv_a = make_ethosu_conv2d(ifm_a, 96, 24, (1, 1), (0, 0), (1, 1), (1, 1))
    conv_b = make_ethosu_conv2d(conv_a, 24, 24, (1, 1), (0, 0), (1, 1), (1, 1))
    add = make_ethosu_binary_elementwise(conv_a, conv_b, 24, 24, "ADD", "int8")

    func = relay.Function(relay.analysis.free_vars(add), add)
    func = run_opt_pass(func, relay.transform.InferType())

    test_mod, _ = _lower_to_tir(func, copy_constants())
    reference_mod = DiamondGraphTir
    tvm.ir.assert_structural_equal(test_mod["main"], reference_mod["main"], True)


def test_copy_constants_fully_connected_weights():
    """Check that MatMul-like conv2d ops do not copy weights to SRAM."""
    ifm = relay.var("IFM", shape=(1, 1, 1, 32), dtype="int8")
    conv = make_ethosu_conv2d(ifm, 32, 8, (1, 1), (0, 0), (1, 1), (1, 1))
    func = relay.Function(relay.analysis.free_vars(conv), conv)
    func = run_opt_pass(func, relay.transform.InferType())

    func, const_dict = extract_constants(func)
    cached_func = lower_to_te(func)

    sch = te.create_schedule([cached_func.outputs[0].op])
    planner = copy_constants()
    planner(cached_func, const_dict, sch)
    assert True not in [".global" in s.op.name for s in sch.stages]


if __name__ == "__main__":
    tvm.testing.main()
