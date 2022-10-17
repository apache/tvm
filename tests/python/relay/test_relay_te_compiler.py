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
import numpy as np
import tvm
from tvm import te
import tvm.testing
from tvm import relay
from tvm import autotvm
from tvm import topi
from tvm.relay.backend import te_compiler
from tvm.relay.testing import run_infer_type
from tvm.relay.testing.temp_op_attr import TempOpAttr


@autotvm.register_topi_compute("test/conv2d_1")
def _compute_conv2d_1(cfg, input, filter, strides, padding, dilation, out_dtype):
    return topi.nn.conv2d_nchw(input, filter, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("test/conv2d_1")
def _schedule_conv2d_1(cfg, outs):
    return topi.generic.schedule_conv2d_nchw(outs)


@autotvm.register_topi_compute("test/conv2d_2")
def _compute_conv2d_2(cfg, input, filter, strides, padding, dilation, out_dtype):
    return topi.nn.conv2d_nchw(input, filter, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("test/conv2d_2")
def _schedule_conv2d_2(cfg, outs):
    return topi.generic.schedule_conv2d_nchw(outs)


def _compute_conv2d_3(input, filter, strides, padding, dilation, out_dtype):
    return topi.nn.conv2d_nchw(input, filter, strides, padding, dilation, out_dtype)


def _schedule_conv2d_3(outs):
    return topi.generic.schedule_conv2d_nchw(outs)


@tvm.target.override_native_generic_func("test_conv2d_strategy")
def _tmp_strategy(attrs, inputs, out_type, target):
    strategy = relay.op.OpStrategy()
    strategy.add_implementation(
        relay.op.strategy.wrap_compute_conv2d(_compute_conv2d_1),
        relay.op.strategy.wrap_topi_schedule(_schedule_conv2d_1),
        name="conv2d_1",
        plevel=10,
    )
    strategy.add_implementation(
        relay.op.strategy.wrap_compute_conv2d(_compute_conv2d_2),
        relay.op.strategy.wrap_topi_schedule(_schedule_conv2d_2),
        name="conv2d_2",
        plevel=15,
    )
    ic = inputs[0].shape[1]
    with tvm.te.SpecializedCondition(ic >= 16):
        strategy.add_implementation(
            relay.op.strategy.wrap_compute_conv2d(_compute_conv2d_3),
            relay.op.strategy.wrap_topi_schedule(_schedule_conv2d_3),
            name="conv2d_3",
            plevel=20,
        )
    return strategy


def _create_record(task_name, dshape, wshape, target, cost):
    args = [te.placeholder(dshape), te.placeholder(wshape), (1, 1), (1, 1, 1, 1), (1, 1), "float32"]
    task = autotvm.task.create(task_name, args, target)
    cfg = autotvm.ConfigEntity(0, None, {}, [])
    cfg.cost = cost
    inp = autotvm.MeasureInput(target=target, task=task, config=cfg)
    result = autotvm.MeasureResult(costs=(cost,), error_no=0, all_cost=-1, timestamp=-1)
    return (inp, result)


def test_get_valid_implementations():
    target = tvm.target.Target("llvm")

    def _get_impls(dshape, wshape):
        data = relay.var("data", shape=dshape)
        weight = relay.var("wshape", shape=wshape)
        out = relay.nn.conv2d(data, weight, padding=(1, 1))
        out = run_infer_type(out)
        return relay.backend.te_compiler.get_valid_implementations(
            relay.op.get("nn.conv2d"),
            out.attrs,
            [te.placeholder(dshape), te.placeholder(wshape)],
            out.checked_type,
            target,
        )

    with TempOpAttr("nn.conv2d", "FTVMStrategy", _tmp_strategy):
        impls = _get_impls((1, 8, 7, 7), (32, 8, 3, 3))
        assert len(impls) == 2
        impls = _get_impls((1, 16, 7, 7), (32, 16, 3, 3))
        assert len(impls) == 3


def test_select_implementation():
    target = tvm.target.Target("llvm")

    def _select_impl(dshape, wshape, use_autotvm=False):
        data = relay.var("data", shape=dshape)
        weight = relay.var("wshape", shape=wshape)
        out = relay.nn.conv2d(data, weight, padding=(1, 1))
        out = run_infer_type(out)
        return relay.backend.te_compiler.select_implementation(
            relay.op.get("nn.conv2d"),
            out.attrs,
            [te.placeholder(dshape), te.placeholder(wshape)],
            out.checked_type,
            target,
            use_autotvm,
        )

    with TempOpAttr("nn.conv2d", "FTVMStrategy", _tmp_strategy):
        impl, _ = _select_impl((1, 8, 7, 7), (32, 8, 3, 3))
        assert impl.name == "conv2d_2"
        impl, _ = _select_impl((1, 8, 7, 7), (32, 8, 3, 3), True)
        assert impl.name == "conv2d_2"
        impl, _ = _select_impl((1, 16, 7, 7), (32, 16, 3, 3))
        assert impl.name == "conv2d_3"
        impl, _ = _select_impl((1, 16, 7, 7), (32, 16, 3, 3), True)
        assert impl.name == "conv2d_3"

        # add autotvm record
        records = []
        records.append(_create_record("test/conv2d_1", (1, 8, 7, 7), (32, 8, 3, 3), target, 0.5))
        records.append(_create_record("test/conv2d_1", (1, 16, 7, 7), (32, 16, 3, 3), target, 1.0))
        with target:
            with autotvm.apply_history_best(records):
                impl, _ = _select_impl((1, 8, 7, 7), (32, 8, 3, 3), True)
                assert impl.name == "conv2d_1"
                impl, _ = _select_impl((1, 16, 7, 7), (32, 16, 3, 3), True)
                assert impl.name == "conv2d_1"

        records.append(_create_record("test/conv2d_2", (1, 8, 7, 7), (32, 8, 3, 3), target, 0.2))
        records.append(_create_record("test/conv2d_1", (1, 16, 7, 7), (32, 16, 3, 3), target, 1.2))
        with target:
            with autotvm.apply_history_best(records):
                impl, _ = _select_impl((1, 8, 7, 7), (32, 8, 3, 3), True)
                assert impl.name == "conv2d_2"
                impl, _ = _select_impl((1, 16, 7, 7), (32, 16, 3, 3), True)
                assert impl.name == "conv2d_1"


def test_te_compiler():
    tec = relay.backend.te_compiler.get()

    def get_func(shape):
        x = relay.var("x", shape=shape)
        y = relay.add(x, x)
        z = relay.add(y, x)
        f = relay.Function([x], z)
        mod = tvm.IRModule.from_expr(f)
        mod = relay.transform.InferType()(mod)
        return mod["main"]

    z1 = tec.lower(get_func((10,)), "llvm")
    z2 = tec.lower(get_func((10,)), "llvm")
    z3 = tec.lower(get_func(()), "llvm")
    assert z1.same_as(z2)
    assert not z3.same_as(z1)
    if tvm.testing.device_enabled("cuda"):
        z4 = tec.lower(get_func(()), "cuda")
        assert not z3.same_as(z4)

    # Test JIT target
    for target in ["llvm"]:
        dev = tvm.device(target)
        if tvm.testing.device_enabled(target):
            f = tec.jit(get_func((10,)), target)
            x = tvm.nd.array(np.ones(10).astype("float32"), device=dev)
            y = tvm.nd.empty((10,), device=dev)
            f(x, y)
            tvm.testing.assert_allclose(y.numpy(), x.numpy() * 3)


# Note: Once the te compiler is removed, we should keep this test so that
# we make sure that opt_level=0 passes are being called correctly.
def test_compile_placeholder_bypass():
    te_compiler = relay.backend.te_compiler.get()
    x = relay.var("x", shape=(2, 3))
    y = relay.var("y", shape=(2, 3))
    z = relay.var("z", shape=(2, 3))
    result = relay.Tuple([x, relay.op.concatenate([y, z], axis=0)])
    func = relay.Function(relay.analysis.free_vars(result), result)
    with tvm.transform.PassContext(opt_level=0):
        graph, lib, params = relay.build(tvm.IRModule.from_expr(func), "llvm")


def test_compile_injective_with_tuple():
    x = relay.var("x", shape=(2, 3))
    y = relay.var("y", shape=(2, 3))
    x_transpose = relay.transpose(x)
    output = relay.Tuple([x_transpose, y])
    func = relay.Function([x, y], output)
    relay.build(tvm.IRModule.from_expr(func), "llvm")


def test_compile_tuple_dup():
    x = relay.var("data", shape=(16, 16))
    log = relay.log(x)
    output = relay.Tuple([log, log])
    f = relay.Function([x], output)
    relay.build(tvm.IRModule.from_expr(f), "llvm")


def test_compile_full():
    # Shape calculations can happen in int64. The test checks that full operator
    # can handle when shapes are not int32
    shape = (
        tvm.tir.IntImm("int32", 1),
        tvm.tir.IntImm("int64", 16),
        tvm.tir.IntImm("int64", 16),
        tvm.tir.IntImm("int32", 64),
    )
    output = relay.full(relay.const(0, "int32"), shape=shape, dtype="int32")
    f = relay.Function([], output)
    mod = tvm.IRModule.from_expr(f)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    relay.build(mod, "llvm")


def test_compile_nhwc_pack():
    data = relay.var("data", shape=(1, 1, 1, 1024), dtype="uint8")
    weight = relay.var("weight", shape=(1, 1, 1024, 1001), dtype="int8")
    p2 = relay.var("p2", shape=(1, 1, 1, 1), dtype="int32")
    conv = relay.nn.conv2d(
        data,
        weight,
        kernel_size=(1, 1),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="int32",
    )
    multiply = relay.multiply(relay.const(-22, dtype="int32"), p2)
    tile = relay.tile(multiply, reps=(1, 1, 1, 1001))
    subtract = relay.subtract(conv, tile)

    func = subtract
    mod = relay.Function(relay.analysis.free_vars(func), func)
    relay.build(mod, target="llvm")


def test_compile_propogate_hash():
    data = relay.var("data", shape=(1, 1, 1, 1024), dtype="uint8")
    weight = relay.var("weight", shape=(1, 1, 1024, 1001), dtype="int8")
    p2 = relay.var("p2", shape=(1, 1, 1, 1), dtype="int32")
    conv = relay.nn.conv2d(
        data,
        weight,
        kernel_size=(1, 1),
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype="int32",
    )
    multiply = relay.multiply(relay.const(-22, dtype="int32"), p2)
    tile = relay.tile(multiply, reps=(1, 1, 1, 1001))
    subtract = relay.subtract(conv, tile)

    func = subtract
    mod = tvm.IRModule.from_expr(relay.Function(relay.analysis.free_vars(func), func))
    vm = relay.vm.VMCompiler()
    opt_mod, _ = vm.optimize(mod, target="llvm")
    for f in opt_mod.functions.values():
        assert "hash" in f.attrs.keys()


if __name__ == "__main__":
    test_get_valid_implementations()
    test_select_implementation()
    test_te_compiler()
    test_compile_placeholder_bypass()
    test_compile_injective_with_tuple()
    test_compile_tuple_dup()
    test_compile_full()
    test_compile_nhwc_pack()
