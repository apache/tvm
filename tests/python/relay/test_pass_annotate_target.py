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
"""Unit tests for annotating external targets."""
import os
import sys
import numpy as np
import pytest

import tvm
import tvm.relay.testing
import tvm.relay.transform as transform
from tvm import relay
from tvm import runtime
from tvm.contrib import utils


def check_result(
    mod, map_inputs, out_shape, result, tol=1e-5, target="llvm", device=tvm.cpu(), params=None
):
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        return

    def update_lib(lib):
        test_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
        source_dir = os.path.join(test_dir, "..", "..", "..")
        contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")

        kwargs = {}
        kwargs["options"] = ["-O2", "-std=c++17", "-I" + contrib_path]
        tmp_path = utils.tempdir()
        lib_name = "lib.so"
        lib_path = tmp_path.relpath(lib_name)
        lib.export_library(lib_path, fcompile=False, **kwargs)
        lib = runtime.load_module(lib_path)

        return lib

    def check_vm_result():
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            exe = relay.vm.compile(mod, target=target, params=params)
        code, lib = exe.save()
        lib = update_lib(lib)
        exe = runtime.vm.Executable.load_exec(code, lib)
        vm = runtime.vm.VirtualMachine(exe, device)
        out = vm.run(**map_inputs)
        tvm.testing.assert_allclose(out.numpy(), result, rtol=tol, atol=tol)

    def check_graph_executor_result():
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            json, lib, param = relay.build(mod, target=target, params=params)
        lib = update_lib(lib)
        rt_mod = tvm.contrib.graph_executor.create(json, lib, device)

        for name, data in map_inputs.items():
            rt_mod.set_input(name, data)
        rt_mod.set_input(**param)
        rt_mod.run()
        out = tvm.nd.empty(out_shape, device=device)
        out = rt_mod.get_output(0, out)

        tvm.testing.assert_allclose(out.numpy(), result, rtol=tol, atol=tol)

    check_vm_result()
    check_graph_executor_result()


def test_extern_dnnl():
    def annotated(dtype, ishape, w1shape):
        data = relay.var("data", shape=(ishape), dtype=dtype)
        weight1 = relay.var("weight1", shape=(w1shape), dtype=dtype)
        depthwise_conv2d_1 = relay.nn.conv2d(
            data, weight1, kernel_size=(3, 3), padding=(1, 1), groups=32
        )
        depthwise_conv2d_2 = relay.nn.conv2d(
            depthwise_conv2d_1, weight1, kernel_size=(3, 3), padding=(1, 1), groups=32
        )
        out = relay.add(depthwise_conv2d_1, depthwise_conv2d_2)

        f = relay.Function([data, weight1], out)

        mod = tvm.IRModule.from_expr(f)
        return mod

    def expected(dtype, ishape, w1shape):
        data = relay.var("data", shape=(ishape), dtype=dtype)
        weight1 = relay.var("weight1", shape=(w1shape), dtype=dtype)
        begin0 = relay.annotation.compiler_begin(data, "dnnl")
        begin1 = relay.annotation.compiler_begin(weight1, "dnnl")
        depthwise_conv2d_1 = relay.nn.conv2d(
            begin0, begin1, kernel_size=(3, 3), padding=(1, 1), groups=32
        )
        end0 = relay.annotation.compiler_end(depthwise_conv2d_1, "dnnl")
        end1 = relay.annotation.compiler_end(depthwise_conv2d_1, "dnnl")
        begin2 = relay.annotation.compiler_begin(end1, "dnnl")
        begin3 = relay.annotation.compiler_begin(end0, "dnnl")
        begin4 = relay.annotation.compiler_begin(weight1, "dnnl")
        depthwise_conv2d_2 = relay.nn.conv2d(
            begin3, begin4, kernel_size=(3, 3), padding=(1, 1), groups=32
        )
        end2 = relay.annotation.compiler_end(depthwise_conv2d_2, "dnnl")
        begin5 = relay.annotation.compiler_begin(end2, "dnnl")
        out = relay.add(begin2, begin5)
        end3 = relay.annotation.compiler_end(out, "dnnl")
        f = relay.Function([data, weight1], end3)
        mod = tvm.IRModule.from_expr(f)
        return mod

    dtype = "float32"
    ishape = (1, 32, 14, 14)
    w1shape = (32, 1, 3, 3)

    def test_annotate():
        mod = annotated(dtype, ishape, w1shape)
        mod = transform.AnnotateTarget("dnnl")(mod)
        mod = relay.transform.InferType()(mod)
        ref_mod = expected(dtype, ishape, w1shape)
        ref_mod = relay.transform.InferType()(ref_mod)
        tvm.ir.assert_structural_equal(mod, ref_mod)

    def test_run():
        if not tvm.get_global_func("relay.ext.dnnl", True):
            print("skip because DNNL codegen is not available")
            return

        ref_mod = annotated(dtype, ishape, w1shape)
        mod = annotated(dtype, ishape, w1shape)
        mod = transform.PartitionGraph()(mod)

        i_data = np.random.uniform(0, 1, ishape).astype(dtype)
        w1_data = np.random.uniform(0, 1, w1shape).astype(dtype)

        ref_res = relay.create_executor("graph", mod=ref_mod, device=tvm.cpu()).evaluate()(
            i_data, w1_data
        )

        check_result(
            mod, {"data": i_data, "weight1": w1_data}, (1, 32, 14, 14), ref_res.numpy(), tol=1e-5
        )

    test_annotate()
    test_run()


@pytest.mark.skip(reason="fix constant node before opening this case")
def test_extern_dnnl_mobilenet():
    if not tvm.get_global_func("relay.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = "float32"
    ishape = (1, 3, 224, 224)
    mod, params = relay.testing.mobilenet.get_workload(batch_size=1, dtype="float32")

    mod["main"] = relay.build_module.bind_params_by_name(mod["main"], params)
    mod = transform.AnnotateTarget("dnnl")(mod)
    mod = transform.PartitionGraph()(mod)
    i_data = np.random.uniform(0, 1, ishape).astype(dtype)

    ref_mod, params = relay.testing.mobilenet.get_workload(batch_size=1, dtype="float32")
    ref_res = relay.create_executor("graph", mod=ref_mod, device=tvm.cpu(0)).evaluate()(
        i_data, **params
    )

    check_result(mod, {"data": i_data}, (1, 1000), ref_res.numpy(), tol=1e-5, params=params)


def test_multiple_ends():
    @tvm.ir.register_op_attr("nn.relu", "target.test")
    def relu(expr):  # pylint: disable=unused-variable
        return True

    def before():
        x = relay.var("x", shape=(10, 10))
        r = relay.nn.relu(x)
        a_1 = relay.abs(r)
        a_2 = relay.abs(r)
        out = relay.add(a_1, a_2)
        f = relay.Function([x], out)
        mod = tvm.IRModule.from_expr(f)
        return mod

    def after():
        x = relay.var("x", shape=(10, 10))
        cb_1 = relay.annotation.compiler_begin(x, "test")
        r = relay.nn.relu(cb_1)
        ce_1 = relay.annotation.compiler_end(r, "test")
        ce_2 = relay.annotation.compiler_end(r, "test")
        cb_2 = relay.annotation.compiler_begin(ce_1, "default")
        cb_3 = relay.annotation.compiler_begin(ce_2, "default")
        a_1 = relay.abs(cb_2)
        a_2 = relay.abs(cb_3)
        ce_3 = relay.annotation.compiler_end(a_1, "default")
        ce_4 = relay.annotation.compiler_end(a_2, "default")
        cb_4 = relay.annotation.compiler_begin(ce_3, "default")
        cb_5 = relay.annotation.compiler_begin(ce_4, "default")
        out = relay.add(cb_4, cb_5)
        ce_6 = relay.annotation.compiler_end(out, "default")
        f = relay.Function([x], ce_6)
        mod = tvm.IRModule.from_expr(f)
        return mod

    for annotate_non_call_ops in [False, True]:
        result = transform.AnnotateTarget("test", annotate_non_call_ops)(before())
        expected = transform.InferType()(after())
        assert tvm.ir.structural_equal(expected, result)


def test_type_propagation():
    target = "test_type_propagation"

    @tvm.ir.register_op_attr("nn.relu", "target." + target)
    def relu(expr):  # pylint: disable=unused-variable
        return expr.args[0].checked_type.dtype == "float32"

    def before():
        x = relay.var("x", shape=(10, 10))
        r = relay.nn.relu(x)
        out = relay.nn.relu(r)
        f = relay.Function([x], out)
        mod = tvm.IRModule.from_expr(f)
        return mod

    for annotate_non_call_ops in [False, True]:
        # If the type isn't propogated, then the relu checker function will fail to get the dtype.
        assert transform.AnnotateTarget(target, annotate_non_call_ops)(before())


def test_ref_create_read_write():
    target = "relu"

    @tvm.ir.register_op_attr("nn.relu", "target." + target)
    def annotate(expr):
        return True

    def before():
        ref = relay.expr.RefCreate(relay.const(1.0))
        r = relay.expr.RefWrite(ref, relay.nn.relu(relay.expr.RefRead(ref)))
        return tvm.IRModule.from_expr(r)

    def after(annotate_non_call_ops):
        co = relay.const(1.0)
        if annotate_non_call_ops:
            co = relay.annotation.compiler_begin(co, "default")

        ref = relay.expr.RefCreate(co)
        ref1 = ref
        if annotate_non_call_ops:
            ref = relay.annotation.compiler_end(ref, "default")
            ref = relay.annotation.compiler_begin(ref, "default")
            ref1 = relay.annotation.compiler_end(ref1, "default")
            ref1 = relay.annotation.compiler_begin(ref1, "default")

        read = relay.expr.RefRead(ref1)
        if annotate_non_call_ops:
            read = relay.annotation.compiler_end(read, "default")

        beg = relay.annotation.compiler_begin(read, target)
        relu = relay.nn.relu(beg)
        end = relay.annotation.compiler_end(relu, target)

        if annotate_non_call_ops:
            end = relay.annotation.compiler_begin(end, "default")

        r = relay.expr.RefWrite(ref, end)

        if annotate_non_call_ops:
            r = relay.annotation.compiler_end(r, "default")
        return tvm.IRModule.from_expr(r)

    for annotate_non_call_ops in [True, False, True]:
        result = transform.AnnotateTarget(target, annotate_non_call_ops)(before())
        expected = transform.InferType()(after(annotate_non_call_ops))
        assert tvm.ir.structural_equal(expected, result)


def test_tuple():
    target = "test_tuple"

    @tvm.ir.register_op_attr("nn.relu", "target." + target)
    def relu(expr):  # pylint: disable=unused-variable
        return True

    @tvm.ir.register_op_attr("concatenate", "target." + target)
    def concatenate(expr):  # pylint: disable=unused-variable
        return True

    """Test that TupleNode is included in annotation when surrounded by supported nodes."""

    def before():
        x = relay.var("x", shape=(10, 5))
        y = relay.var("y", shape=(10, 5))
        a_1 = relay.nn.relu(x)
        a_2 = relay.nn.relu(y)
        out = relay.concatenate((a_1, a_2), axis=1)
        f = relay.Function([x, y], out)
        mod = tvm.IRModule.from_expr(f)
        return mod

    def after(annotate_non_call_ops):
        x = relay.var("x", shape=(10, 5))
        y = relay.var("y", shape=(10, 5))
        cb_1 = relay.annotation.compiler_begin(x, target)
        cb_2 = relay.annotation.compiler_begin(y, target)
        a_1 = relay.nn.relu(cb_1)
        a_2 = relay.nn.relu(cb_2)
        ce_1 = relay.annotation.compiler_end(a_1, target)
        ce_2 = relay.annotation.compiler_end(a_2, target)

        if annotate_non_call_ops:
            cb_3 = relay.annotation.compiler_begin(ce_1, target)
            cb_4 = relay.annotation.compiler_begin(ce_2, target)
            tup = relay.Tuple([cb_3, cb_4])
            ce_3 = relay.annotation.compiler_end(tup, target)
        else:
            ce_3 = relay.Tuple([ce_1, ce_2])

        cb_3 = relay.annotation.compiler_begin(ce_3, target)
        out = relay.op._make.concatenate(cb_3, 1)
        ce_4 = relay.annotation.compiler_end(out, target)
        f = relay.Function([x, y], ce_4)
        mod = tvm.IRModule.from_expr(f)
        return mod

    for annotate_non_call_ops in [False, True]:
        result = transform.AnnotateTarget(target, annotate_non_call_ops)(before())
        expected = transform.InferType()(after(annotate_non_call_ops))
        assert tvm.ir.structural_equal(expected, result)


def test_composite_function():
    def before():
        a = relay.var("a", shape=(10, 10))
        b = relay.var("b", shape=(10, 10))

        # add_relu function
        in_1 = relay.var("in_1", shape=(10, 10))
        in_2 = relay.var("in_2", shape=(10, 10))
        add_node = relay.add(in_1, in_2)
        relu_node = relay.nn.relu(add_node)
        add_relu = relay.Function([in_1, in_2], relu_node)
        add_relu = add_relu.with_attr("Composite", "test.add_relu")

        # merged function
        r = relay.Call(add_relu, [a, b])
        f = relay.Function([a, b], r)
        mod = tvm.IRModule.from_expr(f)
        return mod

    def after():
        a = relay.var("a", shape=(10, 10))
        b = relay.var("b", shape=(10, 10))

        # add_relu function
        in_1 = relay.var("in_1", shape=(10, 10))
        in_2 = relay.var("in_2", shape=(10, 10))
        add_node = relay.add(in_1, in_2)
        relu_node = relay.nn.relu(add_node)
        add_relu = relay.Function([in_1, in_2], relu_node)
        add_relu = add_relu.with_attr("Composite", "test.add_relu")

        # merged function
        cb_1 = relay.annotation.compiler_begin(a, "test")
        cb_2 = relay.annotation.compiler_begin(b, "test")
        r = relay.Call(add_relu, [cb_1, cb_2])
        ce_1 = relay.annotation.compiler_end(r, "test")
        f = relay.Function([a, b], ce_1)
        mod = tvm.IRModule.from_expr(f)
        return mod

    result = transform.AnnotateTarget("test")(before())
    expected = transform.InferType()(after())
    assert tvm.ir.structural_equal(expected, result)


def test_double_target():
    @tvm.ir.register_op_attr("nn.relu", "target.double.A")
    def relu(expr):  # pylint: disable=unused-variable
        return True

    def before():
        x = relay.var("x", shape=(10, 5))
        a_1 = relay.nn.relu(x)
        mod = tvm.IRModule.from_expr(a_1)
        return mod

    for annotate_non_call_ops in [True, False]:
        mod = before()
        mod1 = transform.AnnotateTarget("double.A", annotate_non_call_ops)(mod)
        mod2 = transform.AnnotateTarget("double.A", annotate_non_call_ops)(mod1)
        assert tvm.ir.structural_equal(mod1, mod2)


def test_different_targets():
    @tvm.ir.register_op_attr("nn.relu", "target.different.A")
    def relu(expr):  # pylint: disable=unused-variable
        return True

    @tvm.ir.register_op_attr("add", "target.different.B")
    def relu(expr):  # pylint: disable=unused-variable
        return True

    def before():
        x = relay.var("x", shape=(10, 5))
        a_1 = relay.nn.relu(x)
        b_1 = relay.add(a_1, a_1)
        mod = tvm.IRModule.from_expr(b_1)
        return mod

    for annotate_non_call_ops in [True, False]:
        mod = before()
        mod1 = transform.AnnotateTarget("different.A", annotate_non_call_ops)(mod)
        mod1 = transform.AnnotateTarget("different.B", annotate_non_call_ops)(mod1)
        mod2 = transform.AnnotateTarget(["different.A", "different.B"], annotate_non_call_ops)(mod)
        assert tvm.ir.structural_equal(mod1, mod2)


def test_multiple_runs():
    @tvm.ir.register_op_attr("nn.relu", "target.A")
    def relu(expr):  # pylint: disable=unused-variable
        return True

    @tvm.ir.register_op_attr("add", "target.B")
    def add(expr):  # pylint: disable=unused-variable
        return True

    def before():
        x = relay.var("x", shape=(10, 5))
        a_1 = relay.nn.relu(x)
        a_2 = relay.abs(a_1)
        a_3 = relay.nn.relu(a_1)
        out = relay.add(a_2, a_3)

        f = relay.Function([x], out)
        mod = tvm.IRModule.from_expr(f)
        return mod

    for annotate_non_call_ops in [True, False]:
        mod = transform.AnnotateTarget("A", annotate_non_call_ops)(before())
        mod = transform.AnnotateTarget("B", annotate_non_call_ops)(mod)
        expected = transform.AnnotateTarget(["A", "B"], annotate_non_call_ops)(before())
        assert tvm.ir.structural_equal(expected, mod)


def test_ends_with_tuple():
    trgt = "clip"

    @tvm.ir.register_op_attr("clip", "target." + trgt)
    def relu(expr):  # pylint: disable=unused-variable
        return True

    def get_model(get_item):
        """Return a model"""
        a = relay.var("a", shape=(1, 16, 16, 4), dtype="uint8")
        z = relay.op.clip(a, 0, 255)
        b = relay.op.clip(z, 0, 15)
        c = relay.op.clip(z, 16, 31)
        t = relay.Tuple((c, b))
        tgi = relay.TupleGetItem(t, 1) if get_item else t
        foo = relay.Function([a], tgi)
        return tvm.IRModule.from_expr(tgi)

    def get_expected(annotate_non_call_ops, get_item):
        a_ = relay.var("a", shape=(1, 16, 16, 4), dtype="uint8")
        a = relay.annotation.compiler_begin(a_, trgt)
        z = relay.op.clip(a, 0, 255)
        z1 = relay.annotation.compiler_end(z, trgt)
        z1 = relay.annotation.compiler_begin(z1, trgt)
        b = relay.op.clip(z1, 0, 15)
        b = relay.annotation.compiler_end(b, trgt)
        b = relay.annotation.compiler_begin(b, trgt) if annotate_non_call_ops else b
        z2 = relay.annotation.compiler_end(z, trgt)
        z2 = relay.annotation.compiler_begin(z2, trgt)
        c = relay.op.clip(z2, 16, 31)
        c = relay.annotation.compiler_end(c, trgt)
        c = relay.annotation.compiler_begin(c, trgt) if annotate_non_call_ops else c
        t = relay.Tuple((c, b))
        t = relay.annotation.compiler_end(t, trgt) if annotate_non_call_ops else t
        if get_item:
            t = relay.annotation.compiler_begin(t, trgt) if annotate_non_call_ops else t
            tgi = relay.TupleGetItem(t, 1)
            tgi = relay.annotation.compiler_end(tgi, trgt) if annotate_non_call_ops else tgi
        else:
            tgi = t
        foo = relay.Function([a_], tgi)
        return tvm.IRModule.from_expr(foo)

    for get_item in [True, False]:
        for annotate_non_call_ops in [False, True]:
            mod = get_model(get_item)
            mod = transform.AnnotateTarget("clip", annotate_non_call_ops)(mod)
            expected = transform.InferType()(get_expected(annotate_non_call_ops, get_item))
            assert tvm.ir.structural_equal(expected, mod)


def test_if_else():
    target = "test_if_else"

    @tvm.ir.register_op_attr("equal", "target." + target)
    def relu(expr):  # pylint: disable=unused-variable
        return True

    @tvm.ir.register_op_attr("tanh", "target." + target)
    def tanh(expr):  # pylint: disable=unused-variable
        return True

    @tvm.ir.register_op_attr("sigmoid", "target." + target)
    def sigmoid(expr):  # pylint: disable=unused-variable
        return True

    @tvm.ir.register_op_attr("erf", "target." + target)
    def erf(expr):  # pylint: disable=unused-variable
        return True

    """Test that If-else nodes compiles correctly when surrounded by supported nodes."""

    def before():
        data = relay.var("data", shape=(1, 32))
        eq1 = relay.var("e1", shape=[], dtype="float32")
        eq2 = relay.var("e2", shape=[], dtype="float32")
        eq = relay.equal(eq1, eq2)

        true_branch = relay.tanh(data)
        false_branch = relay.sigmoid(data)
        ife = relay.If(eq, true_branch, false_branch)
        out = relay.erf(ife)
        func = relay.Function([data, eq1, eq2], out)
        mod = tvm.IRModule.from_expr(func)

        return mod

    def after():

        data = relay.var("data", shape=(1, 32))
        eq1 = relay.var("e1", shape=[], dtype="float32")
        eq2 = relay.var("e2", shape=[], dtype="float32")

        cb_1 = relay.annotation.compiler_begin(eq1, target)
        cb_2 = relay.annotation.compiler_begin(eq2, target)

        equality_condition = relay.equal(cb_1, cb_2)
        ce_1 = relay.annotation.compiler_end(equality_condition, target)

        # if condition
        cb_3 = relay.annotation.compiler_begin(data, target)
        true_branch = relay.tanh(cb_3)
        ce_2 = relay.annotation.compiler_end(true_branch, target)

        # else condition
        cb_4 = relay.annotation.compiler_begin(data, target)
        false_branch = relay.sigmoid(cb_4)
        ce_3 = relay.annotation.compiler_end(false_branch, target)

        if_condition = relay.If(ce_1, ce_2, ce_3)
        cb_5 = relay.annotation.compiler_begin(if_condition, target)
        erf_out = relay.erf(cb_5)
        ce_4 = relay.annotation.compiler_end(erf_out, target)
        func = relay.Function([data, eq1, eq2], ce_4)
        mod = tvm.IRModule.from_expr(func)
        return mod

    expected = transform.InferType()(after())
    for annotate_non_call_ops in [True, False]:
        result = transform.AnnotateTarget(target, annotate_non_call_ops)(before())
        assert tvm.ir.structural_equal(expected, result)


def test_while_let():
    target = "test_while_let"

    @tvm.ir.register_op_attr("less", "target." + target)
    def less(expr):  # pylint: disable=unused-variable
        return True

    @tvm.ir.register_op_attr("add", "target." + target)
    def add(expr):  # pylint: disable=unused-variable
        return True

    @tvm.ir.register_op_attr("zeros_like", "target." + target)
    def zeros_like(expr):  # pylint: disable=unused-variable
        return True

    """Test that let nodes compiles correctly when surrounded by other nodes."""

    def before():

        var1 = relay.var("var1", shape=(2,))
        var2 = relay.var("var2", shape=(), dtype="int32")
        var3 = relay.var("var3", shape=(2,))
        cond = relay.less(var2, relay.const(10, dtype="int32"))

        loop = relay.var("while_loop")
        ii = var2 + relay.const(1, dtype="int32")
        ss = var3 + var1
        true_branch = loop(ii, ss)
        ife = relay.If(cond, true_branch, var3)
        func_1 = relay.Function([var2, var3], ife)

        ret = relay.Let(loop, func_1, loop(relay.const(0, dtype="int32"), relay.zeros_like(var1)))
        func_2 = relay.Function([var1], ret)
        mod = tvm.IRModule.from_expr(func_2)
        return mod

    def after(annotate_non_call_ops):
        var1 = relay.var("var1", shape=(2,))
        var2 = relay.var("var2", shape=(), dtype="int32")
        var3 = relay.var("var3", shape=(2,))
        var4 = relay.const(10, dtype="int32")

        cb_1 = relay.annotation.compiler_begin(var2, target)
        cb_2 = relay.annotation.compiler_begin(var4, target)

        less_condition = relay.less(cb_1, cb_2)
        ce_1 = relay.annotation.compiler_end(less_condition, target)

        loop = relay.var("while_loop")

        # if condition
        cb_3 = relay.annotation.compiler_begin(var2, target)
        cb_4 = relay.annotation.compiler_begin(relay.const(1, dtype="int32"), target)
        add_op_1 = relay.add(cb_3, cb_4)
        ce_2 = relay.annotation.compiler_end(add_op_1, target)

        cb_5 = relay.annotation.compiler_begin(ce_2, "default") if annotate_non_call_ops else ce_2

        cb_6 = relay.annotation.compiler_begin(var3, target)
        cb_7 = relay.annotation.compiler_begin(var1, target)
        add_op_2 = relay.add(cb_6, cb_7)
        ce_3 = relay.annotation.compiler_end(add_op_2, target)

        cb_8 = relay.annotation.compiler_begin(ce_3, "default") if annotate_non_call_ops else ce_3

        true_branch = loop(cb_5, cb_8)  # while loop
        ce_4 = (
            relay.annotation.compiler_end(true_branch, "default")
            if annotate_non_call_ops
            else true_branch
        )
        if_condition = relay.If(ce_1, ce_4, var3)
        const_1 = relay.const(0, dtype="int32")
        cb_9 = (
            relay.annotation.compiler_begin(const_1, "default")
            if annotate_non_call_ops
            else const_1
        )
        cb_10 = relay.annotation.compiler_begin(var1, target)
        zeros_like = relay.zeros_like(cb_10)
        ce_5 = relay.annotation.compiler_end(zeros_like, target)
        cb_11 = relay.annotation.compiler_begin(ce_5, "default") if annotate_non_call_ops else ce_5
        while_condition = loop(cb_9, cb_11)
        ce_6 = (
            relay.annotation.compiler_end(while_condition, "default")
            if annotate_non_call_ops
            else while_condition
        )

        func_1 = relay.Function([var2, var3], if_condition)
        ret = relay.Let(loop, func_1, ce_6)
        func_2 = relay.Function([var1], ret)
        mod = tvm.IRModule.from_expr(func_2)
        return mod

    for annotate_non_call_ops in [False, True]:
        result = transform.AnnotateTarget(target, annotate_non_call_ops)(before())
        expected = transform.InferType()(after(annotate_non_call_ops))
        assert tvm.ir.structural_equal(expected, result)


def test_if_free_vars():
    target = "test_if_free_vars"

    @tvm.ir.register_op_attr("equal", "target." + target)
    def equal(expr):  # pylint: disable=unused-variable
        return True

    @tvm.ir.register_op_attr("sigmoid", "target." + target)
    def sigmoid(expr):  # pylint: disable=unused-variable
        return True

    @tvm.ir.register_op_attr("erf", "target." + target)
    def erf(expr):  # pylint: disable=unused-variable
        return True

    """Test that If-else nodes compiles correctly when surrounded by free variables"""

    def before():
        data = relay.var("data", shape=(1, 32))
        eq1 = relay.var("e1", shape=[], dtype="float32")
        eq2 = relay.var("e2", shape=[], dtype="float32")
        eq = relay.equal(eq1, eq2)

        true_branch = relay.zeros(shape=(1, 32), dtype="float32")
        false_branch = relay.sigmoid(data)
        ife = relay.If(eq, true_branch, false_branch)
        out = relay.erf(ife)

        func = relay.Function([data, eq1, eq2], out)
        mod = tvm.IRModule.from_expr(func)

        return mod

    def after():
        data = relay.var("data", shape=(1, 32))
        eq1 = relay.var("e1", shape=[], dtype="float32")
        eq2 = relay.var("e2", shape=[], dtype="float32")

        cb_1 = relay.annotation.compiler_begin(eq1, target)
        cb_2 = relay.annotation.compiler_begin(eq2, target)

        equality_condition = relay.equal(cb_1, cb_2)
        ce_1 = relay.annotation.compiler_end(equality_condition, target)

        # if condition
        true_branch = relay.zeros(shape=(1, 32), dtype="float32")

        # else condition
        cb_3 = relay.annotation.compiler_begin(data, target)
        false_branch = relay.sigmoid(cb_3)
        ce_2 = relay.annotation.compiler_end(false_branch, target)

        if_condition = relay.If(ce_1, true_branch, ce_2)
        cb_4 = relay.annotation.compiler_begin(if_condition, target)
        erf_out = relay.erf(cb_4)
        ce_3 = relay.annotation.compiler_end(erf_out, target)
        func = relay.Function([data, eq1, eq2], ce_3)
        mod = tvm.IRModule.from_expr(func)
        return mod

    for annotate_non_call_ops in [True, False]:
        result = transform.AnnotateTarget(target, annotate_non_call_ops)(before())
        expected = transform.InferType()(after())
        assert tvm.ir.structural_equal(expected, result)


def test_free_vars_zeros():
    target = "test_free_vars_zeros"

    """Test that free variables compile correctly on their own"""

    def before():
        func = relay.Function([], relay.zeros(shape=(0), dtype="float32"))
        mod = tvm.IRModule.from_expr(func)
        return mod

    def after():
        func = relay.Function([], relay.zeros(shape=(0), dtype="float32"))
        mod = tvm.IRModule.from_expr(func)
        return mod

    result = transform.AnnotateTarget(target)(before())
    expected = transform.InferType()(after())
    assert tvm.ir.structural_equal(expected, result)


def test_empty_tuple():
    target = "test_empty_tuple"

    """An empty tuple should behave just like a call with no args (see above test)."""

    def before():
        func = relay.Function([], relay.Tuple([]))
        mod = tvm.IRModule.from_expr(func)
        return mod

    def after():
        func = relay.Function([], relay.Tuple([]))
        mod = tvm.IRModule.from_expr(func)
        return mod

    for annotate_non_call_ops in [True, False]:
        result = transform.AnnotateTarget(target, annotate_non_call_ops)(before())
        expected = transform.InferType()(after())
        assert tvm.ir.structural_equal(expected, result)


if __name__ == "__main__":
    test_extern_dnnl()
    test_composite_function()
    # test_extern_dnnl_mobilenet()
    test_multiple_ends()
    test_type_propagation()
    test_tuple()
    test_multiple_runs()
    test_if_else()
    test_while_let()
    test_if_free_vars()
    test_free_vars_zeros()
    test_different_targets()
    test_double_target()
    test_ends_with_tuple()
    test_ref_create_read_write()
    test_empty_tuple()
