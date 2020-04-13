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
"""Unit tests for graph partitioning."""
import os
import sys

import numpy as np
import pytest

import tvm
import tvm.relay.testing
from tvm import relay
from tvm import runtime
from tvm.relay import transform
from tvm.contrib import util
from tvm.relay.backend import compile_engine
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.build_module import bind_params_by_name


# Leverage the pass manager to write a simple white list based annotator
@transform.function_pass(opt_level=0)
class WhiteListAnnotator:
    def __init__(self, op_list, compiler):
        assert isinstance(op_list, (list, tuple, set))
        self.op_list = op_list
        self.compiler = compiler

    def transform_function(self, func, mod, ctx):

        annotator = self
        class Annotator(tvm.relay.ExprMutator):
            def visit_call(self, call):
                op_name = call.op.name
                if op_name in annotator.op_list:
                    new_args = []
                    for arg in call.args:
                        ann = compiler_begin(super().visit(arg),
                                             annotator.compiler)
                        new_args.append(ann)
                    new_call = relay.Call(call.op, new_args, call.attrs,
                                          call.type_args)
                    return compiler_end(new_call, annotator.compiler)
                else:
                    return super().visit_call(call)
        return Annotator().visit(func)


class CcompilerAnnotator(ExprMutator):
    """
    A simple annotator that creates the following program:
           |
      -- begin --
           |
          add
           |
        subtract
           |
        multiply
           |
       -- end --
           |
    """

    def __init__(self):
        super(CcompilerAnnotator, self).__init__()
        self.in_compiler = 0

    def visit_call(self, call):
        if call.op.name == "add":  # Annotate begin at args
            if self.in_compiler == 1:
                lhs = compiler_begin(super().visit(call.args[0]), "ccompiler")
                rhs = compiler_begin(super().visit(call.args[1]), "ccompiler")
                op = relay.add(lhs, rhs)
                self.in_compiler = 2
                return op
        elif call.op.name == "subtract":
            if self.in_compiler == 1:
                lhs = super().visit(call.args[0])
                rhs = super().visit(call.args[1])
                if isinstance(lhs, relay.expr.Var):
                    lhs = compiler_begin(lhs, "ccompiler")
                if isinstance(rhs, relay.expr.Var):
                    rhs = compiler_begin(rhs, "ccompiler")
                return relay.subtract(lhs, rhs)
        elif call.op.name == "multiply":  # Annotate end at output
            self.in_compiler = 1
            lhs = super().visit(call.args[0])
            rhs = super().visit(call.args[1])
            if isinstance(lhs, relay.expr.Var):
                lhs = compiler_begin(lhs, "ccompiler")
            if isinstance(rhs, relay.expr.Var):
                rhs = compiler_begin(rhs, "ccompiler")
            op = relay.multiply(lhs, rhs)
            if self.in_compiler == 2:
                op = compiler_end(op, "ccompiler")
            self.in_compiler = 0
            return op
        return super().visit_call(call)


class WholeGraphAnnotator(ExprMutator):
    """
    An annotator that creates a compiler for an entire graph.
    """

    def __init__(self, compiler):
        super(WholeGraphAnnotator, self).__init__()
        self.compiler = compiler
        self.last_call = True

    def visit_call(self, call):
        curr_last = self.last_call
        self.last_call = False

        params = []
        for arg in call.args:
            param = super().visit(arg)
            if isinstance(param, relay.expr.Var):
                param = compiler_begin(param, self.compiler)
            params.append(param)

        new_call = relay.Call(call.op, params, call.attrs)
        if curr_last:
            new_call = compiler_end(new_call, self.compiler)
        return new_call


class MobileNetAnnotator(ExprMutator):
    """
    Annotate mobilenet until global_avg_pool.
    """

    def __init__(self, compiler):
        super(MobileNetAnnotator, self).__init__()
        self.compiler = compiler
        self.compiler_open = False

    def visit_call(self, call):

        if call.op.name == 'nn.global_avg_pool2d':
            self.compiler_open = True
        compiler_open = self.compiler_open

        params = []
        for arg in call.args:
            param = super().visit(arg)
            if call.op.name == 'nn.global_avg_pool2d':
                param = compiler_end(param, self.compiler)
            if compiler_open and isinstance(param, relay.expr.Var):
                param = compiler_begin(param, self.compiler)
            params.append(param)

        new_call = relay.Call(call.op, params, call.attrs)
        return new_call


def check_result(mod, map_inputs, out_shape, result, tol=1e-5, target="llvm",
                 ctx=tvm.cpu(), params=None):
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        return

    def update_lib(lib):
        test_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
        source_dir = os.path.join(test_dir, "..", "..", "..")
        contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")

        kwargs = {}
        kwargs["options"] = ["-O2", "-std=c++11", "-I" + contrib_path]
        tmp_path = util.tempdir()
        lib_name = 'lib.so'
        lib_path = tmp_path.relpath(lib_name)
        lib.export_library(lib_path, fcompile=False, **kwargs)
        lib = runtime.load_module(lib_path)

        return lib

    def check_vm_result():
        compile_engine.get().clear()
        with relay.build_config(opt_level=3):
            exe = relay.vm.compile(mod, target=target, params=params)
        code, lib = exe.save()
        lib = update_lib(lib)
        exe = runtime.vm.Executable.load_exec(code, lib)
        vm = runtime.vm.VirtualMachine(exe)
        vm.init(ctx)
        out = vm.run(**map_inputs)
        tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)

    def check_graph_runtime_result():
        compile_engine.get().clear()
        with relay.build_config(opt_level=3):
            json, lib, param = relay.build(mod, target=target, params=params)
        lib = update_lib(lib)
        rt_mod = tvm.contrib.graph_runtime.create(json, lib, ctx)

        for name, data in map_inputs.items():
            rt_mod.set_input(name, data)
        rt_mod.set_input(**param)
        rt_mod.run()
        out = tvm.nd.empty(out_shape, ctx=ctx)
        out = rt_mod.get_output(0, out)

        tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)

    check_vm_result()
    check_graph_runtime_result()


def test_multi_node_compiler():
    x = relay.var('x', shape=(10, 10))
    w0 = relay.var('w0', shape=(10, 10))
    w1 = relay.var('w1', shape=(10, 10))
    w2 = relay.var('w2', shape=(10, 10))
    w3 = relay.var('w3', shape=(10, 10))
    w4 = relay.var('w4', shape=(10, 10))
    w5 = relay.var('w5', shape=(10, 10))
    w6 = relay.var('w6', shape=(10, 10))
    w7 = relay.var('w7', shape=(10, 10))

    # C compiler
    # FIXME: We generate two compilers for this case but they should be merged to one
    # due to the common input (x).
    z0 = relay.add(x, w0)
    p0 = relay.subtract(z0, w1)
    q0 = relay.multiply(p0, w2)

    z1 = relay.add(x, w3)
    p1 = relay.subtract(z1, w4)
    q1 = relay.multiply(p1, w5)

    # Other parts on TVM
    z2 = relay.add(x, w6)
    q2 = relay.subtract(z2, w7)

    r = relay.concatenate((q0, q1, q2), axis=0)
    f = relay.Function([x, w0, w1, w2, w3, w4, w5, w6, w7], r)
    mod = tvm.IRModule()
    ann = CcompilerAnnotator()
    mod["main"] = ann.visit(f)
    mod = transform.PartitionGraph()(mod)
    mod = transform.InferType()(mod)

    x_data = np.random.rand(10, 10).astype('float32')
    w_data = []
    for _ in range(8):
        w_data.append(np.random.rand(10, 10).astype('float32'))

    map_inputs = {"w{}".format(i): w_data[i] for i in range(8)}
    map_inputs["x"] = x_data
    check_result(
        mod, map_inputs, (30, 10),
        np.concatenate((((x_data + w_data[0]) - w_data[1]) * w_data[2],
                        ((x_data + w_data[3]) - w_data[4]) * w_data[5],
                        x_data + w_data[6] - w_data[7]),
                       axis=0))


def test_extern_ccompiler_single_op():
    @transform.function_pass(opt_level=0)
    class MyAnnotator:
        def transform_function(self, func, mod, ctx):
            class Annotator(tvm.relay.ExprMutator):
                def visit_call(self, call):
                    new_args = []
                    for arg in call.args:
                        ann = compiler_begin(self.visit(arg), "ccompiler")
                        new_args.append(ann)
                    new_call = relay.Call(call.op, new_args)
                    return compiler_end(new_call, "ccompiler")
            return Annotator().visit(func)

    x = relay.var('x', shape=(8, 8))
    y = relay.var('y', shape=(8, 8))
    z = x + y
    f = relay.Function([x, y], z)
    x_data = np.random.rand(8, 8).astype('float32')
    y_data = np.random.rand(8, 8).astype('float32')
    mod = tvm.IRModule()
    mod["main"] = f
    mod = MyAnnotator()(mod)
    mod = transform.PartitionGraph()(mod)

    check_result(mod, {"x": x_data, "y": y_data}, (8, 8), x_data + y_data)


def test_extern_ccompiler_default_ops():
    def expected():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(8, 8))
        y = relay.var("y", shape=(8, 8))
        x0 = relay.var("x0", shape=(8, 8))
        y0 = relay.var("y0", shape=(8, 8))
        add = x0 + y0
        # Function that uses C compiler
        func = relay.Function([x0, y0], add)
        func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        func = func.with_attr("Compiler", "ccompiler")
        func = func.with_attr("global_symbol", "ccompiler_0")
        glb_0 = relay.GlobalVar("ccompiler_0")
        mod[glb_0] = func
        add_call = relay.Call(glb_0, [x, y])
        # Function that uses default compiler. Ops are fused in this function.
        p0 = relay.var("p0", shape=(8, 8))
        log = relay.log(p0)
        exp = relay.exp(p0)
        concat = relay.concatenate([log, exp], axis=0)
        fused_func = relay.Function([p0], concat)
        fused_func = fused_func.with_attr("Primitive",
                                          tvm.tir.IntImm("int32", 1))
        fused_call = relay.Call(fused_func, [add_call])
        main = relay.Function([x, y], fused_call)
        mod["main"] = main
        return mod

    x = relay.var("x", shape=(8, 8))
    y = relay.var("y", shape=(8, 8))
    add = x + y
    log = relay.log(add)
    exp = relay.exp(add)
    concat = relay.concatenate([log, exp], axis=0)
    f = relay.Function([x, y], concat)
    mod = tvm.IRModule()
    mod["main"] = f
    mod = WhiteListAnnotator(["add", "subtract", "multiply"], "ccompiler")(mod)
    mod = transform.PartitionGraph()(mod)

    fused_mod = transform.FuseOps(2)(mod)
    expected_mod = expected()
    assert tvm.ir.structural_equal(fused_mod, expected_mod, map_free_vars=True)

    x_data = np.random.rand(8, 8).astype('float32')
    y_data = np.random.rand(8, 8).astype('float32')
    np_add = x_data + y_data
    res = np.concatenate([np.log(np_add), np.exp(np_add)])
    check_result(mod, {"x": x_data, "y": y_data}, (16, 8), res)


def test_extern_ccompiler():
    x = relay.var('x', shape=(2, 2))
    y = relay.var('y', shape=(2, 2))
    z = x + x
    p = y * y
    f = relay.Function([x, y], p - z)
    x_data = np.random.rand(2, 2).astype('float32')
    y_data = np.random.rand(2, 2).astype('float32')
    mod = tvm.IRModule()
    mod["main"] = f
    mod = WhiteListAnnotator(["add", "subtract", "multiply"], "ccompiler")(mod)
    mod = transform.PartitionGraph()(mod)

    check_result(mod, {"x": x_data, "y": y_data}, (2, 2), (y_data * y_data) - (x_data + x_data))


def test_extern_dnnl():
    if not tvm.get_global_func("relay.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = 'float32'
    ishape = (1, 32, 14, 14)
    w1shape = (32, 1, 3, 3)

    def expected():
        data0 = relay.var("data", shape=(ishape), dtype=dtype)
        input0 = relay.var("input0", shape=(w1shape), dtype=dtype)
        input1 = relay.var("input1", shape=(w1shape), dtype=dtype)
        depthwise_conv2d_1 = relay.nn.conv2d(data0,
                                             input0,
                                             kernel_size=(3, 3),
                                             padding=(1, 1),
                                             groups=32)
        depthwise_conv2d_2 = relay.nn.conv2d(depthwise_conv2d_1,
                                             input1,
                                             kernel_size=(3, 3),
                                             padding=(1, 1),
                                             groups=32)
        out = relay.add(depthwise_conv2d_1, depthwise_conv2d_2)

        func = relay.Function([data0, input0, input1], out)
        func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        func = func.with_attr("Compiler", "dnnl")
        func = func.with_attr("global_symbol", "dnnl_0")
        glb_var = relay.GlobalVar("dnnl_0")
        mod = tvm.IRModule()
        mod[glb_var] = func

        data = relay.var("data", shape=(ishape), dtype=dtype)
        weight = relay.var("input", shape=(w1shape), dtype=dtype)
        main_f = relay.Function([data, weight], glb_var(data, weight, weight))
        mod["main"] = main_f

        return mod

    def get_func():
        data = relay.var("data", shape=(ishape), dtype=dtype)
        weight1 = relay.var("weight1", shape=(w1shape), dtype=dtype)
        depthwise_conv2d_1 = relay.nn.conv2d(data,
                                             weight1,
                                             kernel_size=(3, 3),
                                             padding=(1, 1),
                                             groups=32)
        depthwise_conv2d_2 = relay.nn.conv2d(depthwise_conv2d_1,
                                             weight1,
                                             kernel_size=(3, 3),
                                             padding=(1, 1),
                                             groups=32)
        out = relay.add(depthwise_conv2d_1, depthwise_conv2d_2)

        return relay.Function([data, weight1], out)

    mod = tvm.IRModule()
    mod["main"] = WholeGraphAnnotator("dnnl").visit(get_func())
    mod = transform.PartitionGraph()(mod)

    assert tvm.ir.structural_equal(mod, expected(), map_free_vars=True)

    ref_mod = tvm.IRModule()
    ref_mod["main"] = get_func()

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w1_data = np.random.uniform(0, 1, w1shape).astype(dtype)

    ref_ex = relay.create_executor("graph", mod=ref_mod, ctx=tvm.cpu())
    ref_res = ref_ex.evaluate()(i_data, w1_data)
    check_result(mod, {"data": i_data, "weight1": w1_data},
                 (1, 32, 14, 14), ref_res.asnumpy(), tol=1e-5)

@pytest.mark.skip(reason="fix constant node before opening this case")
def test_extern_dnnl_mobilenet():
    if not tvm.get_global_func("relay.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = 'float32'
    ishape = (1, 3, 224, 224)
    mod, params = relay.testing.mobilenet.get_workload(
        batch_size=1, dtype='float32')

    mod["main"] = bind_params_by_name(mod["main"], params)
    mod = transform.AnnotateTarget(["dnnl"])(mod)
    mod = transform.MergeCompilerRegions()(mod)
    mod = transform.PartitionGraph()(mod)
    i_data = np.random.uniform(0, 1, ishape).astype(dtype)

    ref_mod, params = relay.testing.mobilenet.get_workload(batch_size=1,
                                                           dtype='float32')
    ref_ex = relay.create_executor("graph", mod=ref_mod, ctx=tvm.cpu(0))
    ref_res = ref_ex.evaluate()(i_data, **params)

    check_result(mod, {"data": i_data},
                 (1, 1000), ref_res.asnumpy(), tol=1e-5, params=params)


def test_function_lifting():
    def partition():
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("weight", relay.TensorType((16, 3, 3, 3), "float32"))
        bn_gamma = relay.var("bn_gamma", relay.TensorType((16, ), "float32"))
        bn_beta = relay.var("bn_beta", relay.TensorType((16, ), "float32"))
        bn_mmean = relay.var("bn_mean", relay.TensorType((16, ), "float32"))
        bn_mvar = relay.var("bn_var", relay.TensorType((16, ), "float32"))

        conv = relay.nn.conv2d(
            data=data,
            weight=weight,
            kernel_size=(3, 3),
            channels=16,
            padding=(1, 1))
        bn_output = relay.nn.batch_norm(conv, bn_gamma, bn_beta, bn_mmean,
                                        bn_mvar)

        func = relay.Function([data, weight, bn_gamma, bn_beta, bn_mmean,
                               bn_mvar], bn_output.astuple())
        mod = tvm.IRModule()
        mod["main"] = func
        op_list = ["nn.batch_norm", "nn.conv2d"]
        mod = WhiteListAnnotator(op_list, "test_compiler")(mod)

        opt_pass = transform.Sequential([
            transform.InferType(),
            transform.PartitionGraph(),
            transform.SimplifyInference(),
            transform.FoldConstant(),
            transform.AlterOpLayout(),
        ])

        with relay.build_config(opt_level=3):
            mod = opt_pass(mod)

        return mod

    def expected():
        # function for batch_norm
        data0 = relay.var("data0", relay.TensorType((1, 16, 224, 224),
                                                    "float32"))
        mod = tvm.IRModule()
        bn_gamma = relay.var("bn_gamma1", relay.TensorType((16, ), "float32"))
        bn_beta = relay.var("bn_beta1", relay.TensorType((16, ), "float32"))
        bn_mmean = relay.var("bn_mean1", relay.TensorType((16, ), "float32"))
        bn_mvar = relay.var("bn_var1", relay.TensorType((16, ), "float32"))

        bn = relay.nn.batch_norm(data0, bn_gamma, bn_beta, bn_mmean, bn_mvar)
        func0 = relay.Function([data0, bn_gamma, bn_beta, bn_mmean, bn_mvar],
                               bn.astuple())
        func0 = func0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func0 = func0.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        func0 = func0.with_attr("Compiler", "test_compiler")
        func0 = func0.with_attr("global_symbol", "test_compiler_0")
        gv0 = relay.GlobalVar("test_compiler_0")
        mod[gv0] = func0

        # function for conv2d
        data1 = relay.var("data1", relay.TensorType((1, 3, 224, 224), "float32"))
        weight1 = relay.var("weight1", relay.TensorType((16, 3, 3, 3), "float32"))
        conv = relay.nn.conv2d(
            data=data1,
            weight=weight1,
            kernel_size=(3, 3),
            channels=16,
            padding=(1, 1))
        func1 = relay.Function([data1, weight1], conv)
        func1 = func1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func1 = func1.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        func1 = func1.with_attr("Compiler", "test_compiler")
        func1 = func1.with_attr("global_symbol", "test_compiler_1")
        gv1 = relay.GlobalVar("test_compiler_1")
        mod[gv1] = func1

        # main function
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("weight", relay.TensorType((16, 3, 3, 3), "float32"))
        bn_gamma0 = relay.var("bn_gamma", relay.TensorType((16, ), "float32"))
        bn_beta0 = relay.var("bn_beta", relay.TensorType((16, ), "float32"))
        bn_mmean0 = relay.var("bn_mean", relay.TensorType((16, ), "float32"))
        bn_mvar0 = relay.var("bn_var", relay.TensorType((16, ), "float32"))

        call1 = gv1(data, weight)
        call0 = gv0(call1, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0)
        mod["main"] = relay.Function([data, weight, bn_gamma0, bn_beta0, bn_mmean0,
                                      bn_mvar0], call0)
        mod = transform.InferType()(mod)
        return mod

    partitioned = partition()
    ref_mod = expected()
    assert tvm.ir.structural_equal(partitioned, ref_mod, map_free_vars=True)


def test_function_lifting_inline():
    def partition():
        data = relay.var("data", relay.TensorType((1, 16, 224, 224), "float32"))
        bn_gamma = relay.var("bn_gamma", relay.TensorType((16, ), "float32"))
        bn_beta = relay.var("bn_beta", relay.TensorType((16, ), "float32"))
        bn_mmean = relay.var("bn_mean", relay.TensorType((16, ), "float32"))
        bn_mvar = relay.var("bn_var", relay.TensorType((16, ), "float32"))

        bn_output = relay.nn.batch_norm(data, bn_gamma, bn_beta, bn_mmean,
                                        bn_mvar)

        func = relay.Function([data, bn_gamma, bn_beta, bn_mmean,
                               bn_mvar], bn_output.astuple())
        mod = tvm.IRModule()
        mod["main"] = func
        op_list = ["nn.batch_norm", "nn.conv2d"]
        mod = WhiteListAnnotator(op_list, "test_compiler")(mod)

        opt_pass = transform.Sequential([
            transform.InferType(),
            transform.PartitionGraph(),
            transform.SimplifyInference(),
            transform.FoldConstant(),
            transform.AlterOpLayout(),
            transform.Inline(),
        ])

        with relay.build_config(opt_level=3):
            mod = opt_pass(mod)

        return mod

    def expected():
        # function for batch_norm
        data0 = relay.var("data0", relay.TensorType((1, 16, 224, 224),
                                                    "float32"))
        mod = tvm.IRModule()
        bn_gamma = relay.var("bn_gamma1", relay.TensorType((16, ), "float32"))
        bn_beta = relay.var("bn_beta1", relay.TensorType((16, ), "float32"))
        bn_mmean = relay.var("bn_mean1", relay.TensorType((16, ), "float32"))
        bn_mvar = relay.var("bn_var1", relay.TensorType((16, ), "float32"))

        bn = relay.nn.batch_norm(data0, bn_gamma, bn_beta, bn_mmean, bn_mvar)
        func0 = relay.Function([data0, bn_gamma, bn_beta, bn_mmean, bn_mvar],
                               bn.astuple())
        func0 = func0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func0 = func0.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        func0 = func0.with_attr("Compiler", "test_compiler")
        func0 = func0.with_attr("global_symbol", "test_compiler_0")

        # main function
        data = relay.var("data", relay.TensorType((1, 16, 224, 224), "float32"))
        bn_gamma0 = relay.var("bn_gamma", relay.TensorType((16, ), "float32"))
        bn_beta0 = relay.var("bn_beta", relay.TensorType((16, ), "float32"))
        bn_mmean0 = relay.var("bn_mean", relay.TensorType((16, ), "float32"))
        bn_mvar0 = relay.var("bn_var", relay.TensorType((16, ), "float32"))

        call0 = func0(data, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0)
        mod["main"] = relay.Function([data, bn_gamma0, bn_beta0, bn_mmean0,
                                      bn_mvar0], call0)
        mod = transform.InferType()(mod)
        return mod

    partitioned = partition()
    ref_mod = expected()
    assert tvm.ir.structural_equal(partitioned, ref_mod, map_free_vars=True)


def test_constant_propagation():
    ones = np.ones(shape=(8, 8), dtype="float32")

    def expected():
        mod = tvm.IRModule()
        x = relay.const(ones)
        y = relay.var("y", shape=(8, 8))
        x0 = relay.const(ones)
        y0 = relay.var("y0", shape=(8, 8))
        add = x0 + y0
        # Function that uses C compiler
        func = relay.Function([y0], add)
        func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        func = func.with_attr("Compiler", "ccompiler")
        func = func.with_attr("global_symbol", "ccompiler_0")
        glb_0 = relay.GlobalVar("ccompiler_0")
        mod[glb_0] = func
        add_call = relay.Call(glb_0, [y])
        log = relay.log(add_call)
        main = relay.Function([y], log)
        mod["main"] = main
        return mod

    x = relay.var("x", shape=(8, 8))
    y = relay.var("y", shape=(8, 8))
    add = x + y
    log = relay.log(add)
    f = relay.Function([x, y], log)
    f = bind_params_by_name(f, {"x": tvm.nd.array(ones)})
    mod = tvm.IRModule()
    mod["main"] = f
    mod = WhiteListAnnotator(["add"], "ccompiler")(mod)
    mod = transform.PartitionGraph()(mod)

    expected_mod = expected()
    assert tvm.ir.structural_equal(mod, expected_mod, map_free_vars=True)

    y_data = np.random.rand(8, 8).astype('float32')
    np_add = ones + y_data
    check_result(mod, {"y": y_data}, (8, 8), np.log(np_add))


def test_multiple_outputs():

    def create_graph():
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("weight", relay.TensorType((16, 3, 3, 3), "float32"))
        bn_gamma = relay.var("bn_gamma", relay.TensorType((16, ), "float32"))
        bn_beta = relay.var("bn_beta", relay.TensorType((16, ), "float32"))
        bn_mean = relay.var("bn_mean", relay.TensorType((16, ), "float32"))
        bn_var = relay.var("bn_var", relay.TensorType((16, ), "float32"))

        data_cb = compiler_begin(data, 'test_target')
        weight_cb = compiler_begin(weight, 'test_target')
        bn_gamma_cb = compiler_begin(bn_gamma, 'test_target')
        bn_beta_cb = compiler_begin(bn_beta, 'test_target')
        bn_mean_cb = compiler_begin(bn_mean, 'test_target')
        bn_var_cb = compiler_begin(bn_var, 'test_target')

        conv_o = relay.nn.conv2d(
            data=data_cb,
            weight=weight_cb,
            kernel_size=(3, 3),
            channels=16,
            padding=(1, 1))

        bn_o = relay.nn.batch_norm(conv_o, bn_gamma_cb, bn_beta_cb, bn_mean_cb,
                                   bn_var_cb)

        relu_o = relay.nn.relu(bn_o[0])
        relu_o_ce = compiler_end(relu_o, 'test_target')

        bn_omean = bn_o[1]
        rebn_omean_ce = compiler_end(bn_omean, 'test_target')
        bn_ovar = bn_o[2]
        bn_ovar_ce = compiler_end(bn_ovar, 'test_target')

        dummy_mean_abs = relay.abs(rebn_omean_ce)
        dummy_ovar_abs = relay.abs(bn_ovar_ce)
        dummy_tuple = relay.Tuple((relu_o_ce, dummy_mean_abs,dummy_ovar_abs))

        func = relay.Function([data, weight, bn_gamma, bn_beta,
                               bn_mean, bn_var], dummy_tuple)
        return func

    def expected():
        mod = tvm.IRModule()

        # function 0
        data = relay.var("test_target_2_i0", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("test_target_2_i1", relay.TensorType((16, 3, 3, 3), "float32"))
        bn_gamma = relay.var("test_target_2_i2", relay.TensorType((16, ), "float32"))
        bn_beta = relay.var("test_target_2_i3", relay.TensorType((16, ), "float32"))
        bn_mean = relay.var("test_target_2_i4", relay.TensorType((16, ), "float32"))
        bn_var = relay.var("test_target_2_i5", relay.TensorType((16, ), "float32"))

        conv_o = relay.nn.conv2d(
            data=data,
            weight=weight,
            kernel_size=(3, 3),
            channels=16,
            padding=(1, 1))

        bn_o = relay.nn.batch_norm(conv_o, bn_gamma, bn_beta, bn_mean,
                                   bn_var)

        relu_o = relay.nn.relu(bn_o[0])
        tuple_o = relay.Tuple((bn_o[2], bn_o[1], relu_o))

        func0 = relay.Function([data, weight, bn_gamma, bn_beta,
                                bn_mean, bn_var], tuple_o)
        func0 = func0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func0 = func0.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        func0 = func0.with_attr("Compiler", "test_target")
        func0 = func0.with_attr("global_symbol", "test_target_2")
        gv0 = relay.GlobalVar("test_target_2")
        mod[gv0] = func0

        # body
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("weight", relay.TensorType((16, 3, 3, 3), "float32"))
        bn_gamma = relay.var("bn_gamma", relay.TensorType((16, ), "float32"))
        bn_beta = relay.var("bn_beta", relay.TensorType((16, ), "float32"))
        bn_mean = relay.var("bn_mean", relay.TensorType((16, ), "float32"))
        bn_var = relay.var("bn_var", relay.TensorType((16, ), "float32"))

        f0_o = gv0(data, weight, bn_gamma, bn_beta, bn_mean, bn_var)
        f0_relu_o = relay.TupleGetItem(f0_o, 2)
        f0_mean_o = relay.TupleGetItem(f0_o, 1)
        f0_var_o = relay.TupleGetItem(f0_o, 0)

        f0_mean_abs = relay.abs(f0_mean_o)
        f0_var_abs = relay.abs(f0_var_o)
        main_tuple = relay.Tuple((f0_relu_o, f0_mean_abs, f0_var_abs))

        func = relay.Function([data, weight, bn_gamma,
                               bn_beta, bn_mean, bn_var], main_tuple)
        mod["main"] = func
        return mod

    mod = tvm.IRModule()
    mod["main"] = create_graph()
    ref_mod = expected()
    partitioned = transform.PartitionGraph()(mod)
    assert tvm.ir.structural_equal(partitioned, ref_mod, map_free_vars=True)


def test_mixed_single_multiple_outputs():
    def create_graph():
        data = relay.var('data', shape=(10, 10))

        cb_1 = compiler_begin(data, 'test_target')
        O_1 = relay.abs(cb_1)
        ce_2 = compiler_end(O_1, 'test_target')
        O_2 = relay.nn.relu(O_1)
        ce_3 = compiler_end(O_2, 'test_target')

        X = relay.tanh(ce_2)

        cb_3 = compiler_begin(ce_3, 'test_target')
        cb_4 = compiler_begin(X, 'test_target')
        O_3 = relay.add(cb_3, cb_4)
        ce_4 = compiler_end(O_3, 'test_target')

        func = relay.Function([data], ce_4)
        return func

    def expected():
        mod = tvm.IRModule()

        # function 1
        f1_cb1 = relay.var('test_target_1_i0', shape=(10, 10))
        f1_O_1 = relay.abs(f1_cb1)
        f1_O_2 = relay.nn.relu(f1_O_1)
        f1_out = relay.Tuple((f1_O_2, f1_O_1))
        func1 = relay.Function([f1_cb1], f1_out)

        func1 = func1.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func1 = func1.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        func1 = func1.with_attr("Compiler", "test_target")
        func1 = func1.with_attr("global_symbol", "test_target_1")
        gv1 = relay.GlobalVar("test_target_1")
        mod[gv1] = func1

        # function 0
        f2_cb3 = relay.var('test_target_0_i0', shape=(10, 10))
        f2_cb4 = relay.var('test_target_0_i1', shape=(10, 10))
        f2_O_3 = relay.add(f2_cb3, f2_cb4)
        func0 = relay.Function([f2_cb3, f2_cb4], f2_O_3)

        func0 = func0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func0 = func0.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        func0 = func0.with_attr("Compiler", "test_target")
        func0 = func0.with_attr("global_symbol", "test_target_0")
        gv0 = relay.GlobalVar("test_target_0")
        mod[gv0] = func0

        # body
        data = relay.var('data', shape=(10, 10))
        tuple_out = gv1(data)
        ce_2 = relay.TupleGetItem(tuple_out, 1)
        ce_3 = relay.TupleGetItem(tuple_out, 0)

        X = relay.tanh(ce_2)
        ce_4 = gv0(ce_3, X)
        func = relay.Function([data], ce_4)
        mod["main"] = func

        return mod

    mod = tvm.IRModule()
    mod["main"] = create_graph()

    ref_mod = expected()
    partitioned = transform.PartitionGraph()(mod)
    assert tvm.ir.structural_equal(partitioned, ref_mod, map_free_vars=True)


def test_dnnl_fuse():
    def make_pattern(with_bias=True):
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("weight")
        bias = relay.var("bias")
        conv = relay.nn.conv2d(data=data, weight=weight, kernel_size=(3, 3),
                               channels=8, padding=(1, 1))
        if with_bias:
            conv_out = relay.add(conv, bias)
        else:
            conv_out = conv
        return relay.nn.relu(conv_out)

    conv2d_bias_relu_pat = ("dnnl.conv2d_bias_relu", make_pattern(with_bias=True))
    conv2d_relu_pat = ("dnnl.conv2d_relu", make_pattern(with_bias=False))
    dnnl_patterns = [conv2d_bias_relu_pat, conv2d_relu_pat]

    def get_blocks(prefix, data, in_channel, out_channel,
                   include_bn=True, include_sigmoid=False):
        weight = relay.var(prefix + "weight")
        bn_gamma = relay.var(prefix + "bn_gamma")
        bn_beta = relay.var(prefix + "bn_beta")
        bn_mmean = relay.var(prefix + "bn_mean")
        bn_mvar = relay.var(prefix + "bn_var")

        layer = relay.nn.conv2d(data=data, weight=weight, kernel_size=(3, 3),
                                channels=out_channel, padding=(1, 1))
        if include_bn:
            bn_output = relay.nn.batch_norm(layer, bn_gamma, bn_beta,
                                            bn_mmean, bn_mvar)
            layer = bn_output[0]
        if include_sigmoid:
            # dummy layer to prevent pattern detection
            layer = relay.sigmoid(layer)
        layer = relay.nn.relu(layer)
        return layer

    def get_net(include_bn=True, include_sigmoid=False):
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        block1 = get_blocks("block1_", data, 3, 8, include_bn, include_sigmoid)
        # The second block is always conv + relu, to make it more interesting
        block2 = get_blocks("block2_", block1, 8, 8, False, include_sigmoid)
        return relay.Function(relay.analysis.free_vars(block2), block2)

    def get_partitoned_mod(mod, params, pattern_table):
        # This is required for constant folding
        mod["main"] = bind_params_by_name(mod["main"], params)

        remove_bn_pass = transform.Sequential([
            transform.InferType(),
            transform.SimplifyInference(),
            transform.FoldConstant(),
            transform.FoldScaleAxis(),
        ])
        composite_partition = transform.Sequential([
            remove_bn_pass,
            transform.MergeComposite(pattern_table),
            transform.AnnotateTarget("dnnl"),
            transform.PartitionGraph()
        ])

        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
            return composite_partition(mod)

    def test_detect_pattern(pattern_table, include_bn, include_sigmoid,
                            num_expected_partition):
        net = get_net(include_bn, include_sigmoid)
        mod, params = tvm.relay.testing.create_workload(net)
        mod = get_partitoned_mod(mod, params, pattern_table)
        assert(len(mod.functions) - 1 == num_expected_partition)  # -1 for main

    def test_partition():
        # conv + bn + relu, conv + relu -> fused conv_bias_relu, conv, and relu
        test_detect_pattern([conv2d_bias_relu_pat], True, False, 3)
        # conv + bn + relu, conv + relu -> conv, bias, relu, and fused conv_relu
        test_detect_pattern([conv2d_relu_pat], True, False, 4)
        # conv + bn + relu, conv + relu -> fused conv_bias_relu, and fused conv_relu
        test_detect_pattern([conv2d_bias_relu_pat, conv2d_relu_pat], True, False, 2)
        # conv + relu, conv + relu -> two fused conv_relu
        test_detect_pattern([conv2d_relu_pat], False, False, 2)
        # conv + relu, conv + relu -> no fusion, 4 partition each with a single op
        test_detect_pattern([conv2d_bias_relu_pat], False, False, 4)
        # conv + bn + sigmoid + relu, conv + sigmoid + relu -> no fusion
        test_detect_pattern([conv2d_bias_relu_pat, conv2d_relu_pat], True, True, 5)

    def test_partition_mobilenet():
        mod, params = relay.testing.mobilenet.get_workload()
        mod = get_partitoned_mod(mod, params, dnnl_patterns)
        # 27 fused conv + bn + relu and one dense
        assert(len(mod.functions) - 1 == 28)  # -1 for main

    def test_exec(mod, params, ref_mod, ref_params, out_shape):
        ishape = (1, 3, 224, 224)
        i_data = np.random.randn(*ishape).astype(np.float32)
        ref_ex = relay.create_executor("graph", mod=ref_mod, ctx=tvm.cpu(0))
        ref_res = ref_ex.evaluate()(i_data, **ref_params)
        compile_engine.get().clear()

        mod = get_partitoned_mod(mod, params, dnnl_patterns)

        check_result(mod, {"data": i_data},
                     out_shape, ref_res.asnumpy(), tol=1e-5, params=params)

    test_partition()
    test_partition_mobilenet()

    if not tvm.get_global_func("relay.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    net = get_net()
    mod, params = tvm.relay.testing.create_workload(net)
    ref_mod, ref_params = tvm.relay.testing.create_workload(net)
    test_exec(mod, params, ref_mod, ref_params, (1, 8, 224, 224))

    # exec test on mobilenet is not possible due to manually inlined constants
    # mod, params = relay.testing.mobilenet.get_workload()
    # ref_mod, ref_params = relay.testing.mobilenet.get_workload()
    # test_exec(mod, params, ref_mod, ref_params, (1, 1000))


if __name__ == "__main__":
    test_multi_node_compiler()
    test_extern_ccompiler_single_op()
    test_extern_ccompiler_default_ops()
    test_extern_ccompiler()
    test_extern_dnnl()
    # TODO(@comaniac, @zhiics): Fix constant node and re-open this case.
    #test_extern_dnnl_mobilenet()
    test_function_lifting()
    test_function_lifting_inline()
    test_constant_propagation()
    test_multiple_outputs()
    test_mixed_single_multiple_outputs()
    test_dnnl_fuse()
