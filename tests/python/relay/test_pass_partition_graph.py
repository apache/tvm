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
# pylint: disable=not-callable
import os
import sys

import numpy as np

import tvm
from tvm.relay.backend import te_compiler
from tvm.relay.backend.runtime import Runtime
import tvm.relay.testing
import tvm.relay.op as reg
from tvm import relay
from tvm import runtime as tvm_runtime
from tvm.relay import transform
from tvm.relay.testing import byoc
from tvm.contrib import utils
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.op.contrib.register import get_pattern_table
from tvm.relay.build_module import bind_params_by_name


# Leverage the pass manager to write a simple allowed list based annotator
@transform.function_pass(opt_level=0)
class AllowedListAnnotator:
    def __init__(self, op_list, compiler):
        assert isinstance(op_list, (list, tuple, set))
        self.op_list = op_list
        self.compiler = compiler

    def transform_function(self, func, mod, dev):

        annotator = self

        class Annotator(tvm.relay.ExprMutator):
            def visit_call(self, call):
                op_name = call.op.name
                if op_name in annotator.op_list:
                    new_args = []
                    for arg in call.args:
                        ann = compiler_begin(super().visit(arg), annotator.compiler)
                        new_args.append(ann)
                    new_call = relay.Call(call.op, new_args, call.attrs, call.type_args)
                    return compiler_end(new_call, annotator.compiler)
                else:
                    return super().visit_call(call)

        return Annotator().visit(func)


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

        if call.op.name == "nn.global_avg_pool2d":
            self.compiler_open = True
        compiler_open = self.compiler_open

        params = []
        for arg in call.args:
            param = super().visit(arg)
            if call.op.name == "nn.global_avg_pool2d":
                param = compiler_end(param, self.compiler)
            if compiler_open and isinstance(param, relay.expr.Var):
                param = compiler_begin(param, self.compiler)
            params.append(param)

        new_call = relay.Call(call.op, params, call.attrs)
        return new_call


def check_result(
    mod,
    map_inputs,
    out_shape,
    result,
    tol=1e-5,
    target="llvm",
    device=tvm.cpu(),
    params=None,
    runtime=Runtime("cpp"),
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
        lib = tvm_runtime.load_module(lib_path)

        return lib

    def check_vm_result():
        te_compiler.get().clear()
        with tvm.transform.PassContext(opt_level=3):
            exe = relay.vm.compile(mod, target=target, params=params)
        code, lib = exe.save()
        lib = update_lib(lib)
        exe = tvm_runtime.vm.Executable.load_exec(code, lib)
        vm = tvm_runtime.vm.VirtualMachine(exe, device)
        outs = vm.run(**map_inputs)
        outs = outs if isinstance(outs, tvm_runtime.container.ADT) else [outs]
        results = result if isinstance(result, list) else [result]
        for out, ref in zip(outs, results):
            tvm.testing.assert_allclose(out.numpy(), ref, rtol=tol, atol=tol)

    def check_graph_executor_result():
        te_compiler.get().clear()
        with tvm.transform.PassContext(opt_level=3):
            json, lib, param = relay.build(mod, target=target, params=params, runtime=runtime)
        lib = update_lib(lib)
        rt_mod = tvm.contrib.graph_executor.create(json, lib, device)

        for name, data in map_inputs.items():
            rt_mod.set_input(name, data)
        rt_mod.set_input(**param)
        rt_mod.run()

        out_shapes = out_shape if isinstance(out_shape, list) else [out_shape]
        results = result if isinstance(result, list) else [result]

        for idx, shape in enumerate(out_shapes):
            out = tvm.nd.empty(shape, device=device)
            out = rt_mod.get_output(idx, out)
            tvm.testing.assert_allclose(out.numpy(), results[idx], rtol=tol, atol=tol)

    check_vm_result()
    check_graph_executor_result()


def test_extern_ccompiler_single_op():
    @transform.function_pass(opt_level=0)
    class MyAnnotator:
        def transform_function(self, func, mod, dev):
            class Annotator(tvm.relay.ExprMutator):
                def visit_call(self, call):
                    new_args = []
                    for arg in call.args:
                        ann = compiler_begin(self.visit(arg), "ccompiler")
                        new_args.append(ann)
                    new_call = relay.Call(call.op, new_args)
                    return compiler_end(new_call, "ccompiler")

            return Annotator().visit(func)

    x = relay.var("x", shape=(8, 8))
    y = relay.var("y", shape=(8, 8))
    z = x + y
    f = relay.Function([x, y], z)
    x_data = np.random.rand(8, 8).astype("float32")
    y_data = np.random.rand(8, 8).astype("float32")
    mod = tvm.IRModule()
    mod["main"] = f
    mod = MyAnnotator()(mod)
    mod = transform.PartitionGraph()(mod)

    check_result(mod, {"x": x_data, "y": y_data}, (8, 8), x_data + y_data)


def set_func_attr(func, compile_name, symbol_name):
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", compile_name)
    func = func.with_attr("global_symbol", symbol_name)
    return func


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
        func = set_func_attr(func, "ccompiler", "tvmgen_default_ccompiler_main_0")
        glb_0 = relay.GlobalVar("tvmgen_default_ccompiler_main_0")
        mod[glb_0] = func
        add_call = relay.Call(glb_0, [x, y])
        # Function that uses default compiler. Ops are fused in this function.
        p0 = relay.var("p0", shape=(8, 8))
        log = relay.log(p0)
        exp = relay.exp(p0)
        concat = relay.concatenate([log, exp], axis=0)
        fused_func = relay.Function([p0], concat)
        fused_func = fused_func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        fused_call = relay.Call(fused_func, [add_call])
        main = relay.Function([x, y], fused_call)
        mod["main"] = main
        mod = transform.InferType()(mod)
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
    mod = AllowedListAnnotator(["add", "subtract", "multiply"], "ccompiler")(mod)
    mod = transform.PartitionGraph()(mod)
    fused_mod = transform.FuseOps(2)(mod)
    expected_mod = expected()
    tvm.ir.assert_structural_equal(fused_mod, expected_mod, map_free_vars=True)

    x_data = np.random.rand(8, 8).astype("float32")
    y_data = np.random.rand(8, 8).astype("float32")
    np_add = x_data + y_data
    res = np.concatenate([np.log(np_add), np.exp(np_add)])
    check_result(mod, {"x": x_data, "y": y_data}, (16, 8), res)


def test_extern_compiler_sanitized_ops():
    def expected():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(8, 8))
        y = relay.var("y", shape=(8, 8))
        x0 = relay.var("x0", shape=(8, 8))
        y0 = relay.var("y0", shape=(8, 8))
        add = x0 + y0
        # Function that uses C compiler
        func = relay.Function([x0, y0], add)
        func = set_func_attr(func, "unsanitary-name++", "tvmgen_default_unsanitary_name___main_0")
        glb_0 = relay.GlobalVar("tvmgen_default_unsanitary_name___main_0")
        mod[glb_0] = func
        add_call = relay.Call(glb_0, [x, y])
        # Function that uses default compiler. Ops are fused in this function.
        p0 = relay.var("p0", shape=(8, 8))
        log = relay.log(p0)
        exp = relay.exp(p0)
        concat = relay.concatenate([log, exp], axis=0)
        fused_func = relay.Function([p0], concat)
        fused_func = fused_func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        fused_call = relay.Call(fused_func, [add_call])
        main = relay.Function([x, y], fused_call)
        mod["main"] = main
        mod = transform.InferType()(mod)
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
    mod = AllowedListAnnotator(["add", "subtract", "multiply"], "unsanitary-name++")(mod)
    mod = transform.PartitionGraph()(mod)
    fused_mod = transform.FuseOps(2)(mod)
    expected_mod = expected()
    tvm.ir.assert_structural_equal(fused_mod, expected_mod, map_free_vars=True)


def test_extern_ccompiler_multiple_functions():
    def expected():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(8, 8))
        y = relay.var("y", shape=(8, 8))
        x0 = relay.var("x0", shape=(8, 8))
        y0 = relay.var("y0", shape=(8, 8))
        add = x0 + y0
        # Function that uses C compiler
        func = relay.Function([x0, y0], add)
        func = set_func_attr(func, "ccompiler", "tvmgen_default_ccompiler_main_0")
        glb_0 = relay.GlobalVar("tvmgen_default_ccompiler_main_0")
        mod[glb_0] = func
        add_call = relay.Call(glb_0, [x, y])
        # Function that uses default compiler. Ops are fused in this function.
        p0 = relay.var("p0", shape=(8, 8))
        log = relay.log(p0)
        exp = relay.exp(p0)
        concat = relay.concatenate([log, exp], axis=0)
        fused_func = relay.Function([p0], concat)
        fused_func = fused_func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        fused_call = relay.Call(fused_func, [add_call])
        main = relay.Function([x, y], fused_call)
        mod["main"] = main
        # define the second one
        a = relay.var("a", shape=(16, 16))
        b = relay.var("b", shape=(16, 16))
        a0 = relay.var("a0", shape=(16, 16))
        b0 = relay.var("b0", shape=(16, 16))
        add = a0 + b0
        # Function that uses C compiler
        func = relay.Function([a0, b0], add)
        func = set_func_attr(func, "ccompiler", "tvmgen_default_ccompiler_subfunction_0")
        glb_0 = relay.GlobalVar("tvmgen_default_ccompiler_subfunction_0")
        mod[glb_0] = func
        add_call = relay.Call(glb_0, [a, b])
        # Function that uses default compiler. Ops are fused in this function.
        p0 = relay.var("p0", shape=(16, 16))
        log = relay.log(p0)
        exp = relay.exp(p0)
        concat = relay.concatenate([log, exp], axis=0)
        fused_func = relay.Function([p0], concat)
        fused_func = fused_func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        fused_call = relay.Call(fused_func, [add_call])
        sunfunction = relay.Function([a, b], fused_call)
        mod["subfunction"] = sunfunction
        mod = transform.InferType()(mod)
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
    # define second function
    a = relay.var("a", shape=(16, 16))
    b = relay.var("b", shape=(16, 16))
    add = a + b
    log = relay.log(add)
    exp = relay.exp(add)
    concat = relay.concatenate([log, exp], axis=0)
    f2 = relay.Function([a, b], concat)
    mod["subfunction"] = f2
    mod = AllowedListAnnotator(["add", "subtract", "multiply"], "ccompiler")(mod)
    mod = transform.PartitionGraph()(mod)

    fused_mod = transform.FuseOps(2)(mod)
    expected_mod = expected()
    tvm.ir.assert_structural_equal(fused_mod, expected_mod, map_free_vars=True)

    x_data = np.random.rand(8, 8).astype("float32")
    y_data = np.random.rand(8, 8).astype("float32")
    np_add = x_data + y_data
    res = np.concatenate([np.log(np_add), np.exp(np_add)])
    check_result(mod, {"x": x_data, "y": y_data}, (16, 8), res)


def test_extern_ccompiler():
    x = relay.var("x", shape=(2, 2))
    y = relay.var("y", shape=(2, 2))
    z = x + x
    p = y * y
    f = relay.Function([x, y], p - z)
    x_data = np.random.rand(2, 2).astype("float32")
    y_data = np.random.rand(2, 2).astype("float32")
    mod = tvm.IRModule()
    mod["main"] = f
    mod = AllowedListAnnotator(["add", "subtract", "multiply"], "ccompiler")(mod)
    mod = transform.PartitionGraph()(mod)

    check_result(mod, {"x": x_data, "y": y_data}, (2, 2), (y_data * y_data) - (x_data + x_data))


def test_extern_dnnl():
    if not tvm.get_global_func("relay.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = "float32"
    ishape = (1, 32, 14, 14)
    w1shape = (32, 1, 3, 3)

    def expected():
        data0 = relay.var("data", shape=(ishape), dtype=dtype)
        input0 = relay.var("input", shape=(w1shape), dtype=dtype)
        depthwise_conv2d_1 = relay.nn.conv2d(
            data0, input0, kernel_size=(3, 3), padding=(1, 1), groups=32
        )
        depthwise_conv2d_2 = relay.nn.conv2d(
            depthwise_conv2d_1, input0, kernel_size=(3, 3), padding=(1, 1), groups=32
        )
        out = relay.add(depthwise_conv2d_1, depthwise_conv2d_2)

        func = relay.Function([data0, input0], out)
        func = set_func_attr(func, "dnnl", "tvmgen_default_dnnl_main_0")
        glb_var = relay.GlobalVar("tvmgen_default_dnnl_main_0")
        mod = tvm.IRModule()
        mod[glb_var] = func
        mod = transform.InferType()(mod)

        data = relay.var("data", shape=(ishape), dtype=dtype)
        weight = relay.var("input", shape=(w1shape), dtype=dtype)
        main_f = relay.Function([data, weight], glb_var(data, weight))
        mod["main"] = main_f
        mod = transform.InferType()(mod)

        return mod

    def get_func():
        data = relay.var("data", shape=(ishape), dtype=dtype)
        weight1 = relay.var("weight1", shape=(w1shape), dtype=dtype)
        depthwise_conv2d_1 = relay.nn.conv2d(
            data, weight1, kernel_size=(3, 3), padding=(1, 1), groups=32
        )
        depthwise_conv2d_2 = relay.nn.conv2d(
            depthwise_conv2d_1, weight1, kernel_size=(3, 3), padding=(1, 1), groups=32
        )
        out = relay.add(depthwise_conv2d_1, depthwise_conv2d_2)

        return relay.Function([data, weight1], out)

    mod = tvm.IRModule()
    mod["main"] = WholeGraphAnnotator("dnnl").visit(get_func())
    mod = transform.PartitionGraph()(mod)
    mod = transform.InferType()(mod)

    tvm.ir.assert_structural_equal(mod, expected(), map_free_vars=True)

    ref_mod = tvm.IRModule()
    ref_mod["main"] = get_func()

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w1_data = np.random.uniform(0, 1, w1shape).astype(dtype)

    ref_res = relay.create_executor("graph", mod=ref_mod, device=tvm.cpu()).evaluate()(
        i_data, w1_data
    )
    check_result(
        mod, {"data": i_data, "weight1": w1_data}, (1, 32, 14, 14), ref_res.numpy(), tol=1e-5
    )


def test_extern_dnnl_mobilenet():
    if not tvm.get_global_func("relay.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = "float32"
    ishape = (1, 3, 224, 224)
    ref_mod, params = relay.testing.mobilenet.get_workload(batch_size=1, dtype="float32")
    mod = transform.AnnotateTarget(["dnnl"])(ref_mod)
    mod = transform.MergeCompilerRegions()(mod)
    mod = transform.PartitionGraph()(mod)
    i_data = np.random.uniform(0, 1, ishape).astype(dtype)

    ref_res = relay.create_executor("graph", mod=ref_mod, device=tvm.cpu(0)).evaluate()(
        i_data, **params
    )
    te_compiler.get().clear()

    check_result(mod, {"data": i_data}, (1, 1000), ref_res.numpy(), tol=1e-5, params=params)


def test_function_lifting():
    def partition():
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("weight", relay.TensorType((16, 3, 3, 3), "float32"))
        bn_gamma = relay.var("bn_gamma", relay.TensorType((16,), "float32"))
        bn_beta = relay.var("bn_beta", relay.TensorType((16,), "float32"))
        bn_mmean = relay.var("bn_mean", relay.TensorType((16,), "float32"))
        bn_mvar = relay.var("bn_var", relay.TensorType((16,), "float32"))

        conv = relay.nn.conv2d(
            data=data, weight=weight, kernel_size=(3, 3), channels=16, padding=(1, 1)
        )
        bn_output = relay.nn.batch_norm(conv, bn_gamma, bn_beta, bn_mmean, bn_mvar)

        func = relay.Function(
            [data, weight, bn_gamma, bn_beta, bn_mmean, bn_mvar], bn_output.astuple()
        )
        mod = tvm.IRModule()
        mod["main"] = func
        mod = relay.transform.InferType()(mod)
        op_list = ["nn.batch_norm", "nn.conv2d"]
        mod = AllowedListAnnotator(op_list, "test_compiler")(mod)

        opt_pass = tvm.transform.Sequential(
            [
                transform.InferType(),
                transform.PartitionGraph(),
                transform.SimplifyInference(),
                transform.FoldConstant(),
                transform.AlterOpLayout(),
            ]
        )

        with tvm.transform.PassContext(opt_level=3):
            mod = opt_pass(mod)

        return mod

    def expected():
        # function for batch_norm
        data0 = relay.var("data0", relay.TensorType((1, 16, 224, 224), "float32"))
        mod = tvm.IRModule()
        bn_gamma = relay.var("bn_gamma1", relay.TensorType((16,), "float32"))
        bn_beta = relay.var("bn_beta1", relay.TensorType((16,), "float32"))
        bn_mmean = relay.var("bn_mean1", relay.TensorType((16,), "float32"))
        bn_mvar = relay.var("bn_var1", relay.TensorType((16,), "float32"))

        bn = relay.nn.batch_norm(data0, bn_gamma, bn_beta, bn_mmean, bn_mvar)
        func0 = relay.Function([data0, bn_gamma, bn_beta, bn_mmean, bn_mvar], bn.astuple())
        func0 = set_func_attr(func0, "test_compiler", "tvmgen_default_test_compiler_main_2")
        gv0 = relay.GlobalVar("tvmgen_default_test_compiler_main_2")
        mod[gv0] = func0
        mod = transform.InferType()(mod)

        # function for conv2d
        data1 = relay.var("data1", relay.TensorType((1, 3, 224, 224), "float32"))
        weight1 = relay.var("weight1", relay.TensorType((16, 3, 3, 3), "float32"))
        conv = relay.nn.conv2d(
            data=data1, weight=weight1, kernel_size=(3, 3), channels=16, padding=(1, 1)
        )
        func1 = relay.Function([data1, weight1], conv)
        func1 = set_func_attr(func1, "test_compiler", "tvmgen_default_test_compiler_main_0")
        gv1 = relay.GlobalVar("tvmgen_default_test_compiler_main_0")
        mod[gv1] = func1
        mod = transform.InferType()(mod)

        # main function
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("weight", relay.TensorType((16, 3, 3, 3), "float32"))
        bn_gamma0 = relay.var("bn_gamma", relay.TensorType((16,), "float32"))
        bn_beta0 = relay.var("bn_beta", relay.TensorType((16,), "float32"))
        bn_mmean0 = relay.var("bn_mean", relay.TensorType((16,), "float32"))
        bn_mvar0 = relay.var("bn_var", relay.TensorType((16,), "float32"))

        call1 = gv1(data, weight)
        call0 = gv0(call1, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0)
        mod["main"] = relay.Function(
            [data, weight, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0], call0
        )
        mod = transform.InferType()(mod)
        return mod

    partitioned = partition()
    ref_mod = expected()
    tvm.ir.assert_structural_equal(partitioned, ref_mod, map_free_vars=True)


def test_function_lifting_inline():
    def partition():
        data = relay.var("data", relay.TensorType((1, 16, 224, 224), "float32"))
        bn_gamma = relay.var("bn_gamma", relay.TensorType((16,), "float32"))
        bn_beta = relay.var("bn_beta", relay.TensorType((16,), "float32"))
        bn_mmean = relay.var("bn_mean", relay.TensorType((16,), "float32"))
        bn_mvar = relay.var("bn_var", relay.TensorType((16,), "float32"))

        bn_output = relay.nn.batch_norm(data, bn_gamma, bn_beta, bn_mmean, bn_mvar)

        func = relay.Function([data, bn_gamma, bn_beta, bn_mmean, bn_mvar], bn_output.astuple())
        mod = tvm.IRModule()
        mod["main"] = func
        op_list = ["nn.batch_norm", "nn.conv2d"]
        mod = AllowedListAnnotator(op_list, "test_compiler")(mod)

        opt_pass = tvm.transform.Sequential(
            [
                transform.InferType(),
                transform.PartitionGraph(),
                transform.SimplifyInference(),
                transform.FoldConstant(),
                transform.AlterOpLayout(),
                transform.Inline(),
            ]
        )

        with tvm.transform.PassContext(opt_level=3):
            mod = opt_pass(mod)

        return mod

    def expected():
        # function for batch_norm
        data0 = relay.var("data0", relay.TensorType((1, 16, 224, 224), "float32"))
        mod = tvm.IRModule()
        bn_gamma = relay.var("bn_gamma1", relay.TensorType((16,), "float32"))
        bn_beta = relay.var("bn_beta1", relay.TensorType((16,), "float32"))
        bn_mmean = relay.var("bn_mean1", relay.TensorType((16,), "float32"))
        bn_mvar = relay.var("bn_var1", relay.TensorType((16,), "float32"))

        bn = relay.nn.batch_norm(data0, bn_gamma, bn_beta, bn_mmean, bn_mvar)
        func0 = relay.Function([data0, bn_gamma, bn_beta, bn_mmean, bn_mvar], bn.astuple())
        func0 = set_func_attr(func0, "test_compiler", "tvmgen_default_test_compiler_main_0")

        # main function
        data = relay.var("data", relay.TensorType((1, 16, 224, 224), "float32"))
        bn_gamma0 = relay.var("bn_gamma", relay.TensorType((16,), "float32"))
        bn_beta0 = relay.var("bn_beta", relay.TensorType((16,), "float32"))
        bn_mmean0 = relay.var("bn_mean", relay.TensorType((16,), "float32"))
        bn_mvar0 = relay.var("bn_var", relay.TensorType((16,), "float32"))

        call0 = func0(data, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0)
        mod["main"] = relay.Function([data, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0], call0)
        mod = transform.InferType()(mod)
        return mod

    partitioned = partition()
    ref_mod = expected()
    tvm.ir.assert_structural_equal(partitioned, ref_mod, map_free_vars=True)


def test_constant_propagation():
    ones = np.ones(shape=(8, 8), dtype="float32")

    def expected():
        mod = tvm.IRModule()
        y = relay.var("y", shape=(8, 8))
        x0 = relay.const(ones)
        y0 = relay.var("y0", shape=(8, 8))
        add = x0 + y0
        # Function that uses C compiler
        func = relay.Function([y0], add)
        func = set_func_attr(func, "ccompiler", "tvmgen_default_ccompiler_main_0")
        glb_0 = relay.GlobalVar("tvmgen_default_ccompiler_main_0")
        mod[glb_0] = func
        mod = relay.transform.InferType()(mod)
        add_call = relay.Call(glb_0, [y])
        log = relay.log(add_call)
        main = relay.Function([y], log)
        mod["main"] = main
        mod = relay.transform.InferType()(mod)
        return mod

    x = relay.var("x", shape=(8, 8))
    y = relay.var("y", shape=(8, 8))
    add = x + y
    log = relay.log(add)
    f = relay.Function([x, y], log)
    f = bind_params_by_name(f, {"x": tvm.nd.array(ones)})
    mod = tvm.IRModule()
    mod["main"] = f
    mod = AllowedListAnnotator(["add"], "ccompiler")(mod)
    mod = transform.PartitionGraph()(mod)
    mod = relay.transform.InferType()(mod)

    expected_mod = expected()
    expected_mod = relay.transform.InferType()(expected_mod)
    tvm.ir.assert_structural_equal(mod, expected_mod, map_free_vars=True)

    y_data = np.random.rand(8, 8).astype("float32")
    np_add = ones + y_data
    check_result(mod, {"y": y_data}, (8, 8), np.log(np_add))


def test_multiple_outputs():
    def create_graph():
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("weight", relay.TensorType((16, 3, 3, 3), "float32"))
        bn_gamma = relay.var("bn_gamma", relay.TensorType((16,), "float32"))
        bn_beta = relay.var("bn_beta", relay.TensorType((16,), "float32"))
        bn_mean = relay.var("bn_mean", relay.TensorType((16,), "float32"))
        bn_var = relay.var("bn_var", relay.TensorType((16,), "float32"))

        data_cb = compiler_begin(data, "test_target")
        weight_cb = compiler_begin(weight, "test_target")
        bn_gamma_cb = compiler_begin(bn_gamma, "test_target")
        bn_beta_cb = compiler_begin(bn_beta, "test_target")
        bn_mean_cb = compiler_begin(bn_mean, "test_target")
        bn_var_cb = compiler_begin(bn_var, "test_target")

        conv_o = relay.nn.conv2d(
            data=data_cb, weight=weight_cb, kernel_size=(3, 3), channels=16, padding=(1, 1)
        )

        bn_o = relay.nn.batch_norm(conv_o, bn_gamma_cb, bn_beta_cb, bn_mean_cb, bn_var_cb)

        relu_o = relay.nn.relu(bn_o[0])
        relu_o_ce = compiler_end(relu_o, "test_target")

        bn_omean = bn_o[1]
        rebn_omean_ce = compiler_end(bn_omean, "test_target")
        bn_ovar = bn_o[2]
        bn_ovar_ce = compiler_end(bn_ovar, "test_target")

        dummy_mean_abs = relay.abs(rebn_omean_ce)
        dummy_ovar_abs = relay.abs(bn_ovar_ce)
        dummy_tuple = relay.Tuple((relu_o_ce, dummy_mean_abs, dummy_ovar_abs))

        func = relay.Function([data, weight, bn_gamma, bn_beta, bn_mean, bn_var], dummy_tuple)
        return func

    def expected():
        mod = tvm.IRModule()

        # function 0
        data = relay.var("test_target_0_i0", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("test_target_0_i1", relay.TensorType((16, 3, 3, 3), "float32"))
        bn_gamma = relay.var("test_target_0_i2", relay.TensorType((16,), "float32"))
        bn_beta = relay.var("test_target_0_i3", relay.TensorType((16,), "float32"))
        bn_mean = relay.var("test_target_0_i4", relay.TensorType((16,), "float32"))
        bn_var = relay.var("test_target_0_i5", relay.TensorType((16,), "float32"))

        conv_o = relay.nn.conv2d(
            data=data, weight=weight, kernel_size=(3, 3), channels=16, padding=(1, 1)
        )

        bn_o = relay.nn.batch_norm(conv_o, bn_gamma, bn_beta, bn_mean, bn_var)

        relu_o = relay.nn.relu(bn_o[0])
        tuple_o = relay.Tuple((relu_o, bn_o[1], bn_o[2]))

        func0 = relay.Function([data, weight, bn_gamma, bn_beta, bn_mean, bn_var], tuple_o)
        func0 = set_func_attr(func0, "test_target", "tvmgen_default_test_target_main_0")
        gv0 = relay.GlobalVar("tvmgen_default_test_target_main_0")
        mod[gv0] = func0
        mod = relay.transform.InferType()(mod)

        # body
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("weight", relay.TensorType((16, 3, 3, 3), "float32"))
        bn_gamma = relay.var("bn_gamma", relay.TensorType((16,), "float32"))
        bn_beta = relay.var("bn_beta", relay.TensorType((16,), "float32"))
        bn_mean = relay.var("bn_mean", relay.TensorType((16,), "float32"))
        bn_var = relay.var("bn_var", relay.TensorType((16,), "float32"))

        f0_o = gv0(data, weight, bn_gamma, bn_beta, bn_mean, bn_var)
        f0_relu_o = relay.TupleGetItem(f0_o, 0)
        f0_mean_o = relay.TupleGetItem(f0_o, 1)
        f0_var_o = relay.TupleGetItem(f0_o, 2)

        f0_mean_abs = relay.abs(f0_mean_o)
        f0_var_abs = relay.abs(f0_var_o)
        main_tuple = relay.Tuple((f0_relu_o, f0_mean_abs, f0_var_abs))

        func = relay.Function([data, weight, bn_gamma, bn_beta, bn_mean, bn_var], main_tuple)
        mod["main"] = func
        mod = relay.transform.InferType()(mod)
        return mod

    mod = tvm.IRModule()
    mod["main"] = create_graph()
    ref_mod = expected()
    partitioned = transform.PartitionGraph()(mod)
    tvm.ir.assert_structural_equal(partitioned, ref_mod, map_free_vars=True)


def test_mixed_single_multiple_outputs():
    def create_graph():
        data = relay.var("data", shape=(10, 10))

        cb_1 = compiler_begin(data, "test_target")
        O_1 = relay.abs(cb_1)
        ce_2 = compiler_end(O_1, "test_target")
        O_2 = relay.nn.relu(O_1)
        ce_3 = compiler_end(O_2, "test_target")

        X = relay.tanh(ce_2)

        cb_3 = compiler_begin(ce_3, "test_target")
        cb_4 = compiler_begin(X, "test_target")
        O_3 = relay.add(cb_3, cb_4)
        ce_4 = compiler_end(O_3, "test_target")

        func = relay.Function([data], ce_4)
        return func

    def expected():
        mod = tvm.IRModule()

        # function 1
        f1_cb1 = relay.var("test_target_0_i0", shape=(10, 10))
        f1_O_1 = relay.abs(f1_cb1)
        f1_O_2 = relay.nn.relu(f1_O_1)
        f1_out = relay.Tuple((f1_O_2, f1_O_1))
        func1 = relay.Function([f1_cb1], f1_out)
        func1 = set_func_attr(func1, "test_target", "tvmgen_default_test_target_main_0")
        gv1 = relay.GlobalVar("tvmgen_default_test_target_main_0")
        mod[gv1] = func1
        mod = relay.transform.InferType()(mod)

        # function 0
        f2_cb3 = relay.var("test_target_1_i0", shape=(10, 10))
        f2_cb4 = relay.var("test_target_1_i1", shape=(10, 10))
        f2_O_3 = relay.add(f2_cb3, f2_cb4)
        func0 = relay.Function([f2_cb3, f2_cb4], f2_O_3)
        func0 = set_func_attr(func0, "test_target", "tvmgen_default_test_target_main_1")
        gv0 = relay.GlobalVar("tvmgen_default_test_target_main_1")
        mod[gv0] = func0
        mod = relay.transform.InferType()(mod)

        # body
        data = relay.var("data", shape=(10, 10))
        tuple_out = gv1(data)
        ce_2 = relay.TupleGetItem(tuple_out, 1)
        ce_3 = relay.TupleGetItem(tuple_out, 0)

        X = relay.tanh(ce_2)
        ce_4 = gv0(ce_3, X)
        func = relay.Function([data], ce_4)
        mod["main"] = func
        mod = relay.transform.InferType()(mod)
        return mod

    mod = tvm.IRModule()
    mod["main"] = create_graph()
    mod = transform.InferType()(mod)

    ref_mod = expected()

    partitioned = transform.PartitionGraph()(mod)
    tvm.ir.assert_structural_equal(partitioned, ref_mod, map_free_vars=True)


def test_dnnl_fuse():
    dnnl_patterns = get_pattern_table("dnnl")
    for pattern in dnnl_patterns:
        if pattern[0] == "dnnl.conv2d_bias_relu":
            conv2d_bias_relu_pat = pattern
        elif pattern[0] == "dnnl.conv2d_bias_sigmoid":
            conv2d_bias_sigmoid_pat = pattern
        elif pattern[0] == "dnnl.conv2d_bias":
            conv2d_bias_pat = pattern
        elif pattern[0] == "dnnl.conv2d_relu":
            conv2d_relu_pat = pattern
        elif pattern[0] == "dnnl.conv2d_sigmoid":
            conv2d_sigmoid_pat = pattern
        elif pattern[0] == "dnnl.conv2d_bias_sum":
            conv2d_bias_sum_pat = pattern
        elif pattern[0] == "dnnl.conv2d_bias_sum_relu":
            conv2d_bias_sum_relu_pat = pattern

    def get_blocks(
        prefix,
        data,
        in_channel,
        out_channel,
        include_bias_add=True,
        include_bn=True,
        include_sigmoid=False,
    ):
        weight = relay.var(prefix + "weight")
        bias = relay.var(prefix + "bias")
        bn_gamma = relay.var(prefix + "bn_gamma")
        bn_beta = relay.var(prefix + "bn_beta")
        bn_mmean = relay.var(prefix + "bn_mean")
        bn_mvar = relay.var(prefix + "bn_var")

        layer = relay.nn.conv2d(
            data=data, weight=weight, kernel_size=(3, 3), channels=out_channel, padding=(1, 1)
        )
        if include_bias_add:
            layer = relay.nn.bias_add(layer, bias)
        if include_bn:
            bn_output = relay.nn.batch_norm(layer, bn_gamma, bn_beta, bn_mmean, bn_mvar)
            layer = bn_output[0]
        if include_sigmoid:
            # dummy layer to prevent pattern detection
            layer = relay.sigmoid(layer)
        layer = relay.nn.relu(layer)
        return layer

    def get_net(include_bias_add=True, include_bn=True, include_sigmoid=False):
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        block1 = get_blocks("block1_", data, 3, 8, include_bias_add, include_bn, include_sigmoid)
        # The second block is always conv + relu, to make it more interesting
        block2 = get_blocks("block2_", block1, 8, 8, False, False, include_sigmoid)
        return relay.Function(relay.analysis.free_vars(block2), block2)

    def get_partitoned_mod(mod, params, pattern_table):
        # This is required for constant folding
        mod["main"] = bind_params_by_name(mod["main"], params)

        remove_bn_pass = tvm.transform.Sequential(
            [
                transform.InferType(),
                transform.SimplifyInference(),
                transform.FoldConstant(),
                transform.FoldScaleAxis(),
            ]
        )
        # fold consecutive add ops to simplify pattern `conv2d-bias_add-bn-relu`
        remove_linear_pass = tvm.transform.Sequential(
            [
                transform.SimplifyExpr(),
                transform.FoldConstant(),
            ]
        )
        composite_partition = tvm.transform.Sequential(
            [
                transform.CanonicalizeOps(),
                remove_bn_pass,
                remove_linear_pass,
                transform.MergeComposite(pattern_table),
                transform.AnnotateTarget("dnnl"),
                transform.PartitionGraph(),
            ]
        )

        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            return composite_partition(mod)

    def test_detect_pattern(
        pattern_table, include_bias_add, include_bn, include_sigmoid, num_expected_partition
    ):
        net = get_net(include_bias_add, include_bn, include_sigmoid)
        mod, params = tvm.relay.testing.create_workload(net)
        mod = get_partitoned_mod(mod, params, pattern_table)
        assert len(mod.functions) - 1 == num_expected_partition  # -1 for main

    def test_sum_pattern(pattern_table, num_expected_partition):
        def get_conv2d_bn_sum_relu(
            x_shape=(1, 32, 8, 8),
            k_shape=(16, 32, 3, 3),
            sum_shape=(1, 16, 6, 6),
            dtype="float32",
        ):
            x = relay.var("x", shape=(x_shape), dtype=dtype)
            kernel = relay.const(np.random.randint(0, 1, k_shape).astype(dtype))
            bias = relay.var("bias", shape=(k_shape[0],), dtype=dtype)
            beta = relay.const(np.zeros(k_shape[0]).astype(dtype))
            gamma = relay.const(np.ones(k_shape[0]).astype(dtype))
            moving_mean = relay.const(np.zeros(k_shape[0]).astype(dtype))
            moving_var = relay.const(np.ones(k_shape[0]).astype(dtype))
            sum_data = relay.var("data1", shape=sum_shape, dtype=dtype)

            dic = {"x": x_shape, "bias": (k_shape[0],), "sum_data": sum_shape}
            param_lst = ["bias", "sum_data"]

            conv = relay.nn.conv2d(
                x,
                kernel,
                channels=k_shape[0],
                kernel_size=k_shape[2:4],
            )
            conv_bias = relay.nn.bias_add(conv, bias)
            conv_bias_bn, _, _ = relay.nn.batch_norm(
                conv_bias,
                gamma=gamma,
                beta=beta,
                moving_mean=moving_mean,
                moving_var=moving_var,
                axis=1,
                center=True,
                scale=True,
                epsilon=1e-5,
            )
            conv_bias_bn_sum = relay.add(conv_bias_bn, sum_data)
            return relay.nn.relu(conv_bias_bn_sum), dic, param_lst

        net, dic, param_lst = get_conv2d_bn_sum_relu()
        net = tvm.IRModule.from_expr(net)
        params = {x: np.random.uniform(-1, 1, dic[x]).astype("float32") for x in param_lst}
        mod = get_partitoned_mod(net, params, pattern_table)
        assert len(mod.functions) - 1 == num_expected_partition  # -1 for main

    def test_partition():
        # conv + bn + relu, conv + relu -> fused conv_bias_relu, conv, and relu
        test_detect_pattern([conv2d_bias_relu_pat], False, True, False, 3)
        # conv + bn + relu, conv + relu -> conv, bias, relu, and fused conv_relu
        test_detect_pattern([conv2d_relu_pat], False, True, False, 4)
        # conv + bn + relu, conv + relu -> fused conv_bias_relu, and fused conv_relu
        test_detect_pattern([conv2d_bias_relu_pat, conv2d_relu_pat], False, True, False, 2)
        # conv + bias_add + bn + relu, conv + relu -> fused conv_bias_relu, and fused conv_relu
        test_detect_pattern([conv2d_bias_relu_pat, conv2d_relu_pat], True, True, False, 2)
        # conv + relu, conv + relu -> two fused conv_relu
        test_detect_pattern([conv2d_relu_pat], False, False, False, 2)
        # conv + relu, conv + relu -> no fusion, 4 partition each with a single op
        test_detect_pattern([conv2d_bias_relu_pat], False, False, False, 4)
        # conv + bn + sigmoid + relu, conv + sigmoid + relu -> no fusion
        test_detect_pattern([conv2d_bias_relu_pat, conv2d_relu_pat], False, True, True, 7)
        # conv + bias_add + bn + sigmoid + relu, conv + sigmoid + relu -> fused conv_bias
        # and single op sigmoid, relu, conv, sigmoid, relu
        test_detect_pattern([conv2d_bias_pat, conv2d_relu_pat], True, True, True, 6)
        # conv + bias_add + bn + sigmoid + relu, conv + sigmoid + relu -> fused conv_bias_sigmoid
        # and single op relu, conv, sigmoid, relu
        test_detect_pattern([conv2d_bias_sigmoid_pat, conv2d_relu_pat], True, True, True, 5)
        # conv + bias_add + bn + sigmoid + relu, conv + sigmoid + relu -> fused conv_bias_sigmoid,
        # fused conv_sigmoid and single op relu, relu
        test_detect_pattern([conv2d_bias_sigmoid_pat, conv2d_sigmoid_pat], True, True, True, 4)
        # conv + bias_add + bn + add + relu -> fused conv_bias_sum, relu
        test_sum_pattern([conv2d_bias_sum_pat], 2)
        # conv + bias_add + bn + add + relu -> fused conv_bias_sum_relu,
        test_sum_pattern([conv2d_bias_sum_relu_pat], 1)

    def test_partition_mobilenet():
        mod, params = relay.testing.mobilenet.get_workload()
        mod = get_partitoned_mod(mod, params, dnnl_patterns)
        # 27 fused conv + bn + relu, one dense, one softmax and one global_avg_pooling
        assert len(mod.functions) - 1 == 30  # -1 for main

    def test_exec(mod, params, ref_mod, ref_params, out_shape):
        ishape = (1, 3, 224, 224)
        i_data = np.random.randn(*ishape).astype(np.float32)
        ref_res = relay.create_executor("graph", mod=ref_mod, device=tvm.cpu(0)).evaluate()(
            i_data, **ref_params
        )
        te_compiler.get().clear()

        mod = get_partitoned_mod(mod, params, dnnl_patterns)

        check_result(mod, {"data": i_data}, out_shape, ref_res.numpy(), tol=1e-5, params=params)

    test_partition()
    test_partition_mobilenet()

    if not tvm.get_global_func("relay.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    net = get_net()
    mod, params = tvm.relay.testing.create_workload(net)
    ref_mod, ref_params = tvm.relay.testing.create_workload(net)
    test_exec(mod, params, ref_mod, ref_params, (1, 8, 224, 224))

    mod, params = relay.testing.mobilenet.get_workload()
    ref_mod, ref_params = relay.testing.mobilenet.get_workload()
    test_exec(mod, params, ref_mod, ref_params, (1, 1000))


def test_multiple_use_of_an_output():
    def expected_same_output_region():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(8, 8))
        y = relay.var("y", shape=(8, 8))
        z = relay.var("z", shape=(8, 8))
        x0 = relay.var("x0", shape=(8, 8))
        y0 = relay.var("y0", shape=(8, 8))
        log = relay.log(x0)
        sub = x0 - y0
        mul = log * sub
        # The partitioned graph contains log, subtract, and multiply
        func = relay.Function([x0, y0], mul)
        func = set_func_attr(func, "ccompiler", "tvmgen_default_ccompiler_main_0")
        glb_0 = relay.GlobalVar("tvmgen_default_ccompiler_main_0")
        mod[glb_0] = func
        mod = transform.InferType()(mod)

        add = x + y
        call = relay.Call(glb_0, [add, z])
        main = relay.Function([x, y, z], call)
        mod["main"] = main
        mod = transform.InferType()(mod)
        return mod

    def expected_different_output_region():
        mod = tvm.IRModule()
        x = relay.var("x", shape=(8, 8))
        y = relay.var("y", shape=(8, 8))
        z = relay.var("z", shape=(8, 8))

        # The partitioned graph contains log
        i0 = relay.var("i0", shape=(8, 8))
        log = relay.log(i0)
        func = relay.Function([i0], log)
        func = set_func_attr(func, "ccompiler", "tvmgen_default_ccompiler_main_0")
        glb_0 = relay.GlobalVar("tvmgen_default_ccompiler_main_0")
        mod[glb_0] = func
        mod = transform.InferType()(mod)

        # The partitioned graph contains subtract
        x0 = relay.var("x0", shape=(8, 8))
        y0 = relay.var("y0", shape=(8, 8))
        sub = x0 - y0
        func = relay.Function([x0, y0], sub)
        func = set_func_attr(func, "ccompiler", "tvmgen_default_ccompiler_main_1")
        glb_1 = relay.GlobalVar("tvmgen_default_ccompiler_main_1")
        mod[glb_1] = func
        mod = transform.InferType()(mod)

        add = x + y
        call_log = relay.Call(glb_0, [add])
        call_sub = relay.Call(glb_1, [add, z])
        main = relay.Function([x, y, z], call_log * call_sub)
        mod["main"] = main
        mod = transform.InferType()(mod)
        return mod

    def get_mod():
        x = relay.var("x", shape=(8, 8))
        y = relay.var("y", shape=(8, 8))
        z = relay.var("z", shape=(8, 8))
        add = x + y
        sub = add - z
        log = relay.log(add)
        sub1 = log * sub
        f = relay.Function([x, y, z], sub1)
        mod = tvm.IRModule()
        mod["main"] = f
        return mod

    def test_same_output_region():
        mod = get_mod()
        mod = AllowedListAnnotator(["subtract", "log", "multiply"], "ccompiler")(mod)
        mod = transform.MergeCompilerRegions()(mod)
        mod = transform.PartitionGraph()(mod)

        expected_mod = expected_same_output_region()
        tvm.ir.assert_structural_equal(mod, expected_mod, map_free_vars=True)

    def test_different_output_region():
        mod = get_mod()
        mod = AllowedListAnnotator(["subtract", "log"], "ccompiler")(mod)
        mod = transform.MergeCompilerRegions()(mod)
        mod = transform.PartitionGraph()(mod)

        expected_mod = expected_different_output_region()
        tvm.ir.assert_structural_equal(mod, expected_mod, map_free_vars=True)

    test_same_output_region()
    test_different_output_region()


def test_duplicate_outputs():
    target = "test_duplicate_outputs"

    @tvm.ir.register_op_attr("abs", "target." + target)
    def abs(expr):  # pylint: disable=unused-variable
        return True

    def create_graph():
        data = relay.var("data", shape=(10, 10))
        x = relay.abs(data)
        out_1 = relay.nn.relu(x)
        out_2 = relay.tanh(x)
        out_3 = relay.log(x)
        out = relay.Tuple([out_1, out_2, out_3])
        func = relay.Function([data], out)
        return func

    def expected():
        mod = tvm.IRModule()

        # function 0
        f0_i0 = relay.var(target + "_0_i0", shape=(10, 10))
        f0_o0 = relay.abs(f0_i0)
        func0 = relay.Function([f0_i0], f0_o0)

        func0 = func0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func0 = func0.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        func0 = func0.with_attr("Compiler", target)
        func0 = func0.with_attr("global_symbol", "tvmgen_default_" + target + "_main_0")
        gv0 = relay.GlobalVar("tvmgen_default_" + target + "_main_0")
        mod[gv0] = func0
        mod = transform.InferType()(mod)

        # body
        data = relay.var("data", shape=(10, 10))
        function_out = gv0(data)
        out_1 = relay.nn.relu(function_out)
        out_2 = relay.tanh(function_out)
        out_3 = relay.log(function_out)
        out = relay.Tuple([out_1, out_2, out_3])
        func = relay.Function([data], out)
        mod["main"] = func
        mod = transform.InferType()(mod)
        return mod

    mod = tvm.IRModule()
    mod["main"] = create_graph()

    seq = tvm.transform.Sequential(
        [
            transform.AnnotateTarget(target),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    ref_mod = expected()
    partitioned = seq(mod)
    tvm.ir.assert_structural_equal(partitioned, ref_mod, map_free_vars=True)


def test_duplicate_merge_and_tuplegetitem():
    target = "test_duplicate_merge_and_tuplegetitem"

    @tvm.ir.register_op_attr("nn.batch_norm", "target." + target)
    def batch_norm(expr):  # pylint: disable=unused-variable
        return True

    @tvm.ir.register_op_attr("nn.relu", "target." + target)
    def relu(expr):  # pylint: disable=unused-variable
        return True

    def create_graph():
        data = relay.var("data", shape=(10, 10))
        bn_gamma = relay.var("bn_gamma")
        bn_beta = relay.var("bn_beta")
        bn_mmean = relay.var("bn_mean")
        bn_mvar = relay.var("bn_var")
        x = relay.nn.batch_norm(data, bn_gamma, bn_beta, bn_mmean, bn_mvar)
        out_1 = relay.nn.relu(x[0])
        bn_out_1 = x[1]
        out_2 = relay.tanh(bn_out_1)
        out_3 = relay.log(bn_out_1)
        out = relay.Tuple([out_1, out_2, out_3])
        func = relay.Function([data, bn_gamma, bn_beta, bn_mmean, bn_mvar], out)
        return func

    def expected():
        mod = tvm.IRModule()

        # function 0
        f0_i0 = relay.var(target + "_0_i0", shape=(10, 10))
        f0_i1 = relay.var(target + "_0_i1")
        f0_i2 = relay.var(target + "_0_i2")
        f0_i3 = relay.var(target + "_0_i3")
        f0_i4 = relay.var(target + "_0_i4")
        f0_n0 = relay.nn.batch_norm(f0_i0, f0_i1, f0_i2, f0_i3, f0_i4)
        f0_n1 = f0_n0[1]
        f0_n2 = relay.nn.relu(f0_n0[0])
        f0_o0 = relay.Tuple([f0_n2, f0_n1])
        func0 = relay.Function([f0_i0, f0_i1, f0_i2, f0_i3, f0_i4], f0_o0)

        func0 = func0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func0 = func0.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        func0 = func0.with_attr("Compiler", target)
        func0 = func0.with_attr("global_symbol", "tvmgen_default_" + target + "_main_0")
        gv0 = relay.GlobalVar("tvmgen_default_" + target + "_main_0")
        mod[gv0] = func0
        mod = transform.InferType()(mod)

        # body
        data = relay.var("data", shape=(10, 10))
        bn_gamma = relay.var("bn_gamma")
        bn_beta = relay.var("bn_beta")
        bn_mmean = relay.var("bn_mean")
        bn_mvar = relay.var("bn_var")
        function_out = gv0(data, bn_gamma, bn_beta, bn_mmean, bn_mvar)
        get_out0 = relay.TupleGetItem(function_out, 0)
        get_out1 = relay.TupleGetItem(function_out, 1)
        out_2 = relay.tanh(get_out1)
        out_3 = relay.log(get_out1)
        out = relay.Tuple([get_out0, out_2, out_3])
        func = relay.Function([data, bn_gamma, bn_beta, bn_mmean, bn_mvar], out)
        mod["main"] = func
        mod = transform.InferType()(mod)
        return mod

    mod = tvm.IRModule()
    mod["main"] = create_graph()
    mod = transform.InferType()(mod)

    seq = tvm.transform.Sequential(
        [
            transform.AnnotateTarget(target),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    ref_mod = expected()
    partitioned = seq(mod)
    tvm.ir.assert_structural_equal(partitioned, ref_mod, map_free_vars=True)


def test_constant_tuples():
    @tvm.ir.register_op_attr("qnn.concatenate", "target.const_tuples")
    def add(expr):  # pylint: disable=unused-variable
        return True

    def create_graph():
        a = relay.var("a", shape=(10, 10), dtype="uint8")
        b = relay.var("b", shape=(10, 10), dtype="uint8")
        a1 = relay.abs(a)

        zeroi = relay.const(1, "int32")
        zerof = relay.const(0, "float32")
        con = relay.qnn.op.concatenate(
            (a1, b),
            input_scales=(zerof, zerof),
            input_zero_points=(zeroi, zeroi),
            output_scale=zerof,
            output_zero_point=zeroi,
            axis=1,
        )

        f = relay.Function([a, b], con)
        mod = tvm.IRModule.from_expr(f)
        mod = transform.InferType()(mod)
        return mod

    seq = tvm.transform.Sequential(
        [
            transform.AnnotateTarget("const_tuples"),
            transform.InferType(),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    partitioned = seq(create_graph())

    concat = partitioned["tvmgen_default_const_tuples_main_0"].body
    assert type(concat.args[1]) == relay.Tuple
    assert type(concat.args[2]) == relay.Tuple
    assert type(concat.args[3]) == relay.Constant
    assert type(concat.args[4]) == relay.Constant


def test_flatten_tuple_output():
    target = "test_flatten_tuple_output"

    @tvm.ir.register_op_attr("split", "target." + target)
    def split(expr):  # pylint: disable=unused-variable
        return True

    @tvm.ir.register_op_attr("abs", "target." + target)
    def abs(expr):  # pylint: disable=unused-variable
        return True

    def create_graph():
        a = relay.var("a", shape=(10, 10), dtype="uint8")

        a_split = relay.split(a, 2)
        a_split_0 = relay.TupleGetItem(a_split.astuple(), 0)
        a_split_0_abs = relay.abs(a_split_0)

        a_con = relay.concatenate(a_split, 0)
        a_split_0_relu = relay.nn.relu(a_split_0_abs)

        out = relay.Tuple((a_con, a_split_0_relu))
        f = relay.Function([a], out)
        mod = tvm.IRModule.from_expr(f)
        mod = transform.InferType()(mod)
        return mod

    def expected():
        mod = tvm.IRModule()

        # function 0
        f0_i0 = relay.var(target + "_0_i0", shape=(10, 10), dtype="uint8")
        a_split = relay.split(f0_i0, 2)
        a_split_0 = relay.TupleGetItem(a_split.astuple(), 0)
        a_split_1 = relay.TupleGetItem(a_split.astuple(), 1)
        a_split_abs_in = relay.TupleGetItem(a_split.astuple(), 0)
        abs = relay.abs(a_split_abs_in)
        tuple_out = relay.Tuple((a_split_0, a_split_1, abs))
        func0 = relay.Function([f0_i0], tuple_out)

        func0 = func0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func0 = func0.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        func0 = func0.with_attr("Compiler", target)
        func0 = func0.with_attr("global_symbol", "tvmgen_default_" + target + "_main_0")
        gv0 = relay.GlobalVar("tvmgen_default_" + target + "_main_0")
        mod[gv0] = func0
        mod = transform.InferType()(mod)

        # body
        data = relay.var("a", shape=(10, 10), dtype="uint8")
        f_out = gv0(data)
        f_out_0 = relay.TupleGetItem(f_out, 0)
        f_out_1 = relay.TupleGetItem(f_out, 1)
        tuple = relay.Tuple((f_out_0, f_out_1))
        concat = relay.concatenate(tuple, 0)
        f_out_2 = relay.TupleGetItem(f_out, 2)
        relu = relay.nn.relu(f_out_2)
        ret_tuple = relay.Tuple((concat, relu))
        mod["main"] = relay.Function([data], ret_tuple)
        mod = transform.InferType()(mod)
        return mod

    seq = tvm.transform.Sequential(
        [
            transform.AnnotateTarget(target),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    partitioned = seq(create_graph())
    partitioned = transform.InferType()(partitioned)
    expected_mod = transform.InferType()(expected())
    tvm.ir.assert_structural_equal(partitioned, expected_mod, map_free_vars=True)


def test_tuple_output_exec():
    """Test C codegen and runtime for a subgraph with a tuple output"""
    a = relay.var("a", shape=(10, 10), dtype="float32")
    b = relay.var("b", shape=(10, 10), dtype="float32")
    ba = relay.annotation.compiler_begin(a, "ccompiler")
    bb = relay.annotation.compiler_begin(b, "ccompiler")
    add = relay.add(ba, bb)
    sub = relay.subtract(ba, bb)
    out = relay.Tuple((add, sub))
    eout = relay.annotation.compiler_end(out, "ccompiler")
    func = relay.Function([a, b], eout)

    mod = tvm.IRModule()
    mod["main"] = func
    mod = transform.InferType()(mod)
    mod = transform.PartitionGraph()(mod)

    a_data = np.random.rand(10, 10).astype("float32")
    b_data = np.random.rand(10, 10).astype("float32")

    check_result(
        mod,
        {"a": a_data, "b": b_data},
        [(10, 10), (10, 10)],
        [(a_data + b_data), (a_data - b_data)],
    )


def test_extern_opt():
    def Optimize(mod):
        return relay.transform.FoldConstant()(mod)

    tvm.register_func("relay.ext.test_target.optimize", Optimize)

    x = relay.var("x", shape=(2, 2))
    y0 = relay.var("y0", shape=(2, 2))
    y1 = relay.var("y1", shape=(2, 2))
    yy0 = relay.annotation.compiler_begin(y0, "test_target")
    yy1 = relay.annotation.compiler_begin(y1, "test_target")
    z = yy0 + yy1
    end = relay.annotation.compiler_end(z, "test_target")
    f = relay.Function([x, y0, y1], end * x)
    c = np.ones(shape=(2, 2), dtype="float32")
    f = bind_params_by_name(f, {"y0": tvm.nd.array(c), "y1": tvm.nd.array(c)})
    mod = tvm.IRModule()
    mod["main"] = f
    mod = transform.InferType()(mod)
    mod = transform.PartitionGraph()(mod)

    try:
        t0 = mod["tvmgen_default_test_target_main_0"]
    except:
        raise KeyError("test_target_main_0 not found")

    assert isinstance(t0.body, relay.Constant)
    expected = np.empty([2, 2])
    expected.fill(2)
    tvm.testing.assert_allclose(t0.body.data.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_preserve_type_import():
    """Test to make sure type definition and imports are preserved during the BYOC pipeline."""
    from tvm.relay.prelude import Prelude, StaticTensorArrayOps

    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        tensor_array = p.get_global_var_static("tensor_array", dtype, shape)
        tensor = p.get_tensor_ctor_static("tensor_constructor", dtype, shape)
        write = p.get_global_var_static("tensor_array_write", dtype, shape)
        gather = p.get_global_var_static("tensor_array_gather", dtype, shape)
        v = relay.var("v")
        indice = relay.var("indice")
        init_tensor_array = tensor_array(relay.const(3))
        tensor_array1 = write(init_tensor_array, relay.const(0), tensor(v))
        tensor_array2 = write(tensor_array1, relay.const(1), tensor(v))
        tensor_array3 = write(tensor_array2, relay.const(2), tensor(v))
        out = gather(tensor_array3, indice)
        mod["main"] = relay.Function([v, indice], out)
        mod = transform.RemoveUnusedFunctions()(mod)
        mod = transform.PartitionGraph()(mod)

    run("float32", [2, 3])


def test_not_bind_constant():
    def get_net(prefix, data, out_channel):
        weight = relay.var(prefix + "weight")
        bn_gamma = relay.var(prefix + "bn_gamma")
        bn_beta = relay.var(prefix + "bn_beta")
        bn_mmean = relay.var(prefix + "bn_mean")
        bn_mvar = relay.var(prefix + "bn_var")

        layer = relay.nn.conv2d(
            data=data, weight=weight, kernel_size=(3, 3), channels=out_channel, padding=(1, 1)
        )
        bn_output = relay.nn.batch_norm(layer, bn_gamma, bn_beta, bn_mmean, bn_mvar)
        out = relay.nn.relu(bn_output[0])
        return relay.Function(relay.analysis.free_vars(out), out)

    def get_partitoned_mod(mod, params, pattern_table, bind_constants):
        mod["main"] = bind_params_by_name(mod["main"], params)
        remove_bn_pass = tvm.transform.Sequential(
            [
                transform.InferType(),
                transform.SimplifyInference(),
                transform.FoldConstant(),
                transform.FoldScaleAxis(),
            ]
        )
        composite_partition = tvm.transform.Sequential(
            [
                remove_bn_pass,
                transform.MergeComposite(pattern_table),
                transform.AnnotateTarget("dnnl"),
                transform.PartitionGraph(bind_constants=bind_constants),
            ]
        )

        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            return composite_partition(mod)

    data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
    net = get_net("block_", data, 8)
    mod, params = tvm.relay.testing.create_workload(net)

    mod = get_partitoned_mod(mod, params, get_pattern_table("dnnl"), bind_constants=True)
    len(mod["main"].body.args) == 1

    mod = get_partitoned_mod(mod, params, get_pattern_table("dnnl"), bind_constants=False)
    len(mod["main"].body.args) == 3


if __name__ == "__main__":
    test_multi_node_compiler()
    test_extern_ccompiler_single_op()
    test_extern_ccompiler_default_ops()
    test_extern_ccompiler_multiple_functions()
    test_extern_ccompiler()
    test_extern_dnnl()
    test_extern_dnnl_mobilenet()
    test_function_lifting()
    test_function_lifting_inline()
    test_constant_propagation()
    test_multiple_outputs()
    test_mixed_single_multiple_outputs()
    test_dnnl_fuse()
    test_multiple_use_of_an_output()
    test_duplicate_outputs()
    test_duplicate_merge_and_tuplegetitem()
    test_constant_tuples()
    test_flatten_tuple_output()
    test_tuple_output_exec()
    test_extern_opt()
    test_not_bind_constant()
