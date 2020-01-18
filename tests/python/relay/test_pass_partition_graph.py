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
import tvm.relay.transform as transform
from tvm import relay
from tvm.contrib import util
from tvm.relay.annotation import compiler_begin, compiler_end
from tvm.relay.expr_functor import ExprMutator
from tvm.relay import analysis, expr as _expr
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


class ConvBiasAddReLUAnnotator(ExprMutator):
    import enum
    state = enum.Enum("State", "Init Conv Bias ReLU")

    def __init__(self, backend):
        super().__init__()
        self.current_state = self.state.Init
        self.backend = backend

    def annotate_call(self, call):
        new_args = []
        for arg in call.args:
            new_arg = super().visit(arg)
            if call.op.name == "nn.conv2d" or isinstance(new_arg, relay.expr.Var):
                new_arg = compiler_begin(new_arg, self.backend)
            new_args.append(new_arg)
        return relay.Call(call.op, new_args, call.attrs, call.type_args)

    def visit_call(self, call):
        if call.op.name == "nn.conv2d":
            if self.current_state == self.state.Bias:
                self.current_state = self.state.Conv
                ret = self.annotate_call(call)
                self.current_state = self.state.Conv
                return ret
            self.current_state = self.state.Init
        elif call.op.name == "add":
            if self.current_state == self.state.ReLU:
                self.current_state = self.state.Bias
                return self.annotate_call(call)
            self.current_state = self.state.Init
        elif call.op.name == "nn.relu":
            self.current_state = self.state.ReLU
            op = self.annotate_call(call)
            if self.current_state == self.state.Conv:
                op = compiler_end(op, self.backend)
            self.current_state = self.state.Init
            return op
        self.current_state = self.state.Init
        return super().visit_call(call)


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
        lib = tvm.module.load(lib_path)

        return lib

    def check_vm_result():
        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
            exe = relay.vm.compile(mod, target=target, params=params)
        code, lib = exe.save()
        lib = update_lib(lib)
        exe = relay.vm.Executable.load_exec(code, lib)
        vm = relay.vm.VirtualMachine(exe)
        vm.init(ctx)
        out = vm.run(**map_inputs)
        tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)

    def check_graph_runtime_result():
        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
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
    mod = relay.Module()
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
    mod = relay.Module()
    mod["main"] = f
    mod = MyAnnotator()(mod)
    mod = transform.PartitionGraph()(mod)

    check_result(mod, {"x": x_data, "y": y_data}, (8, 8), x_data + y_data)


def test_extern_ccompiler_default_ops():
    def expected():
        x = relay.var("x", shape=(8, 8))
        y = relay.var("y", shape=(8, 8))
        x0 = relay.var("x0", shape=(8, 8))
        y0 = relay.var("y0", shape=(8, 8))
        add = x0 + y0
        # Function that uses C compiler
        func = relay.Function([x0, y0], add)
        func = func.set_attribute("Primitive", tvm.expr.IntImm("int32", 1))
        func = func.set_attribute("Compiler",
                                  tvm.expr.StringImm("ccompiler"))
        func = func.set_attribute("ExternalSymbol",
                                  tvm.expr.StringImm("ccompiler_0"))
        add_call = relay.Call(func, [x, y])
        # Function that uses default compiler. Ops are fused in this function.
        p0 = relay.var("p0", shape=(8, 8))
        log = relay.log(p0)
        exp = relay.exp(p0)
        concat = relay.concatenate([log, exp], axis=0)
        fused_func = relay.Function([p0], concat)
        fused_func = fused_func.set_attribute("Primitive",
                                              tvm.expr.IntImm("int32", 1))
        fused_call = relay.Call(fused_func, [add_call])
        main = relay.Function([x, y], fused_call)
        mod = relay.Module()
        mod["main"] = main
        return mod

    x = relay.var("x", shape=(8, 8))
    y = relay.var("y", shape=(8, 8))
    add = x + y
    log = relay.log(add)
    exp = relay.exp(add)
    concat = relay.concatenate([log, exp], axis=0)
    f = relay.Function([x, y], concat)
    mod = relay.Module()
    mod["main"] = f
    mod = WhiteListAnnotator(["add", "subtract", "multiply"], "ccompiler")(mod)
    mod = transform.PartitionGraph()(mod)

    fused_mod = transform.FuseOps(2)(mod)
    expected_mod = expected()
    assert relay.alpha_equal(fused_mod, expected_mod)

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
    mod = relay.Module()
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
    data = relay.var('data', shape=(ishape), dtype=dtype)
    weight1 = relay.var('weight1', shape=(w1shape), dtype=dtype)
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

    f = relay.Function([data, weight1], out)

    mod = relay.Module()
    mod['main'] = WholeGraphAnnotator('dnnl').visit(f)
    mod = transform.PartitionGraph()(mod)

    ref_mod = relay.Module()
    ref_mod['main'] = f

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w1_data = np.random.uniform(0, 1, w1shape).astype(dtype)

    ref_ex = relay.create_executor("graph", mod=ref_mod, ctx=tvm.cpu())
    ref_res = ref_ex.evaluate()(i_data, w1_data)
    check_result(mod, {"data": i_data, "weight1": w1_data},
                 (1, 32, 14, 14), ref_res.asnumpy(), tol=1e-5)


def test_extern_dnnl_mobilenet():
    if not tvm.get_global_func("relay.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = 'float32'
    ishape = (1, 3, 224, 224)
    mod, params = relay.testing.mobilenet.get_workload(
        batch_size=1, dtype='float32')

    op_list = ["nn.conv2d", "nn.dense", "nn.relu", "add"]
    mod = WhiteListAnnotator(op_list, "dnnl")(mod)
    mod = transform.PartitionGraph()(mod)
    i_data = np.random.uniform(0, 1, ishape).astype(dtype)

    ref_mod, params = relay.testing.mobilenet.get_workload(batch_size=1,
                                                           dtype='float32')
    ref_ex = relay.create_executor("graph", mod=ref_mod, ctx=tvm.cpu(0))
    ref_res = ref_ex.evaluate()(i_data, **params)

    check_result(mod, {"data": i_data},
                 (1, 1000), ref_res.asnumpy(), tol=1e-5, params=params)


def test_partition_conv_bias_relu():
    if not tvm.get_global_func("relay.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    def get_layers(prefix, data, in_channel, out_channel,
                   include_bn=True, include_sigmoid=False):
        weight = relay.const(np.random.randn(out_channel, in_channel, 3, 3))
        bn_gamma = relay.const(np.random.randn(out_channel))
        bn_beta = relay.const(np.random.randn(out_channel))
        bn_mmean = relay.const(np.random.randn(out_channel))
        bn_mvar = relay.const(np.random.randn(out_channel))

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
        layer1 = get_layers("layer1_", data, 3, 16, include_bn, include_sigmoid)
        layer2 = get_layers("layer2_", layer1, 16, 16, include_bn, include_sigmoid)
        last = layer2
        return relay.Function(relay.analysis.free_vars(last), last)

    def pre_optimize(mod, params):
        remove_bn_pass = transform.Sequential([
            relay.transform.InferType(),
            relay.transform.SimplifyInference(),
            relay.transform.FoldConstant(),
            relay.transform.FoldScaleAxis(),
        ])

        if params != {}:
            # This is required for constant folding on mobilenet
            mod["main"] = bind_params_by_name(mod["main"], params)

        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
            mod = remove_bn_pass(mod)

        return mod

    def get_partitoned_mod(mod):
        mod["main"] = ConvBiasAddReLUAnnotator("dnnl").visit(mod["main"])
        mod = transform.PartitionGraph()(mod)
        return mod

    def get_partitions(mod):
        partitions = []

        def visit_func(expr):
            if isinstance(expr, _expr.Function) and expr != mod["main"]:
                partitions.append(expr)
        analysis.post_order_visit(mod["main"], visit_func)
        return partitions

    def test_detect_pattern(include_bn, include_sigmoid, num_expected_partition):
        net = get_net(include_bn, include_sigmoid)
        mod, params = tvm.relay.testing.create_workload(net)
        mod = pre_optimize(mod, params)
        mod = get_partitoned_mod(mod)
        assert(len(get_partitions(mod)) == num_expected_partition)

    def test_partition():
        # conv + bn + relu -> detection succeed
        test_detect_pattern(True, False, 2)
        # conv + relu -> fail
        test_detect_pattern(False, False, 0)
        # conv + bn + sigmoid + relu -> fail
        test_detect_pattern(True, True, 0)

    def test_partition_mobilenet():
        mod, params = relay.testing.mobilenet.get_workload()
        mod = pre_optimize(mod, params)
        mod = get_partitoned_mod(mod)
        assert(len(get_partitions(mod)) == 27)

    test_partition()
    # test_partition_mobilenet()

    # TODO: Enable executor check once the runtime signature issue is resolved
    net = get_net()
    mod, params = tvm.relay.testing.create_workload(net)
    mod = pre_optimize(mod, params)
    mod = get_partitoned_mod(mod)

    ref_mod, params = tvm.relay.testing.create_workload(net)
    ishape = (1, 3, 224, 224)
    i_data = np.random.randn(*ishape).astype(np.float32)
    ref_ex = relay.create_executor("graph", mod=ref_mod, ctx=tvm.cpu(0))
    ref_res = ref_ex.evaluate()(i_data, **params)

    check_result(mod, {"data": i_data},
                 ishape, ref_res.asnumpy(), tol=1e-5, params=params)


if __name__ == "__main__":
    # test_multi_node_compiler()
    # test_extern_ccompiler_single_op()
    # test_extern_ccompiler_default_ops()
    # test_extern_ccompiler()
    # test_extern_dnnl()
    # test_extern_dnnl_mobilenet()
    test_partition_conv_bias_relu()
