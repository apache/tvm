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
import tvm.relay.op as reg
import tvm.relay.transform as transform
from tvm import relay
from tvm import runtime
from tvm.contrib import util


def check_result(mod, map_inputs, out_shape, result, tol=1e-5, target="llvm",
                 ctx=tvm.cpu(), params=None):
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        return

    def update_lib(lib):
        test_dir = os.path.dirname(
            os.path.realpath(os.path.expanduser(__file__)))
        source_dir = os.path.join(test_dir, "..", "..", "..")
        contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")

        kwargs = {}
        kwargs["options"] = ["-O2", "-std=c++14", "-I" + contrib_path]
        tmp_path = util.tempdir()
        lib_name = 'lib.so'
        lib_path = tmp_path.relpath(lib_name)
        lib.export_library(lib_path, fcompile=False, **kwargs)
        lib = runtime.load_module(lib_path)

        return lib

    def check_vm_result():
        with relay.build_config(opt_level=3, disabled_pass=["AlterOpLayout"]):
            exe = relay.vm.compile(mod, target=target, params=params)
        code, lib = exe.save()
        lib = update_lib(lib)
        exe = runtime.vm.Executable.load_exec(code, lib)
        vm = runtime.vm.VirtualMachine(exe)
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


def test_extern_dnnl():
    def annotated(dtype, ishape, w1shape):
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

        mod = tvm.IRModule.from_expr(f)
        return mod

    def expected(dtype, ishape, w1shape):
        data = relay.var('data', shape=(ishape), dtype=dtype)
        weight1 = relay.var('weight1', shape=(w1shape), dtype=dtype)
        begin0 = relay.annotation.compiler_begin(data, "dnnl")
        begin1 = relay.annotation.compiler_begin(weight1, "dnnl")
        depthwise_conv2d_1 = relay.nn.conv2d(begin0,
                                             begin1,
                                             kernel_size=(3, 3),
                                             padding=(1, 1),
                                             groups=32)
        end0 = relay.annotation.compiler_end(depthwise_conv2d_1, "dnnl")
        end1 = relay.annotation.compiler_end(depthwise_conv2d_1, "dnnl")
        begin2 = relay.annotation.compiler_begin(end1, "dnnl")
        begin3 = relay.annotation.compiler_begin(end0, "dnnl")
        begin4 = relay.annotation.compiler_begin(weight1, "dnnl")
        depthwise_conv2d_2 = relay.nn.conv2d(begin3,
                                             begin4,
                                             kernel_size=(3, 3),
                                             padding=(1, 1),
                                             groups=32)
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
        ref_mod = expected(dtype, ishape, w1shape)
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

        ref_ex = relay.create_executor("graph", mod=ref_mod, ctx=tvm.cpu())
        ref_res = ref_ex.evaluate()(i_data, w1_data)

        check_result(mod, {"data": i_data, "weight1": w1_data},
                     (1, 32, 14, 14), ref_res.asnumpy(), tol=1e-5)

    test_annotate()
    test_run()

@pytest.mark.skip(reason="fix constant node before opening this case")
def test_extern_dnnl_mobilenet():
    if not tvm.get_global_func("relay.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = 'float32'
    ishape = (1, 3, 224, 224)
    mod, params = relay.testing.mobilenet.get_workload(
        batch_size=1, dtype='float32')

    mod["main"] = relay.build_module.bind_params_by_name(mod["main"], params)
    mod = transform.AnnotateTarget("dnnl")(mod)
    mod = transform.PartitionGraph()(mod)
    i_data = np.random.uniform(0, 1, ishape).astype(dtype)

    ref_mod, params = relay.testing.mobilenet.get_workload(batch_size=1,
                                                           dtype='float32')
    ref_ex = relay.create_executor("graph", mod=ref_mod, ctx=tvm.cpu(0))
    ref_res = ref_ex.evaluate()(i_data, **params)

    check_result(mod, {"data": i_data},
                 (1, 1000), ref_res.asnumpy(), tol=1e-5, params=params)


def test_multiple_ends():
    @reg.register("nn.relu", "target.test")
    def relu(attrs, args):  # pylint: disable=unused-variable
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

    result = transform.AnnotateTarget("test")(before())
    expected = transform.InferType()(after())
    assert tvm.ir.structural_equal(expected, result)


def test_type_propagation():
    target = "test_type_propagation"

    @reg.register("nn.relu", "target." + target)
    def relu(attrs, args): # pylint: disable=unused-variable
        return args[0].checked_type.dtype == "float32"

    def before():
        x = relay.var("x", shape=(10, 10))
        r = relay.nn.relu(x)
        out = relay.nn.relu(r)
        f = relay.Function([x], out)
        mod = tvm.IRModule.from_expr(f)
        return mod

    # If the type isn't propogated, then the relu checker function will fail to get the dtype.
    assert transform.AnnotateTarget(target)(before())


def test_tuple():
    target = "test_tuple"

    @reg.register("nn.relu", "target." + target)
    def relu(attrs, args): # pylint: disable=unused-variable
        return True

    @reg.register("concatenate", "target." + target)
    def concatenate(attrs, args):  # pylint: disable=unused-variable
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

    def after():
        x = relay.var("x", shape=(10, 5))
        y = relay.var("y", shape=(10, 5))
        cb_1 = relay.annotation.compiler_begin(x, target)
        cb_2 = relay.annotation.compiler_begin(y, target)
        a_1 = relay.nn.relu(cb_1)
        a_2 = relay.nn.relu(cb_2)
        ce_1 = relay.annotation.compiler_end(a_1, target)
        ce_2 = relay.annotation.compiler_end(a_2, target)
        cb_3 = relay.annotation.compiler_begin(ce_1, target)
        cb_4 = relay.annotation.compiler_begin(ce_2, target)
        tup = relay.Tuple([cb_3, cb_4])
        ce_3 = relay.annotation.compiler_end(tup, target)
        cb_3 = relay.annotation.compiler_begin(ce_3, target)
        out = relay.op._make.concatenate(cb_3, 1)
        ce_4 = relay.annotation.compiler_end(out, target)
        f = relay.Function([x, y], ce_4)
        mod = tvm.IRModule.from_expr(f)
        return mod

    result = transform.AnnotateTarget(target)(before())
    expected = transform.InferType()(after())
    assert tvm.ir.structural_equal(expected, result)


def test_composite_function():
    def before():
        a = relay.var('a', shape=(10, 10))
        b = relay.var('b', shape=(10, 10))

        # add_relu function
        in_1 = relay.var('in_1', shape=(10, 10))
        in_2 = relay.var('in_2', shape=(10, 10))
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
        a = relay.var('a', shape=(10, 10))
        b = relay.var('b', shape=(10, 10))

        # add_relu function
        in_1 = relay.var('in_1', shape=(10, 10))
        in_2 = relay.var('in_2', shape=(10, 10))
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


def test_multiple_runs():
    @reg.register("nn.relu", "target.A")
    def relu(attrs, args):  # pylint: disable=unused-variable
        return True

    @reg.register("add", "target.B")
    def add(attrs, args):  # pylint: disable=unused-variable
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

    mod = transform.AnnotateTarget("A")(before())
    mod = transform.AnnotateTarget("B")(mod)
    expected = transform.AnnotateTarget(["A", "B"])(before())
    assert tvm.ir.structural_equal(expected, mod)


if __name__ == "__main__":
    test_extern_dnnl()
    test_composite_function()
    #test_extern_dnnl_mobilenet()
    test_multiple_ends()
    test_type_propagation()
    test_tuple()
    test_multiple_runs()
