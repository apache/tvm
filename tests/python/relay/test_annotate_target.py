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
        kwargs["options"] = ["-O2", "-std=c++11", "-I" + contrib_path]
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
        begin2 = relay.annotation.compiler_begin(end0, "dnnl")
        begin3 = relay.annotation.compiler_begin(end0, "dnnl")
        begin4 = relay.annotation.compiler_begin(weight1, "dnnl")
        depthwise_conv2d_2 = relay.nn.conv2d(begin3,
                                             begin4,
                                             kernel_size=(3, 3),
                                             padding=(1, 1),
                                             groups=32)
        end1 = relay.annotation.compiler_end(depthwise_conv2d_2, "dnnl")
        begin5 = relay.annotation.compiler_begin(end1, "dnnl")
        out = relay.add(begin2, begin5)
        end2 = relay.annotation.compiler_end(out, "dnnl")
        f = relay.Function([data, weight1], end2)
        mod = tvm.IRModule.from_expr(f)
        return mod

    dtype = "float32"
    ishape = (1, 32, 14, 14)
    w1shape = (32, 1, 3, 3)

    def test_annotate():
        mod = annotated(dtype, ishape, w1shape)
        mod = transform.AnnotateTarget("dnnl")(mod)
        ref_mod = expected(dtype, ishape, w1shape)
        assert relay.analysis.alpha_equal(mod, ref_mod)

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


def test_extern_dnnl_mobilenet():
    if not tvm.get_global_func("relay.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = 'float32'
    ishape = (1, 3, 224, 224)
    mod, params = relay.testing.mobilenet.get_workload(
        batch_size=1, dtype='float32')

    mod = transform.AnnotateTarget("dnnl")(mod)
    mod = transform.PartitionGraph()(mod)
    i_data = np.random.uniform(0, 1, ishape).astype(dtype)

    ref_mod, params = relay.testing.mobilenet.get_workload(batch_size=1,
                                                           dtype='float32')
    ref_ex = relay.create_executor("graph", mod=ref_mod, ctx=tvm.cpu(0))
    ref_res = ref_ex.evaluate()(i_data, **params)

    check_result(mod, {"data": i_data},
                 (1, 1000), ref_res.asnumpy(), tol=1e-5, params=params)


if __name__ == "__main__":
    test_extern_dnnl()
    test_extern_dnnl_mobilenet()
