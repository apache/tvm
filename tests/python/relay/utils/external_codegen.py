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
"""Utilities for testing external code generation"""

import os
import sys

import pytest

import tvm
from tvm import relay, runtime, testing
from tvm.contrib import utils


skip_windows = pytest.mark.skipif(sys.platform == "win32", reason="Skip test on Windows for now")
skip_micro = pytest.mark.skipif(
    tvm.support.libinfo().get("USE_MICRO", "OFF") != "ON",
    reason="MicroTVM support not enabled. Set USE_MICRO=ON in config.cmake to enable.",
)


def parametrize_external_codegen_checks(test):
    """Parametrize over the various check_result functions which are available"""
    return pytest.mark.parametrize(
        "check_result",
        [
            pytest.param(check_aot_executor_result, marks=[skip_windows, skip_micro]),
            pytest.param(check_graph_executor_result, marks=[skip_windows]),
            pytest.param(check_vm_result, marks=[skip_windows]),
        ],
    )(test)


def parametrize_external_json_codegen_checks(test):
    """Parametrize over the various check_result functions which are available for JSON"""
    return pytest.mark.parametrize(
        "check_result",
        [
            pytest.param(check_graph_executor_result, marks=[skip_windows]),
            pytest.param(check_vm_result, marks=[skip_windows]),
        ],
    )(test)


def update_lib(lib):
    test_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.join(test_dir, "..", "..", "..", "..")
    contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")

    kwargs = {}
    kwargs["options"] = ["-O2", "-std=c++17", "-I" + contrib_path]
    tmp_path = utils.tempdir()
    lib_name = "lib.so"
    lib_path = tmp_path.relpath(lib_name)
    lib.export_library(lib_path, fcompile=False, **kwargs)
    lib = tvm.runtime.load_module(lib_path)

    return lib


def check_vm_result(mod, map_inputs, out_shape, result, tol=1e-5, target="llvm", device=tvm.cpu()):
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        exe = relay.vm.compile(mod, target=target)
    code, lib = exe.save()
    lib = update_lib(lib)
    exe = runtime.vm.Executable.load_exec(code, lib)
    vm = runtime.vm.VirtualMachine(exe, device)
    out = vm.run(**map_inputs)
    tvm.testing.assert_allclose(out.numpy(), result, rtol=tol, atol=tol)


def check_graph_executor_result(
    mod, map_inputs, out_shape, result, tol=1e-5, target="llvm", device=tvm.cpu()
):
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        executor_factory = relay.build(mod, target=target)
    lib = update_lib(executor_factory.lib)
    rt_mod = tvm.contrib.graph_executor.create(executor_factory.graph_json, lib, device)

    for name, data in map_inputs.items():
        rt_mod.set_input(name, data)
    rt_mod.run()
    out = tvm.nd.empty(out_shape, device=device)
    out = rt_mod.get_output(0, out)

    tvm.testing.assert_allclose(out.numpy(), result, rtol=tol, atol=tol)


def check_aot_executor_result(
    mod, map_inputs, out_shape, result, tol=1e-5, target="llvm", device=tvm.cpu()
):
    # Late import to avoid breaking test with USE_MICRO=OFF.
    from tvm.testing.aot import AOTTestModel, compile_and_run
    from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER

    interface_api = "packed"
    use_unpacked_api = False
    test_runner = AOT_DEFAULT_RUNNER
    compile_and_run(
        AOTTestModel(module=mod, inputs=map_inputs, outputs={"output": result}),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


def set_external_func_attr(func, compiler, ext_symbol):
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", compiler)
    func = func.with_attr("global_symbol", ext_symbol)
    return func
