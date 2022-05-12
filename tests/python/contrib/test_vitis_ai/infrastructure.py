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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, W0611, C0413

"""Expose Vitis-AI test functions to the Python frontend"""

import sys
import numpy as np

import pytest

pytest.importorskip("pyxir")
import pyxir.contrib.target.DPUCZDX8G

import tvm
from tvm import relay
from tvm import runtime
from tvm.relay import transform
from tvm.relay.op.contrib.vitis_ai import partition_for_vitis_ai
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib.target import vitis_ai
from tvm.contrib import graph_executor
from tvm.contrib import utils


def get_cpu_op_count(mod):
    """Traverse graph counting ops offloaded to TVM."""

    class Counter(tvm.relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.count = 0

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op):
                self.count += 1

            super().visit_call(call)

    c = Counter()
    c.visit(mod["main"])
    return c.count


def build_module(
    mod,
    target,
    dpu_target="DPUCADF8H",
    params=None,
    enable_vitis_ai=True,
    tvm_ops=0,
    vitis_ai_partitions=1,
):
    """Build module for Vitis-AI codegen."""
    if isinstance(mod, tvm.relay.expr.Call):
        mod = tvm.IRModule.from_expr(mod)
    if params is None:
        params = {}

    with tvm.transform.PassContext(
        opt_level=3, config={"relay.ext.vitis_ai.options.target": dpu_target}
    ):
        if enable_vitis_ai:
            mod = partition_for_vitis_ai(mod, params, dpu_target)
            tvm_op_count = get_cpu_op_count(mod)
            assert tvm_op_count == tvm_ops, "Got {} TVM operators, expected {}".format(
                tvm_op_count, tvm_ops
            )
            partition_count = 0
            for global_var in mod.get_global_vars():
                if "vitis_ai" in global_var.name_hint:
                    partition_count += 1

            assert (
                vitis_ai_partitions == partition_count
            ), "Got {} Vitis-AI partitions, expected {}".format(
                partition_count, vitis_ai_partitions
            )
        relay.backend.te_compiler.get().clear()
        return relay.build(mod, target, params=params)


def update_lib(lib, cross_compile=None):
    tmp_path = utils.tempdir()
    lib_name = "lib.so"
    lib_path = tmp_path.relpath(lib_name)
    if cross_compile:
        lib.export_library(lib_path, cc=cross_compile)
    else:
        lib.export_library(lib_path)
    lib = runtime.load_module(lib_path)
    return lib


def extract_vitis_ai_modules(module):
    """Get the Vits-AI runtime module from llvm module."""
    return list(
        filter(lambda mod: mod.type_key == "VitisAIRuntime", module.get_lib().imported_modules)
    )


def verify_codegen(
    module, num_vitis_ai_modules=1, params=None, target="llvm", tvm_ops=0, dpu_target="DPUCADX8G"
):
    """Check Vitis-AI codegen against a known good output."""
    module = build_module(
        module,
        target,
        params=params,
        dpu_target=dpu_target,
        tvm_ops=tvm_ops,
        vitis_ai_partitions=num_vitis_ai_modules,
    )
    vitis_ai_modules = extract_vitis_ai_modules(module)

    assert len(vitis_ai_modules) == num_vitis_ai_modules, (
        f"The number of Vitis-AI modules produced ({len(vitis_ai_modules)}) does not "
        f"match the expected value ({num_vitis_ai_modules})."
    )


def verify_result(
    mod,
    map_inputs,
    out_shape,
    result,
    tol=1e-5,
    target="llvm",
    device=tvm.cpu(),
    params=None,
    dpu_target="DPUCADX8G",
    tvm_ops=0,
):
    """To check the result between reference and byoc vitis-ai flow"""

    lib = build_module(mod, target, params=params, dpu_target=dpu_target, tvm_ops=tvm_ops)
    lib = update_lib(lib)
    rt_mod = graph_executor.GraphModule(lib["default"](tvm.cpu()))

    for name, data in map_inputs.items():
        rt_mod.set_input(name, data)
    rt_mod.set_input(**params)
    rt_mod.run()

    out_shapes = out_shape if isinstance(out_shape, list) else [out_shape]
    results = result if isinstance(result, list) else [result]

    for idx, shape in enumerate(out_shapes):
        out = tvm.nd.empty(shape, device=device)
        out = rt_mod.get_output(idx, out)
        tvm.testing.assert_allclose(out.numpy(), results[idx], rtol=tol, atol=tol)
