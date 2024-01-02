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
from itertools import zip_longest, combinations
import json
import os
import warnings

import numpy as np

import tvm
from tvm import relay
from tvm import rpc

# from tvm.contrib.debugger import debug_runtime as graph_executor
from tvm.contrib import graph_executor
from tvm.relay.op.contrib import clml
from tvm.contrib import utils
from tvm import autotvm
from tvm.autotvm.measure import request_remote
from tvm.relay.expr_functor import ExprMutator, Call

"""Utils for adreno compute/schedules"""

import os
import tvm
import numpy as np
from tvm import relay
from tvm import autotvm
from tvm import rpc
from tvm.contrib import utils, ndk
from tvm.relay import testing
from tvm.relay.transform import recast
from tvm.contrib import graph_runtime
from tvm.runtime.vm import VirtualMachine
import json

from tvm.relay.collage.collage import *

RPC_TRACKER_HOST = os.getenv("TVM_TRACKER_HOST", "localhost")
RPC_TRACKER_PORT = int(os.getenv("TVM_TRACKER_PORT", 9090))
RPC_KEY = os.getenv("RPC_DEVICE_KEY", "android")
NDK_CROSS_COMPILER = os.getenv("TVM_NDK_CC", "aarch64-linux-android-g++")


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


def get_non_cpu_op_count(mod):
    """Traverse graph counting ops not offloaded to TVM."""

    class Counter(tvm.relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.count = 0

        def visit_call(self, call):
            if not isinstance(call.op, tvm.ir.Op):
                self.count += 1

            super().visit_call(call)

    c = Counter()
    c.visit(mod["main"])
    return c.count


# build module run with opencl or clml target with graph executor
def build_and_run(
    remote,
    mod,
    params1,
    inputs,
    target="llvm",
    enable_clml=False,
    stat_file=None,
):
    if remote is None:
        target_host = "llvm"
    else:
        target_host = "llvm -mtriple=arm64-linux-android"

    if isinstance(mod, tvm.relay.expr.Call):
        mod = tvm.IRModule.from_expr(mod)

    with autotvm.apply_history_best(stat_file):
        with tvm.transform.PassContext(opt_level=3):
            if enable_clml:
                mod = clml.partition_for_clml(mod, params1)
            graph, lib, params = relay.build(
                mod, target_host=target_host, target=target, params=params1
            )

    if remote is None:
        ctx = tvm.opencl()
        m = graph_runtime.create(graph, lib, ctx)
    else:
        temp = utils.tempdir()
        dso_binary = "dev_lib_cl.so"
        dso_binary_path = temp.relpath(dso_binary)
        ctx = remote.cl(0)
        lib.export_library(dso_binary_path, fcompile=ndk.create_shared)
        remote.upload(dso_binary_path)
        rlib = remote.load_module(dso_binary)
        m = graph_runtime.create(graph, rlib, ctx)
    m.set_input(**params)
    m.set_input(**inputs)
    m.run()
    return m.get_output(0)


# build module run with opencl or clml target with vm executor
def build_and_run_vm(
    remote,
    mod,
    params1,
    inputs,
    target="llvm",
    enable_clml=False,
    stat_file=None,
):
    if remote is None:
        target_host = "llvm"
    else:
        target_host = "llvm -mtriple=arm64-linux-android"

    target_host = tvm.target.Target(target_host)
    target = tvm.target.Target(target, target_host)
    if isinstance(mod, relay.Function):
        module = tvm.IRModule({})
        module["main"] = mod
        mod = module
    elif isinstance(mod, tvm.relay.expr.Call):
        mod = tvm.IRModule.from_expr(mod)

    with autotvm.apply_history_best(stat_file):
        with tvm.transform.PassContext(opt_level=3):
            if enable_clml:
                mod = clml.partition_for_clml(mod, params1)
            vmc = relay.vm.compile(mod, target=target, params=params1)

    if remote is None:
        dev = tvm.opencl()
        vm = VirtualMachine(vmc, dev, "naive")
    else:
        temp = utils.tempdir()
        dso_binary = "dev_lib_cl.so"
        dso_binary_path = temp.relpath(dso_binary)
        dev = remote.cl(0)
        vmc.mod.export_library(dso_binary_path, cc=NDK_CROSS_COMPILER)
        remote.upload(dso_binary_path)
        rlib = remote.load_module(dso_binary)
        vm = VirtualMachine(rlib, dev, "naive")
    inputs_data = {}
    for key in inputs.keys():
        inputs_data[key] = tvm.nd.array(inputs[key], dev)
    for k, v in params1.items():
        inputs_data[k] = tvm.nd.array(v, dev)
    vm.set_input("main", **inputs_data)
    vm.invoke_stateful("main")
    out = vm.get_outputs()[0]

    return out


def extract_clml_modules(module):
    """Get the CLML module(s) from llvm module."""
    return list(filter(lambda mod: mod.type_key == "clml", module.get_lib().imported_modules))


def verify_codegen(
    remote,
    mod,
    params,
    known_good_codegen,
    target="llvm",
    num_clml_modules=1,
    tvm_ops=0,
):
    if remote is None:
        target_host = "llvm"
    else:
        target_host = "llvm -mtriple=arm64-linux-android"

    """Check clml codegen against a known good output."""
    if isinstance(mod, tvm.relay.expr.Call):
        mod = tvm.IRModule.from_expr(mod)
    with tvm.transform.PassContext(opt_level=3):
        mod = clml.partition_for_clml(mod, params)
        tvm_op_count = get_cpu_op_count(mod)
        assert tvm_op_count == tvm_ops, "Got {} TVM operators, expected {}".format(
            tvm_op_count, tvm_ops
        )
        partition_count = 0
        for global_var in mod.get_global_vars():
            if "clml" in global_var.name_hint:
                partition_count += 1

        assert (
            num_clml_modules == partition_count
        ), "Got {} Open CLML partitions, expected {}".format(partition_count, num_clml_modules)
    relay.backend.te_compiler.get().clear()

    module = relay.build(mod, target=target, target_host=target_host, params=params)
    clml_modules = extract_clml_modules(module)
    assert len(clml_modules) == num_clml_modules, (
        f"The number of CLML modules produced ({len(clml_modules)}) does not "
        f"match the expected value ({num_clml_modules})."
    )

    for mod in clml_modules:
        source = mod.get_source("json")
        codegen = json.loads(source)["nodes"]
        # remove input and const names as these cannot be predetermined
        for node in range(len(codegen)):
            if codegen[node]["op"] == "input" or codegen[node]["op"] == "const":
                codegen[node]["name"] = ""
        codegen_str = json.dumps(codegen, sort_keys=True, indent=2)
        known_good_codegen_str = json.dumps(known_good_codegen, sort_keys=True, indent=2)

        assert codegen_str == known_good_codegen_str, (
            f"The JSON produced by codegen does not match the expected result. \n"
            f"Actual={codegen_str} \n"
            f"Expected={known_good_codegen_str}"
        )


########### Collage Drivers ###########


def compile_and_run(label, model, targets, inputs):
    """Compile model for target and run it with profiling."""
    logging.info(f"Compiling {model['name']} using {label} with {targets}...")
    mod = model["mod"]
    exe = tvm.relay.vm.compile(mod, target=targets, params=model["params"])
    lib = exe.mod
    temp = utils.tempdir()
    dso_binary = "dev_lib_cl.so"
    dso_binary_path = temp.relpath(dso_binary)
    logging.info(f"Exporting library to {dso_binary_path}...")
    lib.export_library(dso_binary_path, cc=NDK_CROSS_COMPILER)
    tracker = rpc.connect_tracker(RPC_TRACKER_HOST, RPC_TRACKER_PORT)
    remote = tracker.request(RPC_KEY, priority=0, session_timeout=600)
    ctx = remote.cl(0)
    remote.upload(dso_binary_path)
    rlib = remote.load_module(dso_binary)
    vm_factory = tvm.runtime.vm.VirtualMachine(rlib, ctx, "naive")
    inputs_data = {}
    for key in inputs.keys():
        inputs_data[key] = tvm.nd.array(inputs[key], ctx)
    for k, v in model["params"].items():
        inputs_data[k] = tvm.nd.array(v, ctx)
    vm_factory.set_input("main", **inputs_data)
    vm_factory.invoke_stateful("main")
    out = vm_factory.get_outputs()[0]
    return out.asnumpy()


# Custom cost function for Opencl RPC targets.
@register_func("tvm.relay.collage.opencl_cost_estimator")
def opencl_cost_estimator(mod, target):
    try:
        # Build the module.
        logging.info("Compiling module to estimate")
        exe = tvm.relay.vm.compile(mod, target)
    except RuntimeError as err:
        # A build failure indicates the partition is not supported.
        # eg trying to build an nn.batch_norm on GPU, which has no schedule since we assume it
        # is only ever used with a tuple projection which is rewritten away.
        logging.info("Assigning module infinite cost since unable to build: %s", err)
        return math.inf

    lib = exe.mod
    tracker = rpc.connect_tracker(RPC_TRACKER_HOST, RPC_TRACKER_PORT)
    remote = tracker.request(RPC_KEY, priority=0, session_timeout=600)
    temp = utils.tempdir()
    dso_binary = "dev_lib_cl.so"
    dso_binary_path = temp.relpath(dso_binary)
    ctx = remote.cl(0)
    lib.export_library(dso_binary_path, cc=NDK_CROSS_COMPILER)
    remote_path = dso_binary
    remote.upload(dso_binary_path, target=remote_path)
    lib = remote.load_module(remote_path)

    vm_factory = tvm.runtime.vm.VirtualMachine(lib, ctx)
    func_name = "main"
    main_args = {v.name_hint: arg_for(v.checked_type, ctx) for v in mod[func_name].params}
    cost = vm_factory.benchmark(
        ctx, repeat=5, number=20, min_repeat_ms=0, func_name=func_name, **main_args
    )
    return cost.mean
