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


class Device:
    """
    Configuration for CLML tests.

    Check tests/python/contrib/clml/ for the presence of an test_config.json file.
    This file can be used to override the default configuration here which will attempt to run the Arm
    Compute Library runtime tests locally if the runtime is available. Changing the configuration
    will allow these runtime tests to be offloaded to a remote Arm device via a tracker for example.

    Notes
    -----
        The test configuration will be loaded once when the the class is created. If the configuration
        changes between tests, any changes will not be picked up.

    Parameters
    ----------
    device : RPCSession
        Allows tests to connect to and use remote device.

    Attributes
    ----------
    connection_type : str
        Details the type of RPC connection to use. Options:
        local - Use the local device,
        tracker - Connect to a tracker to request a remote device,
        remote - Connect to a remote device directly.
    host : str
        Specify IP address or hostname of remote target.
    port : int
        Specify port number of remote target.
    target : str
        The compilation target.
    device_key : str
        The device key of the remote target. Use when connecting to a remote device via a tracker.
    cross_compile : str
        Specify path to cross compiler to use when connecting a remote device from a non-arm platform.
    """

    connection_type = "tracker"
    host = os.getenv("TVM_TRACKER_HOST", "localhost")
    port = int(os.getenv("TVM_TRACKER_PORT", 9090))
    target = "opencl"
    target_host = "llvm -mtriple=aarch64-linux-gnu"
    device_key = "android"
    cross_compile = os.getenv("TVM_NDK_CC", "aarch64-linux-android-g++")

    def __init__(self):
        """Keep remote device for lifetime of object."""
        self.device = self._get_remote()

    @classmethod
    def _get_remote(cls):
        """Get a remote (or local) device to use for testing."""
        if cls.connection_type == "tracker":
            device = request_remote(cls.device_key, cls.host, cls.port, timeout=1000)
        elif cls.connection_type == "remote":
            device = rpc.connect(cls.host, cls.port)
        elif cls.connection_type == "local":
            device = rpc.LocalSession()
        else:
            raise ValueError(
                "connection_type in test_config.json should be one of: " "local, tracker, remote."
            )

        return device


def skip_codegen_test():
    """Skip test if it requires the CLML codegen and it's not present."""
    if not tvm.get_global_func("relay.ext.clml", True):
        print("Skip because CLML codegen is not available.")
        return True


def build_module(mod, target, target_host, params=None, enable_clml=True, tune_log=""):
    """Build module with option to build for CLML."""
    if isinstance(mod, tvm.relay.expr.Call):
        mod = tvm.IRModule.from_expr(mod)

    with autotvm.apply_history_best(tune_log):
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            if enable_clml:
                mod = clml.partition_for_clml(mod, params)
            relay.backend.te_compiler.get().clear()
            return relay.build(mod, target=target, target_host=target_host, params=params)


def build_and_run(
    mod, inputs, outputs, params, device, enable_clml=True, no_runs=1, config=None, tune_log=""
):
    """Build and run the relay module."""
    if config is None:
        config = {}

    try:
        libm = build_module(mod, device.target, device.target_host, params, enable_clml, tune_log)

        clml_modules = extract_clml_modules(libm)
        for mod in clml_modules:
            source = mod.get_source("json")
            codegen = json.loads(source)["nodes"]
            # remove input and const names as these cannot be predetermined
            for node in range(len(codegen)):
                if codegen[node]["op"] == "input" or codegen[node]["op"] == "const":
                    codegen[node]["name"] = ""
            codegen_str = json.dumps(codegen, sort_keys=True, indent=2)

    except Exception as e:
        err_msg = "The module could not be built.\n"
        if config:
            err_msg += f"The test failed with the following parameters: {config}\n"
        err_msg += str(e)
        raise Exception(err_msg)

    lib = update_lib(libm, device.device, device.cross_compile)
    gen_module = graph_executor.GraphModule(lib["default"](device.device.cl(0)))
    gen_module.set_input(**inputs)
    out = []
    for _ in range(no_runs):
        gen_module.run()
        out.append([gen_module.get_output(i) for i in range(outputs)])
    time_f = gen_module.module.time_evaluator("run", device.device.cl(0), number=1)
    cost = time_f().mean
    print("%g secs/iteration\n" % cost)
    return out


def update_lib(lib, device, cross_compile):
    """Export the library to the remote/local device."""
    lib_name = "mod.so"
    temp = utils.tempdir()
    lib_path = temp.relpath(lib_name)
    if cross_compile:
        lib.export_library(lib_path, cc=cross_compile)
    else:
        lib.export_library(lib_path)
    device.upload(lib_path)
    lib = device.load_module(lib_name)
    return lib


def extract_clml_modules(module):
    """Get the CLML module(s) from llvm module."""
    return list(filter(lambda mod: mod.type_key == "clml", module.get_lib().imported_modules))


def verify_codegen(
    module,
    known_good_codegen,
    num_clml_modules=1,
    tvm_ops=0,
    target="llvm -mtriple=aarch64-linux-gnu",
):
    """Check clml codegen against a known good output."""
    module = build_module(module, target, tvm_ops=tvm_ops, clml_partitions=num_clml_modules)
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
