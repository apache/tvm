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
import tvm.testing
from tvm import relay
from tvm import rpc
from tvm.contrib import graph_executor
from tvm.relay.op.contrib import csinn
from tvm.contrib import utils
from tvm.autotvm.measure import request_remote


class Device:
    """
    Configuration for CSINN tests.

    Check tests/python/contrib/test_csinn/ for the presence of an test_config.json file.
    This file can be used to override the default configuration here which will attempt to run the riscv
    Compute Library runtime tests locally if the runtime is available. Changing the configuration
    will allow these runtime tests to be offloaded to a remote riscv device via a tracker for example.

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
        Specify path to cross compiler to use when connecting a remote device from a non-riscv platform.
    """

    connection_type = "local"
    host = "127.0.0.1"
    port = 9090
    target = "llvm"
    device_key = ""
    cross_compile = ""

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

    @classmethod
    def load(cls, file_name):
        """Load test config

        Load the test configuration by looking for file_name relative
        to the test_csinn directory.
        """
        location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        config_file = os.path.join(location, file_name)
        if not os.path.exists(config_file):
            warnings.warn("Config file doesn't exist, resuming CSINN tests with default config.")
            return
        with open(config_file, mode="r") as config:
            test_config = json.load(config)

        cls.connection_type = test_config["connection_type"]
        cls.host = test_config["host"]
        cls.port = test_config["port"]
        cls.target = test_config["target"]
        cls.device_key = test_config.get("device_key") or ""
        cls.cross_compile = test_config.get("cross_compile") or ""


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


def skip_runtime_test():
    """Skip test if it requires the runtime and it's not present."""
    # CSINN codegen not present.
    if not tvm.get_global_func("relay.ext.csinn", True):
        print("Skip because CSINN codegen is not available.")
        return True

    # Remote device is in use or CSINN runtime not present
    # Note: Ensure that the device config has been loaded before this check
    if not Device.connection_type != "local" and not csinn.is_csinn_runtime_enabled():
        print("Skip because runtime isn't present or a remote device isn't being used.")
        return True


def skip_codegen_test():
    """Skip test if it requires the CSINN codegen and it's not present."""
    if not tvm.get_global_func("relay.ext.csinn", True):
        print("Skip because CSINN codegen is not available.")
        return True


def build_module(mod, target, params=None, enable_csinn=True, tvm_ops=0, csinn_partitions=1):
    """Build module with option to build for CSINN."""
    if isinstance(mod, tvm.relay.expr.Call):
        mod = tvm.IRModule.from_expr(mod)
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        if enable_csinn:
            mod = csinn.partition_for_csinn(mod, params)
            tvm_op_count = get_cpu_op_count(mod)
            assert tvm_op_count == tvm_ops, "Got {} TVM operators, expected {}".format(
                tvm_op_count, tvm_ops
            )
            partition_count = 0
            for global_var in mod.get_global_vars():
                if "csinn" in global_var.name_hint:
                    partition_count += 1

            assert (
                csinn_partitions == partition_count
            ), "Got {} CSINN partitions, expected {}".format(partition_count, csinn_partitions)
        relay.backend.te_compiler.get().clear()
        return relay.build(mod, target=target, params=params)


def build_and_run(
    mod,
    inputs,
    outputs,
    params,
    device,
    enable_csinn=True,
    no_runs=1,
    tvm_ops=0,
    csinn_partitions=1,
    config=None,
):
    """Build and run the relay module."""
    if config is None:
        config = {}

    try:
        lib = build_module(mod, device.target, params, enable_csinn, tvm_ops, csinn_partitions)
    except Exception as e:
        err_msg = "The module could not be built.\n"
        if config:
            err_msg += f"The test failed with the following parameters: {config}\n"
        err_msg += str(e)
        raise Exception(err_msg)

    lib = update_lib(lib, device.device, device.cross_compile)
    gen_module = graph_executor.GraphModule(lib["default"](device.device.cpu(0)))
    gen_module.set_input(**inputs)
    out = []
    for _ in range(no_runs):
        gen_module.run()
        out.append([gen_module.get_output(i) for i in range(outputs)])
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


def verify(answers, atol, rtol, verify_saturation=False, config=None):
    """Compare the array of answers. Each entry is a list of outputs."""
    if config is None:
        config = {}

    if len(answers) < 2:
        raise RuntimeError(f"No results to compare: expected at least two, found {len(answers)}")
    for answer in zip_longest(*answers):
        for outs in combinations(answer, 2):
            try:
                if verify_saturation:
                    assert (
                        np.count_nonzero(outs[0].numpy() == 255) < 0.25 * outs[0].numpy().size
                    ), "Output is saturated: {}".format(outs[0])
                    assert (
                        np.count_nonzero(outs[0].numpy() == 0) < 0.25 * outs[0].numpy().size
                    ), "Output is saturated: {}".format(outs[0])
                tvm.testing.assert_allclose(outs[0].numpy(), outs[1].numpy(), rtol=rtol, atol=atol)
            except AssertionError as e:
                err_msg = "Results not within the acceptable tolerance.\n"
                if config:
                    err_msg += f"The test failed with the following parameters: {config}\n"
                err_msg += str(e)
                raise AssertionError(err_msg)


def extract_csinn_modules(module):
    """Get the CSINN module(s) from llvm module."""
    return list(filter(lambda mod: mod.type_key == "csinn", module.get_lib().imported_modules))
