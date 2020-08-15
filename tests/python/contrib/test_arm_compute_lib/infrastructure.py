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

import numpy as np

import tvm
from tvm import relay
from tvm import rpc
from tvm.contrib import graph_runtime
from tvm.relay.op.contrib import arm_compute_lib
from tvm.contrib import util


class Device:
    """Adjust the following settings to connect to and use a remote device for tests."""
    use_remote = False
    target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
    # Enable cross compilation when connecting a remote device from a non-arm platform.
    cross_compile = None
    # cross_compile = "aarch64-linux-gnu-g++"

    def __init__(self):
        """Keep remote device for lifetime of object."""
        self.device = self._get_remote()

    @classmethod
    def _get_remote(cls):
        """Get a remote (or local) device to use for testing."""
        if cls.use_remote:
            # Here you may adjust settings to run the ACL unit tests via a remote
            # device using the RPC mechanism. Use this in the case you want to compile
            # an ACL module on a different machine to what you run the module on i.e.
            # x86 -> AArch64.
            #
            # Use the following to connect directly to a remote device:
            # device = rpc.connect(
            #     hostname="0.0.0.0",
            #     port=9090)
            #
            # Or connect via a tracker:
            # device = tvm.autotvm.measure.request_remote(
            #     host="0.0.0.0",
            #     port=9090,
            #     device_key="device_key",
            #     timeout=1000)
            #
            # return device
            raise NotImplementedError(
                "Please adjust these settings to connect to your remote device.")
        else:
            device = rpc.LocalSession()
            return device


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
    # ACL codegen not present.
    if not tvm.get_global_func("relay.ext.arm_compute_lib", True):
        print("Skip because Arm Compute Library codegen is not available.")
        return True

    # Remote device is in use or ACL runtime not present
    if not Device.use_remote and not arm_compute_lib.is_arm_compute_runtime_enabled():
        print("Skip because runtime isn't present or a remote device isn't being used.")
        return True


def skip_codegen_test():
    """Skip test if it requires the ACL codegen and it's not present."""
    if not tvm.get_global_func("relay.ext.arm_compute_lib", True):
        print("Skip because Arm Compute Library codegen is not available.")
        return True


def build_module(mod, target, params=None, enable_acl=True, tvm_ops=0, acl_partitions=1):
    """Build module with option to build for ACL."""
    if isinstance(mod, tvm.relay.expr.Call):
        mod = tvm.IRModule.from_expr(mod)
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        if enable_acl:
            mod = arm_compute_lib.partition_for_arm_compute_lib(mod, params)
            tvm_op_count = get_cpu_op_count(mod)
            assert tvm_op_count == tvm_ops, \
                "Got {} TVM operators, expected {}".format(tvm_op_count, tvm_ops)
            partition_count = 0
            for global_var in mod.get_global_vars():
                if "arm_compute_lib" in global_var.name_hint:
                    partition_count += 1

            assert acl_partitions == partition_count, \
                "Got {} Arm Compute Library partitions, expected {}".format(
                    partition_count, acl_partitions)
        relay.backend.compile_engine.get().clear()
        return relay.build(mod, target=target, params=params)


def build_and_run(mod, inputs, outputs, params, device, enable_acl=True, no_runs=1,
                  tvm_ops=0, acl_partitions=1):
    """Build and run the relay module."""
    lib = build_module(mod, device.target, params, enable_acl, tvm_ops, acl_partitions)
    lib = update_lib(lib, device.device, device.cross_compile)
    gen_module = graph_runtime.GraphModule(lib['default'](device.device.cpu(0)))
    gen_module.set_input(**inputs)
    out = []
    for _ in range(no_runs):
        gen_module.run()
        out.append([gen_module.get_output(i) for i in range(outputs)])
    return out


def update_lib(lib, device, cross_compile):
    """Export the library to the remote/local device."""
    lib_name = "mod.so"
    temp = util.tempdir()
    lib_path = temp.relpath(lib_name)
    if cross_compile:
        lib.export_library(lib_path, cc=cross_compile)
    else:
        lib.export_library(lib_path)
    device.upload(lib_path)
    lib = device.load_module(lib_name)
    return lib


def verify(answers, atol, rtol, verify_saturation=False, params=None):
    """Compare the array of answers. Each entry is a list of outputs."""
    if params is None:
        params = {}

    if len(answers) < 2:
        raise RuntimeError(
            f"No results to compare: expected at least two, found {len(answers)}")
    for answer in zip_longest(*answers):
        for outs in combinations(answer, 2):
            if verify_saturation:
                assert np.count_nonzero(outs[0].asnumpy() == 255) < 0.25 * outs[0].asnumpy().size, \
                    "Output is saturated: {}".format(outs[0])
                assert np.count_nonzero(outs[0].asnumpy() == 0) < 0.25 * outs[0].asnumpy().size, \
                    "Output is saturated: {}".format(outs[0])
            try:
                tvm.testing.assert_allclose(
                   outs[0].asnumpy(), outs[1].asnumpy(), rtol=rtol, atol=atol)
            except AssertionError as e:
                err_msg = "Results not within the acceptable tolerance.\n"
                if params:
                    err_msg += f"The test failed with the following parameters: {params}\n"
                err_msg += str(e)
                raise AssertionError(err_msg)


def extract_acl_modules(module):
    """Get the ACL module(s) from llvm module."""
    return list(filter(lambda mod: mod.type_key == "arm_compute_lib",
                       module.get_lib().imported_modules))


def verify_codegen(module, known_good_codegen, num_acl_modules,
                   target="llvm -mtriple=aarch64-linux-gnu -mattr=+neon"):
    """Check acl codegen against a known good output."""
    module = build_module(module, target)
    acl_modules = extract_acl_modules(module)

    assert len(acl_modules) == num_acl_modules, \
        f"The number of Arm Compute Library modules produced ({len(acl_modules)}) does not " \
        f"match the expected value ({num_acl_modules})."

    for mod in acl_modules:
        source = mod.get_source("json")
        codegen = json.loads(source)["nodes"]
        # remove input and const names as these cannot be predetermined
        for node in range(len(codegen)):
            if codegen[node]["op"] == "input" or codegen[node]["op"] == "const":
                codegen[node]["name"] = ""
        codegen_str = json.dumps(codegen, sort_keys=True, indent=2)
        known_good_codegen_str = json.dumps(known_good_codegen, sort_keys=True, indent=2)

        assert codegen_str == known_good_codegen_str, \
            f"The JSON produced by codegen does not match the expected result. \n" \
            f"Actual={codegen_str} \n" \
            f"Expected={known_good_codegen_str}"


def generate_trials(space, r_factor=3):
    """Generates a series of trials.

    This algorithm generates a series of non-deterministic trials given a
    space of options to test. A trial is generated by pulling a value from
    each option in the space. On some occasions the values are shuffled to
    ensure a different trial on each r_factor iteration. The algorithm ensures
    that each value from an option is used at least once. The total number of
    trials is determined by the r_factor * the option with the largest number
    of values.

    Parameters
    ----------
    space: List[List[Any]]
        A list of different options with varying values to test.
    r_factor: (optional) int
        The repeat factor.

    Returns
    -------
    A list of trials specifying values for each option.

    """
    np.random.seed(0)
    max_len = 1
    for option in space:
        max_len = max(max_len, len(option))

    num_trials = r_factor * max_len
    trials = []
    for i in range(num_trials):
        trial = []
        for option in space:
            if i % len(option) == 0:
                np.random.shuffle(option)
            trial.append(option[i % len(option)])

        trials.append(trial)

    return trials
