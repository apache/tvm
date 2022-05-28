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
from tvm.contrib import graph_executor
from tvm.relay.op.contrib.bnns import partition_for_bnns
from tvm.contrib import utils
from tvm.autotvm.measure import request_remote
from tvm.relay.analysis import analysis


class Device:
    """
    Common device configuration for python tests.

    Check tests/python/contrib/arm_compute_lib/ for the presence of an test_config.json file.
    This file can be used to override the default configuration here which will attempt to run the BNNS
    runtime tests locally if the runtime is available. Changing the configuration will allow these
    runtime tests to be offloaded to a remote device with BNNS via a tracker for example.

    Notes
    -----
        The test configuration will be loaded once when the class is created. If the configuration
        changes between tests, any changes will not be picked up.


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
        to the test_bnns directory.
        """
        location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        config_file = os.path.join(location, file_name)
        if not os.path.exists(config_file):
            warnings.warn("Config file doesn't exist, resuming tests with default config.")
            return
        with open(config_file, mode="r") as config:
            test_config = json.load(config)

        cls.connection_type = test_config["connection_type"]
        cls.host = test_config["host"]
        cls.port = test_config["port"]
        cls.target = test_config["target"]
        cls.device_key = test_config.get("device_key") or ""
        cls.cross_compile = test_config.get("cross_compile") or ""


Device.target = "llvm"


def skip_runtime_test():
    """Skip test if it requires the runtime and it's not present."""
    # BNNS codegen not present.
    if not tvm.get_global_func("relay.ext.bnns", True):
        print("Skip because BNNS codegen is not available.")
        return True
    return False


def skip_codegen_test():
    """Skip test if it requires the BNNS codegen and it's not present."""
    if not tvm.get_global_func("relay.ext.bnns", True):
        print("Skip because BNNS codegen is not available.")
        return True


def build_module(mod, target, params=None, enable_bnns=True, tvm_ops=0):
    """Build module with option to build for BNNS."""
    if isinstance(mod, tvm.relay.expr.Call):
        mod = tvm.IRModule.from_expr(mod)
    with tvm.transform.PassContext(opt_level=3):
        if enable_bnns:
            mod = partition_for_bnns(mod)
        relay.backend.te_compiler.get().clear()
        return relay.build(mod, target=target, params=params)


def build_and_run(
    mod,
    inputs,
    outputs,
    params,
    device,
    enable_bnns=True,
    no_runs=1,
    tvm_ops=0,
    config=None,
):
    """Build and run the relay module."""
    if config is None:
        config = {}

    try:
        lib = build_module(mod, device.target, params, enable_bnns, tvm_ops)
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


def extract_bnns_modules(module):
    """Get the BNNS module(s) from llvm module."""
    return list(filter(lambda mod: mod.type_key == "bnns_json", module.get_lib().imported_modules))


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


def verify_codegen(
    module,
    known_good_codegen,
    num_bnns_modules,
    tvm_ops=0,
    target=Device.target,
):
    """Check BNNS codegen against a known good output."""
    module = build_module(module, target, tvm_ops=tvm_ops)
    bnns_modules = extract_bnns_modules(module)

    assert len(bnns_modules) == num_bnns_modules, (
        f"The number of BNNS modules produced ({len(bnns_modules)}) does not "
        f"match the expected value ({num_bnns_modules})."
    )

    for mod in bnns_modules:
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


def compare_inference_with_ref(func, params, atol=0.002, rtol=0.007):
    """Compare scoring results for compilation with and without BNNS.

    Provided function will be compiled two times with and without BNNS.
    The scoring results for both type of compilation will be compared
    with provided atol and rtol. The input data will be automatically
    generated based of shape and dtype info provided for var nodes.

    """
    # Generate input tensor values
    inputs = {}
    for free_param in analysis.free_vars(func):
        name = free_param.name_hint
        dtype = free_param.type_annotation.dtype
        shape = [s.value for s in free_param.type_annotation.shape]
        inputs[name] = tvm.nd.array(np.random.uniform(0, 127, shape).astype(dtype))

    # Run for both type of compilation
    device = Device()
    outputs = []
    for bnns in [False, True]:
        outputs.append(build_and_run(func, inputs, 1, params, device, enable_bnns=bnns)[0])

    # Compare result tensors
    verify(outputs, atol=atol, rtol=rtol)


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
    r_factor: Optional[int]
        The repeat factor.

    Returns
    -------
    result: List[Tuple]
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
