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

"""Mostly helper methods which interface the main C++ Collage implementation with Python.
   See relay.transform.CollagePartition for the main Collage entrypoint."""

import tvm
from tvm._ffi.registry import register_func, register_object
from tvm.runtime import Object
from . import _ffi_api
import numpy as np
import logging
import os
import math
import tempfile

# Parameters to use when estimating latency (of both partitions and overall models).
MEASURE_NUMBER = 20
MEASURE_REPEAT = 5
WARMUP_MIN_REPEAT_MS = 250


@register_object("relay.collage.CostEstimator")
class CostEstimator(Object):
    """CostEstimator class"""

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.CostEstimator)


@register_object("relay.collage.MockEstimator")
class MockEstimator(Object):
    """MockEstimator class"""

    def __init__(self, target_costs):
        self.__init_handle_by_constructor__(_ffi_api.MockEstimator, target_costs)


def arg_for(type, device):
    """Returns a test argument of type on device"""
    assert isinstance(type, tvm.ir.TensorType)
    return tvm.nd.array(
        np.random.uniform(-1.0, 1.0, size=type.concrete_shape).astype(type.dtype), device=device
    )


def vm_estimate_seconds(device, vm, func_name, args):
    """Returns the estimated latency, in seconds, of running func_name with args on the given vm."""
    # Warmup
    vm.benchmark(
        device, repeat=1, number=1, min_repeat_ms=WARMUP_MIN_REPEAT_MS, func_name=func_name, **args
    )
    # One more time, with feeling
    return vm.benchmark(
        device,
        repeat=MEASURE_REPEAT,
        number=MEASURE_NUMBER,
        min_repeat_ms=0,
        func_name=func_name,
        **args,
    )


@register_func("tvm.relay.collage.estimate_seconds")
def estimate_seconds(mod, target):
    """Returns the mean execution time of "main" in mod on target with params. The module
    may contain "Primitive" functions, possibly with "Compiler" attributes."""
    device = tvm.device(target.kind.device_type)

    try:
        # Build the module.
        logging.info("Compiling module to estimate")
        exe = tvm.relay.vm.compile(mod, target)
    except RuntimeError as e:
        # A build failure indicates the partition is not supported.
        # eg trying to build an nn.batch_norm on GPU, which has no schedule since we assume it
        # is only ever used with a tuple projection which is rewritten away.
        logging.info(f"Assigning module infinite cost since unable to build: {e}")
        return math.inf

    # Finalize compilation
    tmp_dir = tempfile.mkdtemp()
    code, lib = exe.save()
    lib_path = os.path.join(tmp_dir, "library.so")
    # TODO(mbs): Avoid nvcc dependency?
    lib.export_library(lib_path, workspace_dir=tmp_dir, cc="nvcc")
    lib = tvm.runtime.load_module(lib_path)
    exe = tvm.runtime.vm.Executable.load_exec(code, lib)

    # Benchmark the module.
    vm = tvm.runtime.vm.VirtualMachine(exe, device)
    func_name = "main"
    main_args = {v.name_hint: arg_for(v.checked_type, device) for v in mod[func_name].params}
    logging.info("Benchmarking module to estimate")
    profile = vm_estimate_seconds(device, vm, func_name, main_args)
    logging.info(f"profile: {profile}")
    return profile.median  # seconds


make_labelled_dfpattern_partition_rule = tvm._ffi.get_global_func(
    "relay.collage.make_labelled_dfpattern_partition_rule"
)
make_labelled_dfpattern_partition_rule_with_predicate = tvm._ffi.get_global_func(
    "relay.collage.make_labelled_dfpattern_partition_rule_with_predicate"
)
make_pattern_byoc_partition_rule = tvm._ffi.get_global_func(
    "relay.collage.make_pattern_byoc_partition_rule"
)


def make_labelled_dfpattern_partition_rule_wrapper(compiler, tuple):
    """Returns a DFPatternPartitionRule representing one (label, pattern, predicate) entry from
    the pattern table for external codegen compiler"""
    if len(tuple) == 2:
        rule_name, dataflow_pattern = tuple
        return make_labelled_dfpattern_partition_rule(compiler, rule_name, dataflow_pattern)
    else:
        rule_name, dataflow_pattern, predicate = tuple
        return make_labelled_dfpattern_partition_rule_with_predicate(
            compiler, rule_name, dataflow_pattern, predicate
        )


@register_func("tvm.relay.collage.make_byoc_partition_rule")
def make_byoc_partition_rule(compiler):
    """Returns the PartitionRule for external codegen compiler"""
    pattern_table = tvm.relay.op.contrib.get_pattern_table(compiler)
    assert (
        pattern_table is not None
    ), f"No pattern table entry was found for BYOC compiler {compiler}"
    logging.info(
        f"Converting {len(pattern_table)} rules for {compiler} for use in pattern style BYOC lowering/codegen"
    )
    sub_rules = [
        make_labelled_dfpattern_partition_rule_wrapper(compiler, tuple) for tuple in pattern_table
    ]
    return make_pattern_byoc_partition_rule(compiler, sub_rules)
