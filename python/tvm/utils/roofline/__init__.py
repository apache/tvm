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
"""Utilities for computing an approximate roofline model"""
from typing import Dict, Optional, Union

import numpy as np

from ... import IRModule, auto_scheduler, build, get_global_func, nd, relay, tir, topi, transform
from ...contrib import utils
from ...ir.expr import GlobalVar
from ...ir.instrument import pass_instrument
from ...rpc.base import RPC_SESS_MASK
from ...rpc.client import RPCSession
from ...runtime import Device, num_threads, profiler_vm, profiling
from ...script import tir as T
from ...target import Target
from . import cuda, registry, x86


def _create_args(mod: IRModule, dev: Device, func_name: str = "main", remote=None):
    if dev.device_type >= RPC_SESS_MASK:
        random_fill = remote.get_function("tvm.contrib.random.random_fill")
    else:
        random_fill = get_global_func("tvm.contrib.random.random_fill")
    assert random_fill, "Please make sure USE_RANDOM is ON in config.cmake"
    args = []
    for arg in mod[func_name].params:
        ary = nd.empty(
            [x.value for x in arg.type_annotation.shape],
            arg.type_annotation.dtype,
            device=dev,
        )
        random_fill(ary)
        args.append(ary)
    return args


@pass_instrument
class SaveLoweredTIR:
    """Save TIR functions from right before final lowering. Right now this
    means right before tir.MakePackedAPI."""

    def __init__(self, before_pass: str = "tir.MakePackedAPI"):
        """
        Parameters
        ----------
        before_pass: str
            Pass before which the TIR is saved.
        """
        self.functions = {}
        self.before_pass = before_pass

    def run_before_pass(self, mod, info):
        if info.name == self.before_pass:
            for v, func in mod.functions.items():
                if isinstance(func, tir.PrimFunc):
                    self.functions[v] = func


def roofline_from_existing(
    report: profiling.Report,
    tir_functions: Dict[GlobalVar, tir.PrimFunc],
    target: Target,
    dev: Device,
    remote: Optional[RPCSession] = None,
) -> profiling.Report:
    """Add roofline and other estimated statistics to an existing profiling report.

    :py:func:`roofline_analysis` should always be used instead of this function
    unless you need a custom compilation pipeline.

    Calculating roofline statistics requires features extracted the TIR
    functions in addition to per-operator runtime information (`report`) of the
    same TIR features. The features and TIR functions are not included with the
    compiled library used to generate the per-operator runtime. It is essential
    that the per-operator information comes from the exact same compilation
    pipeline as the TIR functions.


    Example
    -------

    ..code: : python

        import tvm
        import tvm.relay

        mod, params = tvm.relay.testing.mlp.get_workload()

        # it is recommended to use SaveLoweredTIR to get out the tir primfuncs
        save_tir = tvm.utils.roofline.SaveLoweredTIR()
        with tvm.transform.PassContext(opt_level=3, pass_instrument=[save_tir]):
            lib = relay.vm.compile(mod, params=params, target=target)

        vmexec = profiler_vm.VirtualMachineProfiler(lib, dev)
        report = vmexec.profile(*inputs)

        roofline_report = roofline_from_existing(report, save_tir.functions, target, dev)


    Parameters
    ----------
    report : Report
        Existing profiling report from :py:method:`VirtualMachineProfiler.profile`.
    tir_functions : Dict[GlobalVar, PrimFunc]
        TIR primfuncs from the module run to generate `report`. It is nessesary
        that these functions come before the `tir.MakePackedAPI` pass and are
        compatible with auto_scheduler featurization.
        :py:class:`SaveLoweredTIR` is the recommended way to collect these
        functions.
    target : Target
        TVM target that `report` was generated with.
    dev : Device
        Device that `report` was generated with.
    remote : Optional[RPCSession]
      Remote session used to upload artifacts for runtime evaluation. Must be
      the same session used to create `dev`.

    Returns
    -------
    profiling.Report
        New profiling report that includes all information from `report`
        along with additional roofline metrics. See
        :py:func:`roofline_analysis` for more information on which metrics
        are included.
    """

    all_features = {
        prim.attrs["hash"]: (name, prim, auto_scheduler.feature.named_features_from_primfunc(prim))
        for name, prim in tir_functions.items()
        if isinstance(prim, tir.PrimFunc) and "hash" in prim.attrs.keys()
    }

    new_calls = []
    for call in report.calls:
        if "Hash" in call.keys() and call["Hash"] in all_features:
            _, prim, features = all_features[call["Hash"]]
            if features is None:
                continue

            with target:
                flops, peak_flops, flops_name = registry.estimate_peak_flops(
                    prim, features, target, dev, remote
                )
                loaded_bytes, peak_bandwidth, bandwidth_name = registry.estimate_peak_bandwidth(
                    prim, features, target, dev, remote
                )
            ridge_point = peak_flops / peak_bandwidth

            runtime = call["Duration (us)"].microseconds * 1e-6
            arith_inten = flops / loaded_bytes
            call = dict(call)
            call["Loaded Bytes"] = profiling.Count(int(loaded_bytes))
            call["Estimated FLOPs"] = profiling.Count(int(flops))
            call["Arithmetic Intensity"] = profiling.Ratio(arith_inten)
            call["FLOP/s"] = profiling.Ratio(flops / runtime)
            call["Bandwidth"] = profiling.Ratio(loaded_bytes / runtime)
            compute_bound = arith_inten > ridge_point
            call["Bound"] = "compute" if compute_bound else "memory"
            per_mem_bound = (loaded_bytes / runtime) / peak_bandwidth * 100
            per_compute_bound = (flops / runtime) / peak_flops * 100.0
            # We use ratio here because the percentages should be averaged instead of summed.
            call["Percent of Theoretical Optimal"] = profiling.Ratio(
                per_compute_bound if compute_bound else per_mem_bound
            )
            new_calls.append(call)
        else:
            new_calls.append(call)
    new_configuration = dict(report.configuration.items())
    new_configuration[f"Estimated Peak FLOP/s ({flops_name})"] = profiling.Ratio(peak_flops)
    new_configuration[
        f"Estimated Peak Bandwidth ({bandwidth_name}, byte/second)"
    ] = profiling.Ratio(peak_bandwidth)
    return profiling.Report(new_calls, report.device_metrics, new_configuration)


def roofline_analysis(
    mod: IRModule,
    params: Dict[str, nd.NDArray],
    target: Union[str, Target],
    dev: Device,
    remote: Optional[RPCSession] = None,
) -> profiling.Report:
    """
    Create a profiling report that contains roofline and other estimated
    statistics from running a module on the VM.

    The roofline model measures how close a operator gets to best possible
    memory bandwidth or FLOP/s depending on whether it is memory or compute
    bound. This computation uses the runtime of the operator along with two
    numbers extracted from the TIR code: bytes of memory touched and number of
    floating point operations.

    These statistics are calculated by analyzing the lowered TIR of each
    operator, so they are estimates of the true values. The statistics are:
      - Bound: Is the operator memory or compute bound. This is computed by
        assuming that the operator could perfectly cache all loads -- each byte
        of memory is only loaded once.
      - Percent of Theoretical Optimal: What percent of theoretical optimal for
        the bound. i.e. percent of peak memory bandwidth if memory bound,
        percent of peak FLOP/s if compute bound.
      - Loaded Bytes: estimation of the number of bytes loaded from main memory.
      - Estimated Flops: estimated number of floating point operations.
      - Arithmetic Intensity: ratio of FLOPs per byte of data.
      - FLOP/s: floating point operations per second.
      - Bandwidth: Number of bytes loaded per second.

    Parameters
    ----------
    mod : IRModule
      Uncompiled input module

    params : Dict[str, nd.NDArray]

    target : Union[str, Target]
      Target to run on.

    dev : Device
      Device to run on.

    remote : Optional[RPCSession]
      Remote session used to upload artifacts for runtime evaluation. Must be
      the same session used to create `dev`.

    Returns
    -------

    report : profiling.Report
      Profiling report which includes the estimated statistics.
    """
    if isinstance(target, str):
        target = Target(target)

    save_tir = SaveLoweredTIR()
    # copy existing context but add our instrument
    pass_ctx = transform.PassContext.current()
    with transform.PassContext(
        opt_level=pass_ctx.opt_level,
        required_pass=pass_ctx.required_pass,
        disabled_pass=pass_ctx.disabled_pass,
        instruments=list(pass_ctx.instruments) + [save_tir],
        config=pass_ctx.config,
    ):
        lib = relay.vm.compile(mod, params=params, target=target)
    # upload to remote if running over rpc
    if dev.device_type >= RPC_SESS_MASK:
        if remote is None:
            raise RuntimeError("A RPCSession must be provided when using a remote device.")
        temp = utils.tempdir()
        path = temp.relpath("roofline_lib.tar")
        lib.mod.export_library(path)
        remote.upload(path)
        lib = remote.load_module("roofline_lib.tar")
    vmexec = profiler_vm.VirtualMachineProfiler(lib, dev)

    args = _create_args(mod, dev, remote=remote)
    report = vmexec.profile(*args)

    return roofline_from_existing(report, save_tir.functions, target, dev, remote=remote)
