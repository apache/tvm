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
from typing import Dict, Union, Optional
import numpy as np

from .. import auto_scheduler, relay, tir, nd, IRModule, build, topi, transform, get_global_func
from ..target import Target
from ..runtime import profiler_vm, profiling, Device, num_threads
from ..script import tir as T
from ..ir.instrument import pass_instrument
from ..ir.expr import GlobalVar
from ..rpc.base import RPC_SESS_MASK
from ..rpc.client import RPCSession
from ..contrib import utils


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


def _detect_vec_width_registers(
    target: Target, vec_width: Optional[int], num_vector_registers: Optional[int]
):
    """Get the vector width and number of vector registers for a target.

    Parameters
    ----------
    target : Target
        Target to detect vector width and registers for.
    vec_width : Optional[int]
        If None, try and detect vector width from target. Otherwise provided input is used.
    num_vector_registers : Optional[int]
        If None, try and number of vector registers from target. Otherwise provided input is used.

    Returns
    -------
    vec_width: int
        Width of a vector register on `target`.
    num_vector_registers: int
        Number of vector registers on `target`.
    """
    if vec_width is None:
        # Only implemented for x86 so far...
        if (
            str(target.kind) == "llvm"
            and target.device_name == ""
            and len(target.keys) == 1
            and target.keys[0] == "cpu"
        ):
            with target:
                vec_width = topi.x86.utils.get_simd_32bit_lanes()  # in number of float32s
        else:
            raise RuntimeError(f"Cannot determine vector width for target {target}")
    if num_vector_registers is None:
        if target.device_name == "":  # indicates x86
            num_vector_registers = 16  # Assuming for all platforms, probably wrong on older ones
        else:
            raise RuntimeError(f"Cannot determine number of vector registers for target {target}")
    return vec_width, num_vector_registers


@T.prim_func
def peakflops_fma_tir(
    a: T.handle,
    vec_width: T.int32,
    iters: T.int32,
    num_vector_registers: T.int32,
    threads: T.int32,
) -> None:
    # pylint: disable=invalid-name, missing-function-docstring
    A = T.match_buffer(a, [threads, num_vector_registers, vec_width], "float32")
    for t in T.parallel(threads):
        for _j in range(iters):
            for l in T.unroll(num_vector_registers):
                # We want to use as few registers as possible, so we perform
                # all operations on the same element
                for k in T.vectorized(vec_width):
                    A[t, l, k] = A[t, l, k] * A[t, l, k] + A[t, l, k]


def estimate_peak_fma_flops(
    target: Target,
    dev: Device,
    vec_width: Optional[int] = None,
    num_vector_registers: Optional[int] = None,
    remote: Optional[RPCSession] = None,
) -> float:
    """
    Estimate the maximum number of FLOP/s this target/device combo is capable
    of reaching by running a test program. This assumes vectorized f32 FMA
    (fused-multiply-add) instructions.


    Parameters
    ----------
    target : Target
        Target to run on. This should be as specific to the actual hardware as
        possible to make sure that LLVM generates the best vector code.
    dev : Device
        Device to run on.
    vec_width : Optional[int]
        Vector width of SIMD units on the underlying hardware. Will try to
        infer if no value is provided.
    num_vector_registers : Optional[int]
        Number of vector registers on the underlying hardware. Will try to
        infer if no value is provided.
    remote : Optional[RPCSession]
      Remote session used to upload artifacts for runtime evaluation. Must be
      the same session used to create `dev`.

    Returns
    -------
    float
        Approximate sustained FLOP/s of this target/device combo assuming
        vectorized f32 FMA instructions.
    """
    assert str(target.kind) == "llvm", "Only llvm targets are supported"
    vec_width, num_vector_registers = _detect_vec_width_registers(
        target, vec_width, num_vector_registers
    )
    iters = 1000000
    nthreads = num_threads()
    specialized = peakflops_fma_tir.specialize(
        {
            peakflops_fma_tir.params[1]: vec_width,
            peakflops_fma_tir.params[2]: iters,
            peakflops_fma_tir.params[3]: num_vector_registers,
            peakflops_fma_tir.params[4]: nthreads,
        }
    )
    with transform.PassContext(opt_level=3):
        f = build(specialized, target=target)

    # upload to remote if running over rpc
    if dev.device_type >= RPC_SESS_MASK:
        if remote is None:
            raise RuntimeError("A RPCSession must be provided when using a remote device.")
        temp = utils.tempdir()
        path = temp.relpath("peak_fma_flops.tar")
        f.export_library(path)
        remote.upload(path)
        f = remote.load_module("peak_fma_flops.tar")
        random_fill = remote.get_function("tvm.contrib.random.random_fill")
    else:
        random_fill = get_global_func("tvm.contrib.random.random_fill")
    assert random_fill, "Please make sure USE_RANDOM is ON in config.cmake"

    a = nd.empty((nthreads, num_vector_registers, vec_width), dtype="float32", device=dev)
    random_fill(a)
    times = f.time_evaluator(f.entry_name, dev, repeat=100, number=1)(a)
    flops = 2 * vec_width * num_vector_registers * nthreads * iters  # fma is two flops
    flop_s = flops / times.min
    return flop_s


@T.prim_func
def peak_bandwidth_tir(a: T.handle, b: T.handle, threads: T.int32, vec_width: T.int32) -> None:
    # pylint: disable=invalid-name, missing-function-docstring
    N = T.var("int32")
    A = T.match_buffer(a, [threads, N, 4, vec_width], "float32")
    B = T.match_buffer(b, [threads, vec_width, 4], "float32")
    # Parallelism is necessary to hit all cores/nodes
    for i in T.parallel(threads):
        for k in T.serial(N):
            for l in T.unroll(4):
                # vectorized load is necessary to hit peak bandwidth
                for j in T.vectorized(vec_width):
                    # += is necessary to introduce a data dependency for all
                    # elements of A, preventing the backend from removing the
                    # `k` loop and setting `k` to the loop extent.
                    B[i, l, j] += A[i, k, l, j]


def estimate_peak_bandwidth(
    target: Target,
    dev: Device,
    vec_width: Optional[int] = None,
    remote: Optional[RPCSession] = None,
) -> float:
    """Estimate peak memory bandwidth of a target/device combo.

    Peak bandwidth is estimated by running a small experiment on the underlying
    hardware. The peak bandwidth measurement assumes that vector instructions
    are being used to load the data.

    Parameters
    ----------
    target : Target
        Target to use for measurement. This target should be as specific to the
        underlying hardware as possible.
    dev : Device
        Device to measure peak bandwidth on.
    vec_width : Optional[int]
        Vector unit width, determined from target if not supplied.
    remote : Optional[RPCSession]
      Remote session used to upload artifacts for runtime evaluation. Must be
      the same session used to create `dev`.

    Returns
    -------
    float
        Peak memory bandwidth in bytes/seconds.
    """
    # Ideally we'd be able to use this code to measure peak bandwidth of the
    # different cache levels. If we could just generate load commands, then we
    # could use those in a tight loop. Instead we need some code that is
    # limited on the cache bandwidth. With the L1 cache we need an operation
    # that has a very low arithmetic intensity and we haven't come up with one
    # yet.
    vec_width, _ = _detect_vec_width_registers(target, vec_width, 1)
    specialized = peak_bandwidth_tir.specialize(
        {
            peak_bandwidth_tir.params[3]: vec_width,
        }
    )
    with transform.PassContext(opt_level=3):
        f = build(specialized, target=target)

    # upload to remote if running over rpc
    if dev.device_type >= RPC_SESS_MASK:
        if remote is None:
            raise RuntimeError("A RPCSession must be provided when using a remote device.")
        temp = utils.tempdir()
        path = temp.relpath("peak_bandwidth.tar")
        f.export_library(path)
        remote.upload(path)
        f = remote.load_module("peak_bandwidth.tar")
        random_fill = remote.get_function("tvm.contrib.random.random_fill")
    else:
        random_fill = get_global_func("tvm.contrib.random.random_fill")
    assert random_fill, "Please make sure USE_RANDOM is ON in config.cmake"

    threads = num_threads()
    # Data size needs to be larger than last level of cache. We don't have a
    # way of getting cache sizes, so this number should give us a large enough
    # size.
    size = 10**8 // (4 * threads * vec_width)
    a = nd.empty((threads, size, 4, vec_width), dtype="float32", device=dev)
    random_fill(a)
    b = nd.empty((threads, vec_width, 4), dtype="float32", device=dev)
    random_fill(b)
    times = f.time_evaluator(f.entry_name, dev, repeat=10, number=1)(a, b, threads)
    return a.numpy().size * 4 / times.min  # 4 bytes per float32


@pass_instrument
class SaveLoweredTIR:
    """Save TIR functions from right before final lowering. Right now this
    means right before tir.MakePackedAPI."""

    def __init__(self):
        self.functions = {}
        self.done = False

    def run_after_pass(self, mod, info):
        if not self.done:
            if info.name == "tir.MakePackedAPI":
                self.done = True
            else:
                for v, func in mod.functions.items():
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
    peak_bandwidth = estimate_peak_bandwidth(target, dev, remote=remote)
    peak_flops = estimate_peak_fma_flops(target, dev, remote=remote)

    ridge_point = peak_flops / peak_bandwidth

    all_features = {
        prim.attrs["hash"]: (name, auto_scheduler.feature.named_features_from_primfunc(prim))
        for name, prim in tir_functions.items()
        if isinstance(prim, tir.PrimFunc) and "hash" in prim.attrs.keys()
    }

    new_calls = []
    for call in report.calls:
        if "Hash" in call.keys():
            _, features = all_features[call["Hash"]]

            flops = np.sum(features["float_addsub"] + features["float_mul"] + features["float_mad"])
            loaded_bytes = 0.0
            # assume no more than 100 buffers
            for i in range(100):
                key = f"B{i}.bytes"
                if not key in features.keys():
                    break
                loaded_bytes += np.sum(features[key])
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
            per_compute_bound = flops / peak_flops * 100.0
            # We use ratio here because the percentages should be averaged instead of summed.
            call["Percent of Theoretical Optimal"] = profiling.Ratio(
                per_compute_bound if compute_bound else per_mem_bound
            )
            new_calls.append(call)
        else:
            new_calls.append(call)
    return profiling.Report(new_calls, report.device_metrics)


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
      Uncompiled input module>

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
