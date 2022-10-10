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
import numpy as np
import pytest
from io import StringIO
import csv
import os
import json
import platform

import tvm.testing
import tvm.utils
from tvm.runtime import profiler_vm
from tvm import relay
from tvm.relay.testing import mlp
from tvm.contrib.debugger import debug_executor
from tvm import rpc
from tvm.contrib import utils
from tvm.runtime.profiling import Report
from tvm.script import tir as T


@tvm.testing.parametrize_targets("llvm", "cuda")
def test_estimate_peak_flops(target, dev):
    server = rpc.Server(key="roofline_flops")
    remote = rpc.connect("127.0.0.1", server.port, key="roofline_flops")
    dev = remote.device(target)
    # This test uses vectorized instructions so we need a target that supports them
    if target == "llvm":
        target = "llvm -mattr=+fma,+avx2"
    target = tvm.target.Target(target)
    with target:
        flops = tvm.utils.roofline.registry.estimate_peak_flops(target, dev, remote)
    if str(target.kind) == "llvm":
        # Assume we can achieve 1 GFLOP/s per thread, which is 1 FLOP per cycle on a 1GHz cpu.
        assert (
            flops > 10**9 and flops < 10**14
        ), f"FLOP/s should be between 10^9 and 10^14, but it is {flops}"
    elif str(target.kind) == "cuda":
        # should be able to hit a TFLOP/s with tensor cores
        assert (
            flops > 10**12 and flops < 10**14
        ), f"FLOP/s should be between 10^12 and 10^14, but it is {flops}"
    else:
        raise RuntimeError("Unsupported target " + str(target))


@tvm.testing.skip_if_32bit(reason="Cannot allocate enough memory on i386")
@tvm.testing.parametrize_targets("llvm", "cuda")
def test_estimate_peak_bandwidth(target, dev):
    server = rpc.Server(key="roofline_bandwidth")
    remote = rpc.connect("127.0.0.1", server.port, key="roofline_bandwidth")
    dev = remote.device(target)
    # This test uses vectorized instructions so we need a target that supports them
    if target == "llvm":
        target = "llvm -mattr=+fma,+avx2"
    target = tvm.target.Target(target)
    with target:
        bandwidth = tvm.utils.roofline.registry.estimate_peak_bandwidth(target, dev, remote)
    if str(target.kind) == "llvm":
        # Assume we can achieve 1 GB/s. DDR2 should transfer somewhere around 6
        # GB/s, so this should leave enough wiggle room.
        assert (
            bandwidth > 10**9 and bandwidth < 10**12
        ), f"Bandwidth should be between 10^9 and 10^12, but it is {bandwidth}"
    elif str(target.kind) == "cuda":
        # should be able to hit a 100 GB/s on a GPU. GTX 280 hits 140 GB/s and
        # it is really old.
        assert (
            bandwidth > 10**11 and bandwidth < 10**13
        ), f"Bandwidth should be between 10^9 and 10^12, but it is {bandwidth}"
    else:
        raise RuntimeError("Unsupported target " + str(target))


@tvm.testing.skip_if_32bit(reason="Cannot allocate enough memory on i386")
@tvm.testing.parametrize_targets("llvm -mattr=+fma+avx2", "cuda")
def test_roofline_analysis(target, dev):
    a = relay.var("a", relay.TensorType((512, 512), "float32"))
    b = relay.var("b", relay.TensorType((512, 512), "float32"))
    c = relay.nn.dense(a, b)
    mod = tvm.IRModule.from_expr(relay.Function([a, b], c))
    params = {}

    server = rpc.Server(key="roofline")
    remote = rpc.connect("127.0.0.1", server.port, key="roofline")
    dev = remote.device(target)

    report = tvm.utils.roofline_analysis(mod, params, target, dev, remote=remote)
    print(report)

    assert "Bound" in report.table()
    assert "Percent of Theoretical Optimal" in report.table()
    for call in report.calls:
        if "Percent of Theoretical Optimal" in call:
            if target.startswith("llvm"):
                # Ideally we'd like a little tighter bound here, but it is hard to
                # know how well this dense will perform without tuning. And we
                # don't have an operator that uses a specific number of flops.
                assert call["Percent of Theoretical Optimal"].ratio >= 5.0
            elif target == "cuda":
                # The cuda gpu kernel is really poorly optimized
                assert 90 >= call["Percent of Theoretical Optimal"].ratio >= 0.01


if __name__ == "__main__":
    tvm.testing.main()
