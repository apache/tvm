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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Canonical NVIDIA IKET workload.

Run as a normal Python program; :func:`iket.run` performs the replay::

  python tests/python/tirx/iket/iket_profile_workload.py \
    --output-dir /tmp/tvm-iket-workload
"""

import argparse
import json
import os
from functools import partial
from pathlib import Path

import numpy as np

import tvm
from tvm.script import tirx as T
from tvm.tirx.cuda import iket


@T.prim_func
def canonical_iket_workload(out: T.Buffer((32,), "int32")):
    T.device_entry()
    profiler = iket.IketProfiler()
    tx = T.thread_id([32])
    token = profiler.sentinel_token("token")
    profiler.range_end(token)
    token = profiler.range_start("token")
    profiler.mark("checkpoint")
    profiler.range_end(token)
    profiler.range_push("stack")
    profiler.mark("inside_stack")
    profiler.range_pop()
    out[tx] = tx + 1


@T.prim_func
def native_payload_workload(out: T.Buffer((32,), "int32")):
    T.device_entry()
    profiler = iket.IketProfiler()
    tx = T.thread_id([32])
    profiler.mark("lane_payload", tx + 100)
    if tx >= 5:
        profiler.mark("first_active_lane", tx)
    profiler.mark("wide_payload", T.int64(tx) + T.int64(0x100000000))
    profiler.mark("negative_payload", T.int32(-32))
    profiler.mark("bool_true_payload", tx == 0)
    profiler.mark("bool_false_payload", tx != 0)
    profiler.mark("float32_payload", T.float32(-3.25))
    profiler.mark("float64_payload", T.float64(6.5))
    token = profiler.range_start("token_payload", tx + 200)
    profiler.range_end(token, tx + 300)
    profiler.range_push("stack_payload", tx + 400)
    profiler.range_pop()
    out[tx] = tx + 2


@T.prim_func
def extended_payload_workload(out: T.Buffer((32,), "int32")):
    T.device_entry()
    profiler = iket.IketProfiler()
    tx = T.thread_id([32])
    profiler.mark("extended_lane_payload", tx + 500)
    profiler.mark("extended01")
    profiler.mark("extended02")
    profiler.mark("extended03")
    profiler.mark("extended04")
    profiler.mark("extended05")
    profiler.mark("extended06")
    profiler.mark("extended07")
    profiler.mark("extended08")
    profiler.mark("extended09")
    profiler.mark("extended10")
    profiler.mark("extended11")
    profiler.mark("extended12")
    profiler.mark("extended13")
    profiler.mark("extended14")
    profiler.mark("extended15")
    profiler.mark("extended16")
    profiler.mark("extended17")
    profiler.mark("extended18")
    profiler.mark("extended19")
    profiler.mark("extended20")
    profiler.mark("extended21")
    profiler.mark("extended22")
    profiler.mark("extended23")
    profiler.mark("extended24")
    profiler.mark("extended25")
    profiler.mark("extended26")
    profiler.mark("extended27")
    profiler.mark("extended28")
    profiler.mark("extended29")
    profiler.mark("extended30")
    out[tx] = tx + 3


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/tmp/tvm-iket-workload")
    parser.add_argument(
        "--postprocess",
        choices=("perfetto", "json", "html", "none", "all"),
        default="all",
    )
    parser.add_argument("--clobber", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-ts-cnt-per-warp", type=int, default=None)
    parser.add_argument(
        "--fail-capture",
        action="store_true",
        help="Fail only in the real capture pass to verify error propagation",
    )
    return parser.parse_args()


def _injection_tool_name():
    config_path = os.environ.get("SMODEL_INJECTION_CONFIG")
    if not config_path:
        return None
    return json.loads(Path(config_path).read_text(encoding="utf-8")).get("toolName")


def _profile_workload(args):
    if args.fail_capture and _injection_tool_name() == "iket":
        raise RuntimeError("intentional capture-only IKET workload failure")
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    workloads = (
        (canonical_iket_workload, 1),
        (native_payload_workload, 2),
        (extended_payload_workload, 3),
    )
    outputs = []
    for workload, offset in workloads:
        # Plain JIT compilation is intentionally used here.  The validated
        # run-iket child enables LowerIket automatically for these modules.
        executable = tvm.compile(workload, target=target, tir_pipeline="tirx")
        module = executable.jit()
        out = tvm.runtime.empty((32,), "int32", device=tvm.cuda())
        module.main(out)
        outputs.append((out, offset))
    tvm.cuda().sync()
    for out, offset in outputs:
        np.testing.assert_array_equal(out.numpy(), np.arange(32, dtype=np.int32) + offset)


def main():
    args = _parse_args()
    result = iket.run(
        partial(_profile_workload, args),
        output_dir=args.output_dir,
        postprocess=args.postprocess,
        clobber=args.clobber,
        keep=args.keep,
        timeout=args.timeout,
        max_ts_cnt_per_warp=args.max_ts_cnt_per_warp,
    )
    print(f"IKET output directory: {result.output_dir}")
    for path in (*result.json_traces, *result.perfetto_traces, *result.html_reports):
        print(f"IKET artifact: {path}")


if __name__ == "__main__":
    main()
