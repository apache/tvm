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

import os
import os.path
import pathlib
import sys
import pytest
import numpy as np
import logging
import tempfile
import csv

import tvm.testing
from tvm import te
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from tvm.contrib import utils, ndk
from tvm.contrib.hexagon.build import HexagonLauncher
import tvm.contrib.hexagon as hexagon

from .conftest import requires_hexagon_toolchain

RPC_SERVER_PORT = 7070

# This is a fixed detail of the v68 architecture.
HVX_VECTOR_BYTES = 128

# NOTE on server ports:
# These tests use different port numbers for the RPC server (7070 + ...).
# The reason is that an RPC session cannot be gracefully closed without
# triggering TIME_WAIT state on the server socket. This prevents another
# server to bind to the same port until the wait time elapses.


@requires_hexagon_toolchain
def test_elemwise_add(android_serial_number, hexagon_launcher):
    """
    Starting with an elementwise-add computation, try various schedules / optimizations to
    see the impact they have on performance.

    The main motivation for this test is to explore the relationship between these
    schedules / optimizations vs. how effectively the primfunc uses the Hexagon's
    HVX units.
    """
    host_output_dir = tempfile.mkdtemp()

    print("-" * 80)
    print("OUTPUT DIRECTORY: {}".format(host_output_dir))
    print("-" * 80)
    print()

    # TODO: We should move this into a separate test fixture, to make it easier to write
    # additional benchmarking functions.  We'd just need to generalize the assumptions regarding
    # the particular fields being tracked as independent variables.
    class benchmark_results_collection:
        def __init__(self):
            self.row_dicts_ = []

        def num_failures(self):
            num = 0
            for d in self.row_dicts_:
                if d["status"] == "FAIL":
                    num += 1
            return num

        def num_skips(self):
            num = 0
            for d in self.row_dicts_:
                if d["status"] == "SKIP":
                    num += 1
            return num

        def record_success(
            self, dtype, sched_type, mem_scope, num_vecs_per_tensor, benchmark_result
        ):
            median_usec = benchmark_result.median * 1000000
            min_usec = benchmark_result.min * 1000000
            max_usec = benchmark_result.max * 1000000

            self.row_dicts_.append(
                {
                    "dtype": dtype,
                    "sched_type": sched_type,
                    "mem_scope": mem_scope,
                    "num_vecs_per_tensor": num_vecs_per_tensor,
                    "status": "OK",
                    "median(µsec)": f"{median_usec:.3}",
                    "min(µsec)": f"{min_usec:.3}",
                    "max(µsec)": f"{max_usec:.3}",
                }
            )

        def record_failure(self, dtype, sched_type, mem_scope, num_vecs_per_tensor, error_text):
            self.row_dicts_.append(
                {
                    "dtype": dtype,
                    "sched_type": sched_type,
                    "mem_scope": mem_scope,
                    "num_vecs_per_tensor": num_vecs_per_tensor,
                    "status": "FAIL",
                    "comment": error_text,
                }
            )

        def record_skip(self, dtype, sched_type, mem_scope, num_vecs_per_tensor, comment_text):
            self.row_dicts_.append(
                {
                    "dtype": dtype,
                    "sched_type": sched_type,
                    "mem_scope": mem_scope,
                    "num_vecs_per_tensor": num_vecs_per_tensor,
                    "status": "SKIP",
                    "comment": comment_text,
                }
            )

        def dump(self, f):
            csv.register_dialect(
                "benchmarks",
                delimiter="\t",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
            )

            fieldnames = [
                "dtype",
                "sched_type",
                "mem_scope",
                "num_vecs_per_tensor",
                "status",
                "median(µsec)",
                "min(µsec)",
                "max(µsec)",
                "comment",
            ]

            writer = csv.DictWriter(f, fieldnames, dialect="benchmarks", restval="")

            writer.writeheader()
            for d in self.row_dicts_:
                writer.writerow(d)

    br = benchmark_results_collection()

    # Create and benchmark a single primfunc.
    # If an unexpected problem occurs, raise an exception.  Otherwise add a row of output to 'br'.
    def test_one_config(dtype, sched_type, mem_scope, num_vectors_per_tensor):
        version_name = f"dtype:{dtype}-schedtype:{sched_type}-memscope:{mem_scope}-numvecs:{num_vectors_per_tensor}"
        print(f"CONFIGURATION: {version_name}")

        if num_vectors_per_tensor == 1 and mem_scope == "global.vtcm":
            # 2022-04-12 (cconvey): There's currently a bug in which TVM doesn't
            # recognize the mapping of 1D memory <--> 2D memory as being bijective
            # when num_vectors_per_tensor == 1.
            br.record_skip(
                dtype,
                sched_type,
                mem_scope,
                num_vectors_per_tensor,
                f"Expect to hit bug where 1D-2D bijective transform not recognized.",
            )
            return

        if num_vectors_per_tensor == 2048 and mem_scope == "global.vtcm":
            br.record_skip(
                dtype,
                sched_type,
                mem_scope,
                num_vectors_per_tensor,
                f"Expect to exceed VTCM budget.",
            )
            return

        dtype_bits = tvm._ffi.runtime_ctypes.DataType(dtype).bits
        assert dtype_bits % 8 == 0
        dtype_bytes = dtype_bits // 8

        elem_per_hvx_vector = HVX_VECTOR_BYTES // dtype_bytes

        # Note!  We're providing the complete input tensor shapes now,
        # whereas the original code only reveals the exact shape when
        # about to call the kernel.

        shape = [
            num_vectors_per_tensor,
            elem_per_hvx_vector,
        ]

        A = tvm.te.placeholder(shape, dtype=dtype)
        B = tvm.te.placeholder(shape, dtype=dtype)
        C = tvm.te.compute(A.shape, lambda i, j: A[i, j] + B[i, j], name="C")

        sched = tvm.te.create_schedule(C.op)

        if sched_type == 1:
            pass
        elif sched_type == 2:
            sched[C].vectorize(C.op.axis[1])
        else:
            raise Exception("Unknown schedule type")

        # If we're using VTCM, we *must* add a transform_layout step to the schedule.
        # Otherwise the generated code will crash.
        # As of 2022-04-12 the crash does not provide a useful error message to the
        # host Python code.
        if mem_scope == "global.vtcm":
            for tensor in [A, B, C]:
                sched[tensor].transform_layout(lambda i, j: [i, te.AXIS_SEPARATOR, j])

        # This module is only created so humans can inspect its IR.
        module_for_ir_dump = tvm.lower(sched, [A, B, C], "foo")

        report_path = os.path.join(host_output_dir, f"{version_name}.txt")

        with open(report_path, "w") as f:
            f.write("LOWERED IR MODULE:\n")
            f.write(str(module_for_ir_dump))
            f.write("\n")

            target_hexagon = tvm.target.hexagon("v68", link_params=True)
            func = tvm.build(
                sched,
                [A, B, C],
                tvm.target.Target(target_hexagon, host=target_hexagon),
                name="elemwise_add",
            )

            host_dso_binary_path = os.path.join(host_output_dir, f"test_binary-{version_name}.so")
            target_dso_binary_filename = "test_binary.so"

            func.save(str(host_dso_binary_path))
            print("SAVED BINARY TO HOST PATH: {}".format(str(host_dso_binary_path)))

            hexagon_launcher.upload(host_dso_binary_path, target_dso_binary_filename)

            try:
                with hexagon_launcher.start_session() as sess:
                    mod = hexagon_launcher.load_module(target_dso_binary_filename, sess)

                    host_numpy_A_data = np.ndarray(shape, dtype=dtype)
                    host_numpy_B_data = np.ndarray(shape, dtype=dtype)

                    for i in range(shape[0]):
                        for j in range(shape[1]):
                            host_numpy_A_data[i, j] = i + j
                            host_numpy_B_data[i, j] = (i + 1) * (j + 1)

                    host_numpy_C_data_expected = host_numpy_A_data + host_numpy_B_data

                    A_data = tvm.nd.empty(shape, dtype, sess.device, mem_scope)
                    A_data.copyfrom(host_numpy_A_data)

                    B_data = tvm.nd.empty(shape, dtype, sess.device, mem_scope)
                    B_data.copyfrom(host_numpy_B_data)

                    C_data = tvm.nd.empty(shape, dtype, sess.device, mem_scope)

                    # NOTE: We may want to soften these numbers, depending on future findings.
                    timer = mod.time_evaluator("elemwise_add", sess.device, number=10, repeat=1)
                    timing_result = timer(A_data, B_data, C_data)

                    print("TIMING RESULT: {}".format(timing_result))

                    # Verify that the computation actually happened, and produced the correct result.
                    result = C_data.numpy()
                    tvm.testing.assert_allclose(host_numpy_C_data_expected, result)

                    br.record_success(
                        dtype, sched_type, mem_scope, num_vectors_per_tensor, timing_result
                    )

            except Exception as err:
                f.write("ERROR:\n")
                f.write("{}\n".format(err))
                br.record_failure(
                    dtype, sched_type, mem_scope, num_vectors_per_tensor, f"See {report_path}"
                )

    # -----------------------------------------------------------------------------------------------

    # Hexagon v69 allows more dtypes, but we're sticking with v68 for now.
    for dtype in [
        "int8",
    ]:

        # These numbers are only meaningful in the context of this script.
        for sched_type in [
            1,
            2,
        ]:

            for mem_scope in ["global", "global.vtcm"]:

                # These numbers are fairly arbitrary, but they're meant to stress memory/caches to
                # various extents.
                for num_vectors_per_tensor in [
                    1,
                    16,
                    64,
                    512,
                    2048,
                ]:

                    test_one_config(dtype, sched_type, mem_scope, num_vectors_per_tensor)

                    # Report our progress.
                    br.dump(sys.stdout)

    print("-" * 80)
    print(f"OUTPUT DIRECTORY: {host_output_dir}")
    print("-" * 80)
    print()

    tabular_output_filename = os.path.join(host_output_dir, "benchmark-results.csv")
    with open(tabular_output_filename, "w") as csv_file:
        br.dump(csv_file)
    print(f"BENCHMARK RESULTS FILE: {tabular_output_filename}")

    if br.num_failures() > 0:
        pytest.fail("At least one benchmark configuration failed", pytrace=False)
