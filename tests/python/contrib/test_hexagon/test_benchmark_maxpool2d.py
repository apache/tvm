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

"""
This module serves two purposes:
    (1) Demonstrates how to write Python code that exercises various
        Hexagon-related algorithms / features.

    (2) Benchmark the resulting primfuncs.

Current limitations:
    - Input shapes are limited to NHWC --> NHWC_8h8w32c.

    - Testing parameters (input shapes, dtypes, etc.) currently
      support only one value for each parameter.

    - H, W, C must be integer multiples of 8, 8, and 32,
      respectively.  I.e., partial blocks aren't currently
      supported by this script.

    - Requires that I/O tensors reside in "global.VTCM" memory,
      rather than "global" memory.
      This prevents benchmarking with I/O tensors that are too
      large to fit into availble VTCM.

    - The script only develops one primfunc.
      Future revisions to this script are expected to add more
      primfuncs and demonstrate more coding strategies.
"""

import pytest
import numpy as np
import copy
import os

import tvm.testing
from tvm import te, topi, tir
from tvm.topi import testing
from tvm.contrib.hexagon.session import Session
from typing import List

from .infrastructure import allocate_hexagon_array, get_hexagon_target
from . import benchmark_util as bu

# Pytest seems to require that fixture names exist in the current module.
# E.g., it doesn't allow: @pytest.mark.usefixtures("bu.benchmark_group")
benchmark_group = bu.benchmark_group

_SHOULD_SKIP_BENCHMARKS, _SKIP_BENCHMARKS_REASON = bu.skip_bencharks_flag_and_reason()


def _ceil_div(numerator, denominator):
    return (numerator + (denominator - 1)) // denominator


def _int8_nhwc_8h8w32c_map(n, h, w, c):
    return [
        n,
        h // 8,
        w // 8,
        c // 32,
        te.AXIS_SEPARATOR,
        h % 8,
        w % 8,
        c % 32,
    ]


def _int8_nhwc_8h8w32c_shape(n, h, w, c) -> List[int]:
    return [
        n,
        _ceil_div(h, 8),
        _ceil_div(w, 8),
        _ceil_div(c, 32),
        8,
        8,
        32,
    ]


def _int8_nhwc_8h8w32c_xform_immediate(arr_in: np.ndarray) -> np.ndarray:
    """
    Return a deep copy of 'arr_in', transformed from a NWHC to
    NHWC-8h8wc32 shape.  Any newly created array elements have value 0.
    """
    stage1 = copy.copy(arr_in)

    (
        n,
        h,
        w,
        c,
    ) = stage1.shape

    (
        h_minor,
        w_minor,
        c_minor,
    ) = [8, 8, 32]

    h_major = _ceil_div(h, h_minor)
    w_major = _ceil_div(w, w_minor)
    c_major = _ceil_div(c, c_minor)

    # This handles cases where the dimensions of arr_in are not cleanly divided
    # by the minor block size, i.e. [8, 8, 32].
    #
    # Any additional array elements that this creates will ahve value 0.
    # We shouldn't actually care what value is used for those elements, because they
    # shouldn't be treated as meaningful by any of our algorithms.
    if (h % h_minor) or (w % w_minor) or (c % c_minor):
        stage1.resize((n, h_major * h_minor, w_major * w_minor, c_major * c_minor), refcheck=False)

    stage2 = stage1.reshape(n, h_major, h_minor, w_major, w_minor, c_major, c_minor)
    stage3 = stage2.transpose(0, 1, 3, 5, 2, 4, 6)
    return stage3


def _create_test_input(shape, dtype: str) -> np.ndarray:
    np_dtype = np.dtype(dtype)
    min_value = np.iinfo(np_dtype).min
    max_value = np.iinfo(np_dtype).max
    return np.random.randint(low=min_value, high=max_value, size=tuple(shape), dtype=np.int8)


@pytest.mark.usefixtures("benchmark_group")
class TestMaxPool2D:
    csv_column_order = [
        # Identifies which TE-compute / TIRScript is used as the basis for the
        # benchmarked primfunc. Only needs to be meaningful to humans.
        "basic_kernel",
        # When applicable, indicates the particular variation of schedules
        # apply by the Python code. Decoding this may require looking at this
        # script's source code.
        "sched_type",
        # Values directly based on test parameters...
        "input_shape_4d",
        "block_shape",
        "DTYPE",
        "KERNEL",
        "STRIDE",
        "DILATION",
        "PADDING",
        "IO_TENSOR_MEM_SCOPE",
        # Reserved columns defined by the BenchmarksTable class.
        "row_status",
        "timings_min_usecs",
        "timings_max_usecs",
        "timings_median_usecs",
        "timings_mean_usecs",
        "timings_stddev_usecs",
        # For benchmarks that produce files on the host file system, this indicates
        # their location. Useful for post-mortem investigation of benchmark results.
        "host_files_dir_path",
        # Miscellaneous comments about the benchmark.
        "comments",
    ]

    DTYPE = tvm.testing.parameter("int8")

    # FIXME(cconvey): The script currently fails when H, W, or C is not an
    # integer multiple of 8, 8, or 32, respectively.
    N = tvm.testing.parameter(1)
    H = tvm.testing.parameter(*[x * 8 for x in [1, 4, 16]])
    W = tvm.testing.parameter(*[x * 8 for x in [1, 4, 16]])
    C = tvm.testing.parameter(*[x * 32 for x in [1, 2]])

    KERNEL = tvm.testing.parameter((1, 1), (3, 3))
    STRIDE = tvm.testing.parameter((1, 1))
    DILATION = tvm.testing.parameter((1, 1))
    PADDING = tvm.testing.parameter((0, 0, 0, 0))
    IO_TENSOR_MEM_SCOPE = tvm.testing.parameter("global.vtcm")

    @pytest.mark.skipif(_SHOULD_SKIP_BENCHMARKS, reason=_SKIP_BENCHMARKS_REASON)
    @tvm.testing.requires_hexagon
    def test_maxpool2d_nhwc(
        self,
        N,
        H,
        W,
        C,
        DTYPE,
        KERNEL,
        STRIDE,
        DILATION,
        PADDING,
        IO_TENSOR_MEM_SCOPE,
        hexagon_session: Session,
    ):
        keys_dict = {
            "basic_kernel": "max_pool2d",
            "sched_type": 1,
            "input_shape_4d": [N, H, W, C],
            "block_shape": [8, 8, 32],
            "DTYPE": DTYPE,
            "KERNEL": KERNEL,
            "STRIDE": STRIDE,
            "DILATION": DILATION,
            "PADDING": PADDING,
            "IO_TENSOR_MEM_SCOPE": IO_TENSOR_MEM_SCOPE,
        }

        desc = bu.get_benchmark_decription(keys_dict)

        # Create the host-side directory for this benchmark run's files / logs...
        host_files_dir_name = bu.get_benchmark_id(keys_dict)
        host_files_dir_path = os.path.join(self.working_dir, host_files_dir_name)
        os.mkdir(host_files_dir_path)

        keys_dict["host_files_dir_path"] = host_files_dir_path

        log_file_path = os.path.join(host_files_dir_path, "out.txt")
        with open(log_file_path, "w") as log_file:
            print(f"CONFIGURATION: {desc}")
            log_file.write(f"CONFIGURATION: {desc}\n")

            try:
                input_tensor_shape_4d = [N, H, W, C]
                input_tensor_shape_7d = _int8_nhwc_8h8w32c_shape(N, H, W, C)

                data = te.placeholder(tuple(input_tensor_shape_4d), dtype=DTYPE)

                output = topi.nn.pool2d(
                    data, KERNEL, STRIDE, DILATION, PADDING, "max", layout="NHWC"
                )
                primfunc = te.create_prim_func([data, output])

                sch = tir.Schedule(primfunc, debug_mask="all")

                sch.transform_layout(
                    block="tensor", buffer="placeholder", index_map=_int8_nhwc_8h8w32c_map
                )

                built_module = tvm.build(
                    sch.mod,
                    target=get_hexagon_target("v69"),
                )

                # Save a local copy of the Hexagon object code (in the form of a .so file)
                # to allow post-mortem inspection.
                host_dso_binary_path = os.path.join(host_files_dir_path, "test_binary.so")
                built_module.save(host_dso_binary_path)
                print(f"SAVED BINARY TO HOST PATH: {host_dso_binary_path}")

                hexagon_mod = hexagon_session.load_module(built_module)

                # Generate the input tensor's data.
                # Note that we'll eventually need it in two different layouts:
                # (1) NHWC as an argument to testing.poolnd_python.
                # (2) NHWC_8h8w32c for as an argument to our Hexagon primfunc.
                # a_numpy_4d = np.random.randint(low=-128, high=127, size=input_tensor_shape_4d, dtype=np.int8)
                a_numpy_4d = _create_test_input(input_tensor_shape_4d, DTYPE)

                ref_output_4d = testing.poolnd_python(
                    a_numpy_4d.astype("int32"),
                    KERNEL,
                    STRIDE,
                    DILATION,
                    PADDING[0:2],
                    PADDING[2:],
                    pool_type="max",
                    dtype="int32",
                    layout="NHWC",
                ).astype(DTYPE)

                output_tensor_shape_4d = ref_output_4d.shape

                a_numpy_7d = _int8_nhwc_8h8w32c_xform_immediate(a_numpy_4d)

                a_hexagon_7d = allocate_hexagon_array(
                    hexagon_session.device,
                    tensor_shape=input_tensor_shape_7d,
                    axis_separators=[4],
                    dtype=DTYPE,
                    mem_scope=IO_TENSOR_MEM_SCOPE,
                )

                c_hexagon_4d = allocate_hexagon_array(
                    hexagon_session.device,
                    tensor_shape=output_tensor_shape_4d,
                    axis_separators=[],
                    dtype=DTYPE,
                    mem_scope=IO_TENSOR_MEM_SCOPE,
                )

                a_hexagon_7d.copyfrom(a_numpy_7d)

                if DTYPE == "int8":
                    rel_tolerance = 0
                    abs_tolerance = 0
                else:
                    assert False, f"TODO: decide acceptable tolerances for DTYPE {DTYPE}"

                # hexagon_mod(a_hexagon_7d, c_hexagon_4d)
                # tvm.testing.assert_allclose(ref_output_4d, c_hexagon_4d.numpy(), rtol=rel_tolerance, atol=abs_tolerance)

                timer = hexagon_mod.time_evaluator(
                    "main", hexagon_session.device, number=10, repeat=1
                )
                timing_result = timer(a_hexagon_7d, c_hexagon_4d)

                try:
                    tvm.testing.assert_allclose(
                        ref_output_4d, c_hexagon_4d.numpy(), rtol=rel_tolerance, atol=abs_tolerance
                    )
                except AssertionError as e:
                    raise bu.NumericalAccuracyException(str(e))

            except bu.NumericalAccuracyException as e:
                print()
                print(f"FAIL: Numerical accuracy error. See log file.")

                log_file.write("\n")
                log_file.write(f"FAIL: {e}\n")

                self.benchmark_table.record_fail(
                    **keys_dict, comments=f"Numerical accuracy error. See log file."
                )

            except bu.UnsupportedException as e:
                print()
                print(f"SKIP: {e}")

                log_file.write("\n")
                log_file.write(f"SKIP: {e}\n")

                self.benchmark_table.record_skip(
                    **keys_dict, comments=f"Unsupported configuration: {e}"
                )

            self.benchmark_table.record_success(timing_result, **keys_dict)


if __name__ == "__main__":
    tvm.testing.main()
