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
""" benchmark_elemwise_add """

import os
import os.path
import sys
import tempfile

import numpy as np
import pytest

import tvm.script
import tvm.testing
from tvm.contrib.hexagon.session import Session
from tvm.script import tir as T

from . import benchmark_util as bu
from .infrastructure import get_hexagon_target

_SHOULD_SKIP_BENCHMARKS, _SKIP_BENCHMARKS_REASON = bu.skip_benchmarks_flag_and_reason()

# This is a fixed detail of the v68 architecture.
HVX_VECTOR_BYTES = 128

# NOTE on server ports:
# These tests use different port numbers for the RPC server (7070 + ...).
# The reason is that an RPC session cannot be gracefully closed without
# triggering TIME_WAIT state on the server socket. This prevents another
# server to bind to the same port until the wait time elapses.

_BT = bu.BenchmarksTable()

_CSV_COLUMN_ORDER = [
    # Identifies which TE-compute / TIRScript is used as the basis for the
    # benchmarked primfunc. Only needs to be meaningful to humans.
    "basic_kernel",
    # The tensors' element type
    "dtype",
    # When applicable, indicates the particular variation of schedules
    # apply by the Python code. Decoding this may require looking at this
    # script's source code.
    "sched_type",
    # The memory location of the tensors used during the execution of
    # the primfunc.  We currently assume just one location.
    # This will likely need to be generalized as we add more sophisticated
    # primfuncs.
    "mem_scope",
    # For primfuncs that treat tensor buffers as collections of 1D vectors,
    # this is the number of vectors in each tensor.
    # This will likely need to be generalized as we add more sophisticated
    # primfuncs.
    "num_vectors_per_tensor",
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

_HOST_OUTPUT_DIR = tempfile.mkdtemp()

_PRIMFUNC_NAME = "elemwise_add"

print("-" * 80)
print("OUTPUT DIRECTORY: {}".format(_HOST_OUTPUT_DIR))
print("-" * 80)
print()


def _get_irmod_elemwise_add(shape: list, dtype: str, mem_scope: str) -> tvm.ir.module.IRModule:
    """
    Return an IRModule containing a single primfunc, expressed as NS-TIR.

    The primfunc implements elementwise-add. Its signature is (A,B,C), where
    A and B are the input tensors, and C is the output tensor.
    All three tensors have the specfied shape, dtype, and mem_scope.

    If the specified primfunc is known to be unsupported, raise an UnsupportedExcetion.
    """
    assert len(shape) == 2

    # TVMScript can reference simple Python variables, but it doesn't
    # curently support more complex Python expressions...
    (
        dim0_size,
        dim1_size,
    ) = shape

    if mem_scope == "global.vtcm":
        raise bu.UnsupportedException("This benchmark kernel does not yet support VTCM buffers.")

        # This check is currently elided by the one above, but it should become relevant as soon
        # as we add VTCM support to this kernel generator.
        #
        # Also: The VTCM budget is a very rough estimate, based only on experience.
        # Assuming that it's even reasonable to use a hard-coded estimate AT ALL, this number
        # may need tweaking.

        # The below code is commented is commented to avoid unreachable error
        # with pylint. Please enable this once the kernel starts supporting
        # VTCM buffers

        # Code starts below:
        # ---- ------ -----
        # estimated_vtcm_budget_bytes = HVX_VECTOR_BYTES * 1024

        # dtype_bits = tvm._ffi.runtime_ctypes.DataType(dtype).bits
        # assert dtype_bits % 8 == 0
        # dtype_bytes = dtype_bits // 8

        # num_vtcm_tensors = 3
        # estimated_vtcm_needed_bytes = shape[0] * shape[1] * dtype_bytes * num_vtcm_tensors

        # if estimated_vtcm_needed_bytes > estimated_vtcm_budget_bytes:
        #     raise bu.UnsupportedException("Expect to exceed VTCM budget.")

    @tvm.script.ir_module
    class BenchmarkModule:
        """Elementwise STIR module for benchmarking"""

        # pylint: disable=no-self-argument,invalid-name,missing-function-docstring
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            # We exchange data between function by handles, which are similar to pointer.
            T.func_attr({"global_symbol": "main", "tir.noalias": True})

            A = T.match_buffer(a, shape, dtype=dtype)
            B = T.match_buffer(b, shape, dtype=dtype)
            C = T.match_buffer(c, shape, dtype=dtype)

            for i in range(dim0_size):
                for j in range(dim1_size):
                    C[i, j] = A[i, j] + B[i, j]

        # pylint: enable=no-self-argument,invalid-name,missing-function-docstring

    return BenchmarkModule


def _benchmark_hexagon_elementwise_add_kernel(
    hexagon_session: Session, shape: list, dtype: str, mem_scope: str
):
    """
    Generate and benchmark a single elementwise-add kernel for Hexagon.

    Produce these outputs:
      - Printed status updates / results to stdout and/or stderr.

      - Create a new subdirectory under _HOST_OUTPUT_DIR, and populate it with
        various logs and intermediate files.

      - Add to _BT a row describing this benchmark run.
    """
    # Represent the benchmark details in a form required by the benchmark table
    # and for other logging...
    keys_dict = {
        "basic_kernel": "ewise-add",
        "dtype": dtype,
        "shape": shape,
        "mem_scope": mem_scope,
    }

    desc = bu.get_benchmark_decription(keys_dict)

    # Create the host-side directory for this benchmark run's files / logs...
    host_files_dir_name = bu.get_benchmark_id(keys_dict)
    host_files_dir_path = os.path.join(_HOST_OUTPUT_DIR, host_files_dir_name)
    os.mkdir(host_files_dir_path)

    keys_dict["host_files_dir_path"] = host_files_dir_path

    log_file_path = os.path.join(host_files_dir_path, "out.txt")
    with open(log_file_path, "w", encoding="UTF-8") as log_file:
        print(f"CONFIGURATION: {desc}")
        log_file.write(f"CONFIGURATION: {desc}\n")

        try:
            ns_tir_module = _get_irmod_elemwise_add(shape, dtype, mem_scope)

            # Dump the primfunc NS-TIR (as text) to the log file...
            lowered_mod = tvm.lower(ns_tir_module, _PRIMFUNC_NAME)
            log_file.write("LOWERED IR MODULE:\n")
            log_file.write(str(lowered_mod))
            log_file.write("\n")

            # Lower the primfunc's IRModule to Hexagon object code...
            input1 = tvm.te.placeholder(shape, dtype=dtype)
            input2 = tvm.te.placeholder(shape, dtype=dtype)
            output = tvm.te.placeholder(shape, dtype=dtype)

            built_module: tvm.driver.build_module.OperatorModule = tvm.build(
                ns_tir_module,
                [
                    input1,
                    input2,
                    output,
                ],
                get_hexagon_target("v69"),
                name=_PRIMFUNC_NAME,
            )

            # Create an actual Hexagon-native shared object file, initially stored on the
            # host's file system...
            host_dso_binary_path = os.path.join(host_files_dir_path, "test_binary.so")
            built_module.save(host_dso_binary_path)
            print(f"SAVED BINARY TO HOST PATH: {host_dso_binary_path}")

            # Upload the .so to the Android device's file system (or wherever is appropriate
            # when using the Hexagon simulator)...
            target_dso_binary_filename = "test_binary.so"
            target_dso_binary_pathname = hexagon_session.upload(
                host_dso_binary_path, target_dso_binary_filename
            )

            # Generate our testing / validation data...
            (
                host_numpy_input1_data,
                host_numpy_input2_data,
                host_numpy_output_data_expected,
            ) = _get_elemwise_add_reference_value_tensors(shape, dtype)

            # On the target device / simulator, make our Hexagon-native shared object
            # available for use...
            loaded_hexagon_module: tvm.runtime.module.Module = hexagon_session.load_module(
                target_dso_binary_pathname
            )

            # Create the target-side tensors to hold the primfunc's inputs and outputs...
            input1_data = tvm.nd.empty(shape, dtype, hexagon_session.device, mem_scope)
            input2_data = tvm.nd.empty(shape, dtype, hexagon_session.device, mem_scope)
            output_data = tvm.nd.empty(shape, dtype, hexagon_session.device, mem_scope)

            # Populate the primfunc's input tensors...
            input1_data.copyfrom(host_numpy_input1_data)
            input2_data.copyfrom(host_numpy_input2_data)

            # Actually benchmark the primfunc...
            timer = loaded_hexagon_module.time_evaluator(
                "main", hexagon_session.device, number=10, repeat=1
            )
            timing_result = timer(input1_data, input2_data, output_data)

            print(f"TIMING RESULT: {timing_result}")
            log_file.write(f"TIMING RESULT: {timing_result}\n")

            # Verify that the computation actually happened, and produced the correct result.
            result = output_data.numpy()

            if dtype == "float16":
                # These are the closest tolerance we currently expect / require for these
                # kernels.  They may be changed in the future.
                rel_tolerance = 0.005
                abs_tolerance = 2.0
            elif dtype == "int8":
                rel_tolerance = 0
                abs_tolerance = 0
            else:
                raise Exception(f"Unexpected dtype: {dtype}")

            # TODO: We're assuming that *any* assertion thrown by 'assert_allclose' is because
            # the numerical differences were too large.  But ideally this code would
            # differentiate between (a) numerical difference errors, which should simply be
            # recorded as a failed benchmark run, vs. (b) more serious errors that should
            # kill the overall script.
            try:
                tvm.testing.assert_allclose(
                    result, host_numpy_output_data_expected, rel_tolerance, abs_tolerance
                )
            except AssertionError as err:
                raise bu.NumericalAccuracyException(str(err))

            _BT.record_success(timing_result, **keys_dict)

        except bu.NumericalAccuracyException as err:
            print()
            print("FAIL: Numerical accuracy error. See log file.")

            log_file.write("\n")
            log_file.write(f"FAIL: {err}\n")

            _BT.record_fail(**keys_dict, comments="Numerical accuracy error. See log file.")

        except bu.UnsupportedException as err:
            print()
            print(f"SKIP: {err}")

            log_file.write("\n")
            log_file.write(f"SKIP: {err}\n")

            _BT.record_skip(**keys_dict, comments=f"Unsupported configuration: {err}")


def _get_elemwise_add_reference_value_tensors(shape: list, dtype: str):
    """
    Return [A:np.array, B:np.array, C:np.array]

    `A`, `B`, and `C` are reference data used to exercise and validate
    an elementwise-add kernel: C = A+B.

    NOTE: These data are primarily meant for performance testing.
    The values may be helpful in detecting correctness issues, but that's
    a secondary consideration here.
    """
    assert len(shape) == 2

    input1 = np.ndarray(shape, dtype=dtype)
    input2 = np.ndarray(shape, dtype=dtype)

    np_dtype = input1.dtype

    if np_dtype.kind in ["i", "u"]:
        # We allow overflow for integer types because it tends to be well-behaved
        # and well-understood...
        min_value = np.iinfo(np_dtype).min
        max_value = np.iinfo(np_dtype).max

        next_value = min_value

        for i in range(shape[0]):
            for j in range(shape[1]):
                input1[i, j] = next_value
                input2[i, j] = next_value * 2
                next_value += 1

    elif np_dtype.kind == "f":
        # NOTE: For simplicity, we avoid test data that that require
        # well-defined behavior on floating-point overflow.
        # But it may be reasonable to test that in the future.
        min_value = np.finfo(np_dtype).min
        max_value = np.finfo(np_dtype).max

        min_input_value = min_value / 2.0 + 1
        max_input_value = max_value / 2.0 - 2
        delta = (max_input_value - min_input_value) / (shape[0] * shape[1])

        next_value = min_input_value

        for i in range(shape[0]):
            for j in range(shape[1]):
                input1[i, j] = next_value
                input2[i, j] = next_value + 1
                next_value += delta

    else:
        assert False, f"Unexpected data type: {np_dtype}"

    output = input1 + input2
    return [
        input1,
        input2,
        output,
    ]


@pytest.mark.skipif(_SHOULD_SKIP_BENCHMARKS, reason=_SKIP_BENCHMARKS_REASON)
@tvm.testing.requires_hexagon
def test_elemwise_add(hexagon_session: Session):
    """Main elementwise add test function"""
    for dtype in [
        "int8",
        "float16",
    ]:

        for mem_scope in [
            "global",
            "global.vtcm",
        ]:

            # These numbers are fairly arbitrary, but they're meant to stress memory/caches to
            # various extents.
            for num_vectors_per_tensor in [
                1,
                16,
                64,
                512,
                2048,
            ]:

                dtype_bits = tvm._ffi.runtime_ctypes.DataType(dtype).bits
                assert dtype_bits % 8 == 0
                dtype_bytes = dtype_bits // 8

                elem_per_hvx_vector = HVX_VECTOR_BYTES // dtype_bytes

                shape = [
                    num_vectors_per_tensor,
                    elem_per_hvx_vector,
                ]

                print()
                _benchmark_hexagon_elementwise_add_kernel(hexagon_session, shape, dtype, mem_scope)

    print("-" * 80)
    print(f"OUTPUT DIRECTORY: {_HOST_OUTPUT_DIR}")
    print("-" * 80)
    print()

    tabular_output_filename = os.path.join(_HOST_OUTPUT_DIR, "benchmark-results.csv")
    with open(tabular_output_filename, "w", encoding="UTF-8") as csv_file:
        _BT.print_csv(csv_file, _CSV_COLUMN_ORDER)

    print(f"BENCHMARK RESULTS FILE: {tabular_output_filename}")

    _BT.print_csv(sys.stdout, _CSV_COLUMN_ORDER)

    if _BT.has_fail() > 0:
        pytest.fail("At least one benchmark configuration failed", pytrace=False)


if __name__ == "__main__":
    tvm.testing.main()
