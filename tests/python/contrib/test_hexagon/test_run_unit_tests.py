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

# pylint: disable=redefined-outer-name

"""capture gtest output and return over FFI"""

import tvm
import tvm.testing
from tvm.contrib.hexagon.session import Session

unit_test_name = tvm.testing.parameter(
    "HexagonUserDMATest.wait",
    "HexagonUserDMATest.poll",
    "HexagonUserDMATest.bad_copy",
    "HexagonUserDMATest.sync_dma",
    "HexagonUserDMATest.async_dma_wait",
    "HexagonUserDMATest.async_dma_poll",
    "HexagonUserDMATest.pipeline",
    "HexagonUserDMATest.pipeline_write_queue",
    "HexagonUserDMATest.overflow_ring_buffer",
    "HexagonUserDMATest.sync_dma_bypass",
    "HexagonUserDMATest.sync_dma_bypass_vtcm_to_vtcm",
    "HexagonUserDMATest.sync_dma_bypass_",
    "HexagonBuffer.default_scope",
    "HexagonBuffer.ddr_scope",
    "HexagonBuffer.vtcm_scope",
    "HexagonBuffer.invalid_scope",
    "HexagonBuffer.micro_copies_corresponding_regions",
    "HexagonBuffer.micro_copies_src_bigger",
    "HexagonBuffer.micro_copies_dest_bigger",
    "HexagonBuffer.micro_copies_src_overlaps_dest_region",
    "HexagonBuffer.micro_copies_dest_overlaps_src_region",
    "HexagonBuffer.micro_copies_discontiguous_regions",
    "HexagonBuffer.micro_copies_invalid_size",
    "HexagonBuffer.macro_copies_adjacent_corresponding_regions_merged",
    "HexagonBuffer.macro_copies_discontiguous_regions_not_merged",
    "HexagonBuffer.macro_copies_overlapping_regions_merged",
    "HexagonBuffer.copy_from",
    "HexagonBuffer.copy_from_invalid_size",
    "HexagonBuffer.copy_from_smaller_size",
    "HexagonBuffer.nd",
    "HexagonBuffer.nd_copy_from",
    "HexagonBuffer.1d_copy_from_1d",
    "HexagonBuffer.2d_copy_from_1d",
    "HexagonBuffer.1d_copy_from_2d",
    "HexagonBuffer.nd_copy_from_nd_invalid_size",
    "HexagonBuffer.nd_copy_from_nd_smaller_size",
    "HexagonBuffer.md_copy_from_nd",
    "HexagonBuffer.copy_to",
    "HexagonBuffer.nd_copy_to",
    "RingBufferTest.zero_size_ring_buffer",
    "RingBufferTest.in_flight",
    "RingBufferTest.next",
    "RingBufferTest.full",
    "RingBufferTest.wrap",
    "RingBufferTest.wrap_corner",
    "RingBufferTest.half_in_flight",
    "RingBufferTest.half_in_flight_blocked",
    "QueuedRingBufferTest.invalid_queue",
    "QueuedRingBufferTest.two_queues",
    "QueuedRingBufferTest.group_end_before_group_start",
    "QueuedRingBufferTest.group_restart",
    "QueuedRingBufferTest.zero_size_group",
    "QueuedRingBufferTest.in_flight_before_group_end",
    "QueuedRingBufferTest.group_of_one",
    "QueuedRingBufferTest.group_of_two",
    "QueuedRingBufferTest.group_of_three",
    "QueuedRingBufferTest.two_groups_of_two",
    "QueuedRingBufferTest.two_queues_two_groups_of_two",
    "HexagonVtcmPoolTest.basic",
    "HexagonVtcmPoolTest.small_allocations",
    "HexagonVtcmPoolTest.no_free_vtcm",
    "HexagonVtcmPoolTest.not_enough_free_vtcm",
    "HexagonVtcmPoolTest.free_with_wrong_size",
    "HexagonVtcmPoolTest.free_alloc_combinations",
    "HexagonVtcmPoolTest.find_allocation",
    "HexagonVtcmPoolTest.find_smallest_allocation_combinations",
    "HexagonVtcmPoolTest.vtcm_alignment",
    "HexagonThreadManagerTest.ctor_edge_cases",
    "HexagonThreadManagerTest.init",
    "HexagonThreadManagerTest.dispatch",
    "HexagonThreadManagerTest.dispatch_wait",
    "HexagonThreadManagerTest.wait_signal",
    "HexagonThreadManagerTest.re_signal",
    "HexagonThreadManagerTest.re_wait",
    "HexagonThreadManagerTest.wait_signal_x2",
    "HexagonThreadManagerTest.signal_wait",
    "HexagonThreadManagerTest.sync_from_to",
    "HexagonThreadManagerTest.sync_from_to_self",
    "HexagonThreadManagerTest.sync_from_to_x2",
    "HexagonThreadManagerTest.sync_from_to_all",
    "HexagonThreadManagerTest.pipe_fill",
    "HexagonThreadManagerTest.pipe_overflow",
    "HexagonThreadManagerTest.producer_consumer",
    "HexagonThreadManagerTest.producer_consumer_signal_wait",
    "HexagonThreadManagerTest.thread_order",
    "HexagonThreadManagerTest.thread_order_signal_wait",
    "HexagonThreadManagerTest.dispatch_writes",
    "HexagonThreadManagerTest.threads_for_resource_types",
    "HexagonUtilsActivationsBlockizeTest.prepare_nhwc",
    "HexagonUtilsActivationsBlockizeTest.blockize_hwc_16b",
    "HexagonUtilsActivationsBlockizeTest.deblockize_hwc_16b",
    "HexagonUtilsWeightsChunkifyTest.calculate_num_weight_chunks",
    "HexagonUtilsWeightsChunkifyTest.prepare_hwio",
    "HexagonUtilsWeightsChunkifyTest.chunkify_hwio_16b",
    "HexagonUtilsQuantActivationsBlockizeTest.prepare_nhwc",
    "HexagonUtilsQuantActivationsBlockizeTest.blockize_hwc_8b",
    "HexagonUtilsQuantActivationsBlockizeTest.deblockize_hwc_8b",
    "HexagonUtilsQuantWeightsChunkifyTest.calculate_num_weight_chunks",
    "HexagonUtilsQuantWeightsChunkifyTest.prepare_hwio",
    "HexagonUtilsQuantWeightsChunkifyTest.chunkify_hwio_8b",
    "HexagonDeviceAPITest.global",
    "HexagonDeviceAPITest.alloc_free_cpu",
    "HexagonDeviceAPITest.alloc_free_hex",
    "HexagonDeviceAPITest.alloc_errors",
    "HexagonDeviceAPITest.free_errors",
    "HexagonDeviceAPITest.allocnd_free_cpu",
    "HexagonDeviceAPITest.allocnd_free_hex",
    "HexagonDeviceAPITest.allocnd_free_hex_vtcm",
    "HexagonDeviceAPITest.allocnd_erros",
    "HexagonDeviceAPITest.alloc_scalar",
    "HexagonDeviceAPITest.DISABLED_alloc_free_diff_dev",
    "HexagonDeviceAPITest.runtime_buffer_manager",
    "HexagonDeviceAPITest.thread_manager",
    "HexagonDeviceAPITest.user_dma",
    "HexagonDeviceAPITest.vtcm_pool",
)


# use pytest -sv to observe gtest output
# use --gtest_args to pass arguments to gtest
# for example to run all "foo" tests twice and observe gtest output run
# pytest -sv <this file> --gtests_args="--gtest_filter=*foo* --gtest_repeat=2"
@tvm.testing.requires_hexagon
def test_run_unit_tests(hexagon_session: Session, gtest_args, unit_test_name):
    """Try running gtest unit tests and capture output and error code"""
    try:
        func = hexagon_session._rpc.get_function("hexagon.run_unit_tests")
    except:
        print(
            (
                "This test requires TVM Runtime to be built with a Hexagon gtest"
                "version using Hexagon API cmake flag"
                "-DUSE_HEXAGON_GTEST=/path/to/hexagon/sdk/utils/googletest/gtest"
            )
        )
        raise

    # Prepend the unit test name, so command-line arguments still take
    # precedence, but CI runs each gtest as a separate pytest case.
    if gtest_args:
        gtest_args = f"--gtest_filter={unit_test_name} {gtest_args}"
    else:
        gtest_args = f"--gtest_filter={unit_test_name}"

    gtest_error_code_and_output = func(gtest_args)
    gtest_error_code = int(gtest_error_code_and_output.splitlines()[0])
    gtest_output = gtest_error_code_and_output.split("\n", 1)[-1]
    print(gtest_output)
    if gtest_error_code != 0:
        raise RuntimeError(
            f"Hexagon gtest retruned non-zero error code = {gtest_error_code}:\n{gtest_output}"
        )


if __name__ == "__main__":
    tvm.testing.main()
