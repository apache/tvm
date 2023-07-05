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
import sys
import pytest
import random
import tvm
import tvm.testing
from tvm.tir.usmp.utils import BufferInfo
from tvm import WorkspacePoolInfo, PoolInfoProperties


def _check_max_workspace_size(buffer_pool_allocations, pool_info, size, tolerance=0):
    """Helper to check maximum allocated memory size"""
    max_workspace_size = 0
    for buffer_info, pool_allocation in buffer_pool_allocations.items():
        if pool_allocation.pool_info == pool_info:
            size_candidate = pool_allocation.byte_offset + buffer_info.size_bytes
            if size_candidate > max_workspace_size:
                max_workspace_size = size_candidate
    _diff = max_workspace_size.value - size
    return (
        (max_workspace_size.value == size if tolerance == 0 else tolerance > 100 * _diff / size),
        "'{}': expected {} got {}, diff {:0.2f}% ({} bytes)".format(
            pool_info.pool_name, size, max_workspace_size, 100 * _diff / size, _diff
        ),
    )


def _verify_conflicts(buffer_info, pool_allocation, buffer_info_map):
    """Helper to check expected liveness conflicts"""
    for conflict in buffer_info.conflicts:
        conflict_pool_allocation = buffer_info_map[conflict]

        if conflict_pool_allocation.pool_info == pool_allocation.pool_info:
            assert conflict_pool_allocation.byte_offset != pool_allocation.byte_offset
            l2 = max(
                conflict_pool_allocation.byte_offset + conflict.size_bytes,
                pool_allocation.byte_offset + buffer_info.size_bytes,
            ) - min(conflict_pool_allocation.byte_offset, pool_allocation.byte_offset)
            assert (
                conflict.size_bytes + buffer_info.size_bytes <= l2
            ), 'Conflicting: \n"{} @{}"\n"{} @{}"'.format(
                conflict, conflict_pool_allocation, buffer_info, pool_allocation
            )


def _verify_all_conflicts(buffer_pool_allocations):
    """Helper to verify liveness conflicts"""
    for buffer_info, pool_allocation in buffer_pool_allocations.items():
        _verify_conflicts(buffer_info, pool_allocation, buffer_pool_allocations)


def test_bounded(
    random_len=150,
    pools=[
        WorkspacePoolInfo("default", [], PoolInfoProperties(65535)),
        WorkspacePoolInfo("slow", []),
    ],
):
    """Tests two pools, one is bounded and one is not limited"""
    random.seed(0)
    mem_range = [BufferInfo(str(i), random.randrange(1, 65535), pools) for i in range(random_len)]
    for mr in mem_range:
        pr = random.choice(mem_range)
        while pr in (*mr.conflicts, mr):
            pr = random.choice(mem_range)

        mr.set_conflicts([*mr.conflicts, pr])
        pr.set_conflicts([*pr.conflicts, mr])

    fusmp_algo = tvm.get_global_func("tir.usmp.algo.hill_climb")
    result_map = fusmp_algo(mem_range, 0)
    _verify_all_conflicts(result_map)


def __test_data_alloc_max():
    """Test data"""
    intervals = [
        (0, 159, 2048),
        (0, 13, 7904),
        (4, 35, 16),
        (12, 17, 32768),
        (16, 21, 32768),
    ]
    return intervals


def __test_data_deep_speech():
    """Test data"""
    intervals = [
        (0, 159, 2048),
        (0, 151, 2048),
        (0, 13, 7904),
        (2, 49, 16),
        (4, 35, 16),
        (6, 21, 16),
        (12, 17, 32768),
        (16, 21, 32768),
        (20, 27, 32768),
        (26, 31, 32768),
        (30, 35, 32768),
        (34, 41, 32768),
        (40, 45, 32768),
        (44, 49, 32768),
        (48, 145, 32768),
        (54, 59, 2048),
        (58, 483, 4096),
        (60, 65, 2048),
        (64, 461, 4096),
        (66, 71, 2048),
        (70, 439, 4096),
        (72, 77, 2048),
        (76, 417, 4096),
        (78, 83, 2048),
        (82, 395, 4096),
        (84, 89, 2048),
        (88, 373, 4096),
        (90, 95, 2048),
        (94, 351, 4096),
        (96, 101, 2048),
        (100, 329, 4096),
        (102, 107, 2048),
        (106, 307, 4096),
        (108, 113, 2048),
        (112, 285, 4096),
        (114, 119, 2048),
        (118, 263, 4096),
        (120, 125, 2048),
        (124, 241, 4096),
        (126, 131, 2048),
        (130, 219, 4096),
        (132, 137, 2048),
        (136, 197, 4096),
        (138, 143, 2048),
        (142, 175, 4096),
        (144, 149, 2048),
        (148, 153, 4096),
        (152, 163, 8192),
        (154, 171, 2048),
        (156, 181, 2048),
        (160, 167, 2048),
        (162, 165, 2048),
        (168, 171, 2048),
        (170, 509, 2048),
        (174, 185, 8192),
        (176, 193, 2048),
        (178, 203, 2048),
        (182, 189, 2048),
        (184, 187, 2048),
        (190, 193, 2048),
        (192, 511, 2048),
        (196, 207, 8192),
        (198, 215, 2048),
        (200, 225, 2048),
        (204, 211, 2048),
        (206, 209, 2048),
        (212, 215, 2048),
        (214, 513, 2048),
        (218, 229, 8192),
        (220, 237, 2048),
        (222, 247, 2048),
        (226, 233, 2048),
        (228, 231, 2048),
        (234, 237, 2048),
        (236, 515, 2048),
        (240, 251, 8192),
        (242, 259, 2048),
        (244, 269, 2048),
        (248, 255, 2048),
        (250, 253, 2048),
        (256, 259, 2048),
        (258, 517, 2048),
        (262, 273, 8192),
        (264, 281, 2048),
        (266, 291, 2048),
        (270, 277, 2048),
        (272, 275, 2048),
        (278, 281, 2048),
        (280, 519, 2048),
        (284, 295, 8192),
        (286, 303, 2048),
        (288, 313, 2048),
        (292, 299, 2048),
        (294, 297, 2048),
        (300, 303, 2048),
        (302, 521, 2048),
        (306, 317, 8192),
        (308, 325, 2048),
        (310, 335, 2048),
        (314, 321, 2048),
        (316, 319, 2048),
        (322, 325, 2048),
        (324, 523, 2048),
        (328, 339, 8192),
        (330, 347, 2048),
        (332, 357, 2048),
        (336, 343, 2048),
        (338, 341, 2048),
        (344, 347, 2048),
        (346, 525, 2048),
        (350, 361, 8192),
        (352, 369, 2048),
        (354, 379, 2048),
        (358, 365, 2048),
        (360, 363, 2048),
        (366, 369, 2048),
        (368, 527, 2048),
        (372, 383, 8192),
        (374, 391, 2048),
        (376, 401, 2048),
        (380, 387, 2048),
        (382, 385, 2048),
        (388, 391, 2048),
        (390, 529, 2048),
        (394, 405, 8192),
        (396, 413, 2048),
        (398, 423, 2048),
        (402, 409, 2048),
        (404, 407, 2048),
        (410, 413, 2048),
        (412, 531, 2048),
        (416, 427, 8192),
        (418, 435, 2048),
        (420, 445, 2048),
        (424, 431, 2048),
        (426, 429, 2048),
        (432, 435, 2048),
        (434, 533, 2048),
        (438, 449, 8192),
        (440, 457, 2048),
        (442, 467, 2048),
        (446, 453, 2048),
        (448, 451, 2048),
        (454, 457, 2048),
        (456, 535, 2048),
        (460, 471, 8192),
        (462, 479, 2048),
        (464, 489, 2048),
        (468, 475, 2048),
        (470, 473, 2048),
        (476, 479, 2048),
        (478, 537, 2048),
        (482, 493, 8192),
        (484, 501, 2048),
        (486, 497, 2048),
        (490, 497, 2048),
        (492, 495, 2048),
        (496, 626, 2048),
        (498, 501, 2048),
        (500, 626, 2048),
        (504, 549, 16),
        (508, 543, 32768),
        (542, 549, 32768),
        (548, 555, 32768),
        (554, 563, 464),
        (560, 563, 256),
        (562, 617, 2048),
        (564, 567, 1856),
        (566, 573, 1024),
        (568, 619, 1024),
        (570, 573, 1024),
        (572, 577, 1024),
        (576, 579, 1024),
        (578, 605, 1024),
        (580, 593, 1024),
        (584, 587, 1024),
        (586, 603, 1024),
        (594, 597, 1024),
        (596, 613, 1024),
        (604, 607, 1024),
        (606, 617, 1024),
        (616, 621, 2048),
        (618, 621, 1024),
        (620, 626, 464),
    ]
    return intervals


def __test_data_five():
    """Test data"""
    return [
        (4, 5, 95),
        (1, 4, 52135),
        (3, 4, 12136),
        (3, 5, 62099),
        (4, 5, 50458),
    ]


def __test_data_simple():
    """Test data"""
    return [
        (0, 23, 131072),  # 0
        (4, 5, 65568),  # 1
        (4, 9, 8192),  # 2
        (8, 30, 15360),  # 3
        (10, 11, 65568),  # 4
        (10, 15, 4096),  # 5
        (16, 17, 65552),  # 6
        (16, 21, 2048),  # 7
        (22, 23, 32784),  # 8
        (22, 27, 1024),  # 9
    ]


def find_maximum_from_intervals(intervals):
    """Expected list of intervals of (start, end, size)"""
    sorted_list = sorted(intervals, key=lambda _: _[0])
    max_mem = 0
    for t in range(sorted_list[0][0], sorted_list[-1][1] + 1):
        max_mem = max(
            max_mem, sum([size for (start, end, size) in sorted_list if t >= start and t <= end])
        )
    return max_mem


@pytest.mark.parametrize(
    "intervals",
    [__test_data_alloc_max(), __test_data_simple(), __test_data_deep_speech(), __test_data_five()],
)
def test_intervals(intervals):
    """Tests supplied intervals"""
    random.seed(0)
    result = run_intervals(intervals, 5)
    assert result["tir.usmp.algo.hill_climb"] == True, f" {result}"


def generate_range(sz, max_segment_sz=65535):
    """Helper func to generate list of size sz of ranges of random size max_segment_sz"""
    for i in range(0, sz):
        start = random.randrange(i, sz)
        stop = random.randrange(start + 1, start + 2 + ((sz - start) // 2))
        assert stop - start > 0
        yield (start, stop, random.randrange(1, max_segment_sz))


def test_random_intervals(interval_len=16):
    """Tests randomly generated interval of length interval_len"""
    random.seed(0)
    intervals = list(generate_range(interval_len))
    return run_intervals(intervals)


def run_intervals(intervals, tolerance=0):
    """Helper to run intervals"""
    expected_mem = find_maximum_from_intervals(intervals)
    pools = [WorkspacePoolInfo("default", [])]
    buffers = []
    # populate
    for i, (start, stop, size) in enumerate(intervals):
        buf = BufferInfo(str(i), size, pools)
        # buf.set_pool_candidates( ["default"] )
        buffers.append(buf)

    # intersect
    for i, (i_start, i_stop, _) in enumerate(intervals):
        conflicts = set()
        for j, (j_start, j_stop, _) in enumerate(intervals):
            start = min(i_start, j_start)
            stop = max(i_stop, j_stop)
            i_dur = i_stop - i_start + 1
            j_dur = j_stop - j_start + 1

            if i != j and (stop - start + 1 < i_dur + j_dur):
                conflicts.add(buffers[j])

        buffers[i].set_conflicts([c for c in sorted(conflicts, key=lambda c: c.name_hint)])

    result = {}
    for (alg, params) in [
        ("tir.usmp.algo.hill_climb", (expected_mem,)),
        ("tir.usmp.algo.greedy_by_size", (expected_mem,)),
    ]:
        fusmp_algo = tvm.get_global_func(alg)
        print("\n", "started", alg)
        buffer_info_arr = fusmp_algo(buffers, *params)
        print()

        _verify_all_conflicts(buffer_info_arr)
        result[alg], msg = _check_max_workspace_size(
            buffer_info_arr, pools[0], expected_mem, tolerance
        )
        if not result[alg]:
            print(alg, msg)

    return result


if __name__ == "__main__":
    tvm.testing.main()
