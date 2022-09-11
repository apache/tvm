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
"""Test space definition primitives"""

from tvm import te
from tvm.autotvm.task.space import ConfigSpace, FallbackConfigEntity


def gemm_func(cfg, N, filter_y=None, filter_x=None):
    A = te.placeholder((N, N), name="A")
    B = te.placeholder((N, N), name="B")

    k = te.reduce_axis((0, N), name="k")
    C = te.compute((N, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=[k]), name="C")

    s = te.create_schedule([C.op])

    y, x = s[C].op.axis

    cfg.define_split("tile_y", cfg.axis(y), num_outputs=2, filter=filter_y)
    cfg.define_split("tile_x", cfg.axis(x), num_outputs=2, filter=filter_x)

    return s, [A, B, C]


def test_split():
    cfg = ConfigSpace()

    gemm_func(cfg, 128)
    assert cfg.range_length == 64
    assert len(cfg.space_map["tile_y"]) == 8

    # test policy
    cfg = ConfigSpace()
    cfg.define_split("tile_x", cfg.axis(256), policy="factors", num_outputs=3)
    assert len(cfg.space_map["tile_x"]) == 45

    cfg.define_split("tile_y", cfg.axis(256), policy="power2", num_outputs=3)
    assert len(cfg.space_map["tile_y"]) == 45

    cfg.define_split("tile_z", cfg.axis(256), policy="verbose", num_outputs=3)
    assert len(cfg.space_map["tile_z"]) == 45

    cfg.define_split("tile_a", cfg.axis(224), policy="factors", num_outputs=3)
    assert len(cfg.space_map["tile_a"]) == 63

    cfg.define_split("tile_b", cfg.axis(224), policy="power2", num_outputs=3)
    assert len(cfg.space_map["tile_b"]) == 36

    cfg.define_split("tile_c", cfg.axis(224), policy="verbose", num_outputs=3)
    assert len(cfg.space_map["tile_c"]) == 84

    # Count the number of non-negative integer solutions of a + b + c + d = n
    def count4(n):
        cnt = 0
        for a in range(0, n + 1):
            for b in range(0, n - a + 1):
                cnt += n - a - b + 1
        return cnt

    # test overflow
    n = 25
    cfg = ConfigSpace()
    cfg.define_split("x", cfg.axis(2**n), policy="factors", num_outputs=4)
    # count4(25) is 3276.
    assert len(cfg.space_map["x"]) == count4(n)

    # test fallback
    cfg = FallbackConfigEntity()
    cfg.define_split("tile_n", cfg.axis(128), num_outputs=3)
    cfg.fallback_split("tile_n", [-1, 8, 4])
    # verify if define_split override previously manualy defined split params
    cfg.define_split("tile_n", cfg.axis(128), num_outputs=3)
    assert cfg["tile_n"].size == [4, 8, 4]

    cfg = FallbackConfigEntity()
    cfg.define_split("tile_n", cfg.axis(49), num_outputs=3)
    cfg.fallback_split("tile_n", [-1, 8, 4])
    assert cfg["tile_n"].size == [7, 7, 1]

    cfg = FallbackConfigEntity()
    cfg.define_split("tile_n", cfg.axis(49), num_outputs=3)
    try:
        cfg.fallback_split("tile_n", [-1, 1, 0])
        assert False
    except RuntimeError:
        pass


def _raises_exception(f):
    try:
        f()
    except Exception:
        return True
    return False


def test_multi_filter():
    # create config without multi_filter
    cfg = ConfigSpace()
    gemm_func(cfg, 128)
    # create config with multi_filter
    cfg_mf = ConfigSpace()
    gemm_func(cfg_mf, 128)
    cfg_mf.multi_filter(
        filter=lambda entity: 32 <= (entity["tile_x"].size[1] * entity["tile_y"].size[1]) < 1024
    )
    # test len
    assert len(cfg) == 64
    assert len(cfg_mf) == 34
    # test range_length
    assert cfg.range_length == 64
    assert cfg_mf.range_length == 64
    # test dims
    assert cfg.dims == [8, 8]
    assert cfg_mf.dims == [8, 8]
    # test is_index_valid
    assert cfg.is_index_valid(0) is True
    assert cfg.is_index_valid(15) is True
    assert cfg_mf.is_index_valid(0) is False
    assert cfg_mf.is_index_valid(15) is True
    # test get
    assert _raises_exception(lambda: cfg.get(0)) is False
    assert _raises_exception(lambda: cfg.get(15)) is False
    assert _raises_exception(lambda: cfg_mf.get(0)) is True
    assert _raises_exception(lambda: cfg_mf.get(15)) is False
    # test subrange_length
    assert cfg.subrange_length(0, 64) == 64
    assert cfg.subrange_length(0, 32) == 32
    assert cfg.subrange_length(16, 32) == 16
    assert cfg.subrange_length(16, 16) == 0
    assert _raises_exception(lambda: cfg.subrange_length(0, 128))
    assert _raises_exception(lambda: cfg.subrange_length(-64, 64))
    assert _raises_exception(lambda: cfg.subrange_length(64, 0))
    assert cfg_mf.subrange_length(0, 64) == 34
    assert cfg_mf.subrange_length(0, 32) == 17
    assert cfg_mf.subrange_length(16, 32) == 10
    assert cfg_mf.subrange_length(16, 16) == 0
    assert _raises_exception(lambda: cfg_mf.subrange_length(0, 128))
    assert _raises_exception(lambda: cfg_mf.subrange_length(-64, 64))
    assert _raises_exception(lambda: cfg_mf.subrange_length(64, 0))
    # test point2knob
    assert cfg.point2knob(0) == [0, 0]
    assert cfg.point2knob(4) == [4, 0]
    assert cfg.point2knob(8) == [0, 1]
    assert cfg.point2knob(12) == [4, 1]
    assert cfg_mf.point2knob(0) == [0, 0]
    assert cfg_mf.point2knob(4) == [4, 0]
    assert cfg_mf.point2knob(8) == [0, 1]
    assert cfg_mf.point2knob(12) == [4, 1]
    # test knob2point
    assert cfg.knob2point([0, 0]) == 0
    assert cfg.knob2point([4, 0]) == 4
    assert cfg.knob2point([0, 1]) == 8
    assert cfg.knob2point([4, 1]) == 12
    assert cfg_mf.knob2point([0, 0]) == 0
    assert cfg_mf.knob2point([4, 0]) == 4
    assert cfg_mf.knob2point([0, 1]) == 8
    assert cfg_mf.knob2point([4, 1]) == 12
    # get_rand_index
    cfg_valid_indexes = list(filter(lambda idx: cfg.is_index_valid(idx), range(cfg.range_length)))
    assert cfg.get_rand_index() in cfg_valid_indexes
    assert cfg.get_rand_index(start=15, end=16) == 15
    assert 10 <= cfg.get_rand_index(start=10, end=20) < 20
    assert cfg.get_rand_index(to_exclude=cfg_valid_indexes[:-1]) == cfg_valid_indexes[-1:][0]
    cfg_mf_valid_indexes = list(
        filter(lambda idx: cfg_mf.is_index_valid(idx), range(cfg_mf.range_length))
    )
    assert cfg_mf.get_rand_index() in cfg_mf_valid_indexes
    assert cfg_mf.get_rand_index(start=15, end=16) == 15
    assert 10 <= cfg_mf.get_rand_index(start=10, end=20) < 20
    assert (
        cfg_mf.get_rand_index(to_exclude=cfg_mf_valid_indexes[:-1]) == cfg_mf_valid_indexes[-1:][0]
    )
    # get_next_index
    assert cfg.get_next_index(0) == 1
    assert cfg.get_next_index(0, 1) == 1
    assert cfg.get_next_index(0, 2) == 2
    assert cfg.get_next_index(0, -1) is None
    assert cfg.get_next_index(0, -2) is None
    assert cfg.get_next_index(63) is None
    assert cfg.get_next_index(63, 1) is None
    assert cfg.get_next_index(63, 2) is None
    assert cfg.get_next_index(63, -1) == 62
    assert cfg.get_next_index(63, -2) == 61
    assert cfg.get_next_index(60, 1, end=63) == 61
    assert cfg.get_next_index(63, -1, start=60) == 62
    assert cfg_mf.get_next_index(0) == 5
    assert cfg_mf.get_next_index(0, 1) == 5
    assert cfg_mf.get_next_index(0, 2) == 6
    assert cfg_mf.get_next_index(0, -1) is None
    assert cfg_mf.get_next_index(0, -2) is None
    assert cfg_mf.get_next_index(63) is None
    assert cfg_mf.get_next_index(63, 1) is None
    assert cfg_mf.get_next_index(63, 2) is None
    assert cfg_mf.get_next_index(63, -1) == 58
    assert cfg_mf.get_next_index(63, -2) == 57
    assert cfg_mf.get_next_index(60, 1, end=63) is None
    assert cfg_mf.get_next_index(63, -1, start=60) is None
    # test sample_ints
    cfg_ints = cfg.sample_ints(5)
    assert len(cfg_ints) == 5
    assert set(cfg_ints).issubset(cfg_valid_indexes)
    cfg_mf_ints = cfg_mf.sample_ints(5)
    assert len(cfg_mf_ints) == 5
    assert set(cfg_mf_ints).issubset(cfg_mf_valid_indexes)
    # test random_walk
    cfg_walk = cfg.random_walk(15)
    assert cfg_walk != 15
    assert cfg_walk in cfg_valid_indexes
    cfg_mf_walk = cfg_mf.random_walk(15)
    assert cfg_mf_walk != 15
    assert cfg_mf_walk in cfg_mf_valid_indexes


def test_filter_and_multi_filter():
    # test the order: filter -> multi_filter
    cfg = ConfigSpace()
    gemm_func(cfg, 128, filter_y=lambda y: y.size[-1] < 64)
    # after adding filter
    assert len(cfg) == 48
    assert cfg.range_length == 48
    cfg.multi_filter(
        filter=lambda entity: 32 <= (entity["tile_x"].size[1] * entity["tile_y"].size[1]) < 1024
    )
    # after adding multi_filter
    assert len(cfg) == 27
    assert cfg.range_length == 48

    # test the order: multi_filter -> filter
    cfg = ConfigSpace()
    s, (A, B, C) = gemm_func(cfg, 128, filter_y=None)
    cfg.multi_filter(
        filter=lambda entity: 32 <= (entity["tile_x"].size[1] * entity["tile_y"].size[1]) < 1024
    )
    # after adding multi_filter
    assert len(cfg) == 34
    assert cfg.range_length == 64
    y, x = s[C].op.axis
    cfg.define_split("tile_y", cfg.axis(y), num_outputs=2, filter=lambda y: y.size[-1] < 64)
    # after adding filter
    assert len(cfg) == 27
    assert cfg.range_length == 48


if __name__ == "__main__":
    test_split()
    test_multi_filter()
    test_filter_and_multi_filter()
