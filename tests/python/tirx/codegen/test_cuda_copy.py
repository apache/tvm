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
"""Tests for T.cuda.copy_128b / copy_64b / copy_32b / copy_16b / copy_8b intrinsics."""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T

pytestmark = tvm.testing.requires_cuda.marks()

DEV = tvm.cuda(0)
TARGET = tvm.target.Target("cuda")


def _build_and_run(func, *np_args):
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=TARGET, tir_pipeline="tirx")
    rt_args = [tvm.runtime.tensor(a, device=DEV) for a in np_args]
    mod(*rt_args)
    return (*tuple(a.numpy() for a in rt_args), mod)


def test_copy_128b():
    """copy_128b: copies 16 bytes (4 float32 elements) via uint4 load/store."""

    # fmt: off
    @T.prim_func
    def func(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (4,), "float32")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane = T.lane_id([32])
        src_buf = T.alloc_buffer((4,), "float32", scope="shared")
        dst_buf = T.alloc_buffer((4,), "float32", scope="shared")
        if lane < 4:
            src_buf[lane] = T.float32(lane + 1)
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.copy_128b(dst_buf.ptr_to([0]), src_buf.ptr_to([0]))
        T.cuda.cta_sync()
        if lane < 4:
            out[lane] = dst_buf[lane]
        # fmt: on

    out_np = np.zeros(4, dtype="float32")
    result, mod = _build_and_run(func, out_np)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0])
    assert "tvm_builtin_copy_128b" in mod.mod.imports[0].inspect_source()


def test_copy_64b():
    """copy_64b: copies 8 bytes (2 float32 elements) via uint2 load/store."""

    # fmt: off
    @T.prim_func
    def func(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (2,), "float32")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane = T.lane_id([32])
        src_buf = T.alloc_buffer((2,), "float32", scope="shared")
        dst_buf = T.alloc_buffer((2,), "float32", scope="shared")
        if lane < 2:
            src_buf[lane] = T.float32(lane + 10)
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.copy_64b(dst_buf.ptr_to([0]), src_buf.ptr_to([0]))
        T.cuda.cta_sync()
        if lane < 2:
            out[lane] = dst_buf[lane]
        # fmt: on

    out_np = np.zeros(2, dtype="float32")
    result, mod = _build_and_run(func, out_np)
    np.testing.assert_allclose(result, [10.0, 11.0])
    assert "tvm_builtin_copy_64b" in mod.mod.imports[0].inspect_source()


def test_copy_32b():
    """copy_32b: copies 4 bytes (1 float32 element) via unsigned int load/store."""

    # fmt: off
    @T.prim_func
    def func(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (1,), "float32")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane = T.lane_id([32])
        src_buf = T.alloc_buffer((1,), "float32", scope="shared")
        dst_buf = T.alloc_buffer((1,), "float32", scope="shared")
        if lane == 0:
            src_buf[0] = T.float32(42)
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.copy_32b(dst_buf.ptr_to([0]), src_buf.ptr_to([0]))
        T.cuda.cta_sync()
        if lane == 0:
            out[0] = dst_buf[0]
        # fmt: on

    out_np = np.zeros(1, dtype="float32")
    result, mod = _build_and_run(func, out_np)
    np.testing.assert_allclose(result, [42.0])
    assert "tvm_builtin_copy_32b" in mod.mod.imports[0].inspect_source()


def test_copy_16b():
    """copy_16b: copies 2 bytes (1 float16 element) via unsigned short load/store."""

    # fmt: off
    @T.prim_func
    def func(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (1,), "float16")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane = T.lane_id([32])
        src_buf = T.alloc_buffer((1,), "float16", scope="shared")
        dst_buf = T.alloc_buffer((1,), "float16", scope="shared")
        if lane == 0:
            src_buf[0] = T.float16(7)
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.copy_16b(dst_buf.ptr_to([0]), src_buf.ptr_to([0]))
        T.cuda.cta_sync()
        if lane == 0:
            out[0] = dst_buf[0]
        # fmt: on

    out_np = np.zeros(1, dtype="float16")
    result, mod = _build_and_run(func, out_np)
    np.testing.assert_allclose(result, [7.0])
    assert "tvm_builtin_copy_16b" in mod.mod.imports[0].inspect_source()


def test_copy_8b():
    """copy_8b: copies 1 byte (1 uint8 element) via unsigned char load/store."""

    # fmt: off
    @T.prim_func
    def func(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (1,), "uint8")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane = T.lane_id([32])
        src_buf = T.alloc_buffer((1,), "uint8", scope="shared")
        dst_buf = T.alloc_buffer((1,), "uint8", scope="shared")
        if lane == 0:
            src_buf[0] = T.uint8(255)
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.copy_8b(dst_buf.ptr_to([0]), src_buf.ptr_to([0]))
        T.cuda.cta_sync()
        if lane == 0:
            out[0] = dst_buf[0]
        # fmt: on

    out_np = np.zeros(1, dtype="uint8")
    result, mod = _build_and_run(func, out_np)
    np.testing.assert_equal(result, np.array([255], dtype="uint8"))
    assert "tvm_builtin_copy_8b" in mod.mod.imports[0].inspect_source()


@pytest.mark.parametrize(
    "num_bytes,func_suffix", [(16, "128b"), (8, "64b"), (4, "32b"), (2, "16b"), (1, "8b")]
)
def test_codegen_function_names(num_bytes, func_suffix):
    """Verify each copy variant generates the expected C++ function name."""

    copy_fn = getattr(T.cuda, f"copy_{func_suffix}")

    # fmt: off
    @T.prim_func
    def func(dummy_ptr: T.handle):
        dummy = T.match_buffer(dummy_ptr, (16,), "uint8")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane = T.lane_id([32])
        a = T.alloc_buffer((16,), "uint8", scope="shared")
        b = T.alloc_buffer((16,), "uint8", scope="shared")
        if lane == 0:
            copy_fn(b.ptr_to([0]), a.ptr_to([0]))
            dummy[0] = T.uint8(0)
        # fmt: on

    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=TARGET, tir_pipeline="tirx")
    source = mod.mod.imports[0].inspect_source()
    assert f"tvm_builtin_copy_{func_suffix}" in source
