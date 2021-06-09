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
# pylint: disable=missing-function-docstring,missing-module-docstring
import pytest
import tvm
from tvm import tir
from tvm.script import ty

# pylint: disable=no-member,invalid-name,unused-variable


@tvm.script.tir
def elementwise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.alloc_buffer((128, 128))
    C = tir.match_buffer(c, (128, 128))
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def elementwise_multi_producer_consumer(a: ty.handle, c: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.alloc_buffer((128, 128))
    C = tir.match_buffer(c, (128, 128))
    D = tir.match_buffer(d, (128, 128))
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0  # B has two consumers
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0
    with tir.block([128, 128], "D") as [vi, vj]:
        D[vi, vj] = B[vi, vj] + 2.0 + C[vi, vj]  # D has two producers


@tvm.script.tir
def elementwise_multi_consumer_inlined(a: ty.handle, c: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    D = tir.match_buffer(d, (128, 128))
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = A[vi, vj] * 2.0 + 1.0
    with tir.block([128, 128], "D") as [vi, vj]:
        D[vi, vj] = A[vi, vj] * 2.0 + 2.0 + C[vi, vj]


@tvm.script.tir
def elementwise_standalone(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.alloc_buffer((128, 128))
    C = tir.match_buffer(c, (128, 128))
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = A[vi, vj] + 1.0


@tvm.script.tir
def elementwise_standalone_dce(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = A[vi, vj] + 1.0


@tvm.script.tir
def elementwise_under_loop(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    B = tir.alloc_buffer((128, 128))
    for i in tir.serial(0, 128):
        for j in tir.serial(0, 128):
            with tir.block([128, 128], "B") as [vi, vj]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                B[vi, vj] = A[vi, vj] * 2.0
        for j in tir.serial(0, 128):
            with tir.block([128, 128], "C") as [vi, vj]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def elementwise_inlined(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = A[vi, vj] * 2.0 + 1.0


@tvm.script.tir
def fail_multi_reader_writer(a: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.alloc_buffer((128, 128))
    C = tir.alloc_buffer((128, 128))
    D = tir.match_buffer(d, (128, 128))
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
        C[vi, vj] = A[vi, vj] + 2.0
    with tir.block([128, 128], "C") as [vi, vj]:
        D[vi, vj] = B[vi, vj] + C[vi, vj]


@tvm.script.tir
def elementwise_multi_reverse_loads(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.alloc_buffer((128, 128))
    C = tir.match_buffer(c, (128, 128))
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = (B[vi, vj] + 1.0) * (B[vi, vj] * 2.0) + 3.0


@tvm.script.tir
def elementwise_multi_reverse_loads_inlined(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    with tir.block([128, 128], "B") as [vi, vj]:
        C[vi, vj] = (A[vi, vj] * 2.0 + 1.0) * (A[vi, vj] * 2.0 * 2.0) + 3.0


@tvm.script.tir
def opaque_access_load(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.alloc_buffer((128, 128))
    C = tir.match_buffer(c, (128, 128))
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
    with tir.block([128, 128], "C") as [vi, vj]:
        tir.reads(B[0:128, 0:128])
        tir.writes(C[0:128, 0:128])
        C[vi, vj] = tir.load("float32", B.data, vi * 128 + vj) + 1.0


@tvm.script.tir
def opaque_access_store(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.alloc_buffer((128, 128))
    C = tir.match_buffer(c, (128, 128))
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
    with tir.block([128, 128], "C") as [vi, vj]:
        tir.reads(B[0:128, 0:128])
        tir.writes(C[0:128, 0:128])
        tir.store(C.data, vi * 128 + vj, B[vi, vj] + 1.0)
        C[vi, vj] = tir.load("float32", B.data, vi * 16 + vj) + 1.0


@tvm.script.tir
def buffer_matched(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.alloc_buffer((128, 128))
    C = tir.match_buffer(c, (128, 128))
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
    with tir.block([128, 128], "C") as [vi, vj]:
        Bb = tir.match_buffer_region(B[vi : vi + 1, vj])
        C[vi, vj] = Bb[0, 0] + 1.0


@tvm.script.tir
def elementwise_predicate(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.alloc_buffer((128, 128))
    C = tir.match_buffer(c, (128, 128))
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "C") as [vi, vj]:
            tir.where(B[i, j] < 10.0)
            C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def elementwise_predicate_inlined(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "C") as [vi, vj]:
            tir.where(A[i, j] * 2.0 < 10.0)
            C[vi, vj] = A[vi, vj] * 2.0 + 1.0


@tvm.script.tir
def elementwise_multi_loads(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.alloc_buffer((128, 128))
    C = tir.match_buffer(c, (128, 128))
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
    with tir.block([128, 126], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + B[vi, vj + 1] + B[vi, vj + 2]


@tvm.script.tir
def elementwise_multi_loads_inlined(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    with tir.block([128, 126], "C") as [vi, vj]:
        C[vi, vj] = A[vi, vj] * 2.0 + A[vi, vj + 1] * 2.0 + A[vi, vj + 2] * 2.0


# pylint: enable=no-member,invalid-name,unused-variable


def test_compute_inline_elementwise():
    sch = tir.Schedule(elementwise, debug_mode=True)
    block_b = sch.get_block("B")
    block_c = sch.get_block("C")
    sch.compute_inline(block_b)
    tvm.ir.assert_structural_equal(elementwise_inlined, sch.mod["main"])
    assert sch.get(block_c).name_hint == "C"


def test_compute_inline_under_loop():
    sch = tir.Schedule(elementwise_under_loop, debug_mode=True)
    block_b = sch.get_block("B")
    block_c = sch.get_block("C")
    sch.compute_inline(block_b)
    tvm.ir.assert_structural_equal(elementwise_inlined, sch.mod["main"])
    assert sch.get(block_c).name_hint == "C"


def test_compute_inline_as_dce():
    sch = tir.Schedule(elementwise_standalone, debug_mode=True)
    block_b = sch.get_block("B")
    block_c = sch.get_block("C")
    sch.compute_inline(block_b)
    tvm.ir.assert_structural_equal(elementwise_standalone_dce, sch.mod["main"])
    assert sch.get(block_c).name_hint == "C"


def test_compute_inline_multi_consumer():
    sch = tir.Schedule(elementwise_multi_producer_consumer, debug_mode=True)
    block_b = sch.get_block("B")
    block_c = sch.get_block("C")
    block_d = sch.get_block("D")
    sch.compute_inline(block_b)
    tvm.ir.assert_structural_equal(elementwise_multi_consumer_inlined, sch.mod["main"])
    assert sch.get(block_c).name_hint == "C"
    assert sch.get(block_d).name_hint == "D"


def test_compute_inline_fail_multi_writer():
    sch = tir.Schedule(fail_multi_reader_writer, debug_mode=True, error_render_level="detail")
    block_b = sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.compute_inline(block_b)


def test_reverse_compute_inline_elementwise():
    sch = tir.Schedule(elementwise, debug_mode=True)
    block_b = sch.get_block("B")
    block_c = sch.get_block("C")
    sch.reverse_compute_inline(block_c)
    tvm.ir.assert_structural_equal(elementwise_inlined, sch.mod["main"])
    assert sch.get(block_b).name_hint == "B"


def test_reverse_compute_inline_under_loop():
    sch = tir.Schedule(elementwise_under_loop, debug_mode=True)
    block_b = sch.get_block("B")
    block_c = sch.get_block("C")
    sch.reverse_compute_inline(block_c)
    tvm.ir.assert_structural_equal(elementwise_inlined, sch.mod["main"])
    assert sch.get(block_b).name_hint == "B"


def test_reverse_compute_inline_fail_as_dce():
    sch = tir.Schedule(elementwise_standalone, debug_mode=True)
    block_b = sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reverse_compute_inline(block_b)


def test_reverse_compute_inline_fail_multi_producer():
    sch = tir.Schedule(elementwise_multi_producer_consumer, debug_mode=True)
    block_d = sch.get_block("D")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reverse_compute_inline(block_d)


def test_reverse_compute_inline_fail_multi_reader():
    sch = tir.Schedule(fail_multi_reader_writer, debug_mode=True)
    block_c = sch.get_block("C")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reverse_compute_inline(block_c)


def test_reverse_compute_multi_reverse_loads():
    sch = tir.Schedule(elementwise_multi_reverse_loads, debug_mode=True)
    block_c = sch.get_block("C")
    sch.reverse_compute_inline(block_c)
    tvm.ir.assert_structural_equal(elementwise_multi_reverse_loads_inlined, sch.mod["main"])


def test_reverse_compute_fail_multi_reverse_loads():
    sch = tir.Schedule(elementwise_multi_loads, debug_mode=True)
    block_c = sch.get_block("C")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reverse_compute_inline(block_c)


def test_opaque_access_load():
    sch = tir.Schedule(opaque_access_load, debug_mode=True)
    block_b = sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.compute_inline(block_b)


def test_opaque_access_store():
    sch = tir.Schedule(opaque_access_store, debug_mode=True)
    block_b = sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.compute_inline(block_b)


def test_buffer_matched():
    sch = tir.Schedule(buffer_matched, debug_mode=True)
    block_b = sch.get_block("B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.compute_inline(block_b)


def test_compute_inline_predicate():
    sch = tir.Schedule(elementwise_predicate, debug_mode=True)
    block_b = sch.get_block("B")
    sch.compute_inline(block_b)
    tvm.ir.assert_structural_equal(elementwise_predicate_inlined, sch.mod["main"])


def test_compute_inline_multi_loads():
    sch = tir.Schedule(elementwise_multi_loads, debug_mode=True)
    block_b = sch.get_block("B")
    sch.compute_inline(block_b)
    tvm.ir.assert_structural_equal(elementwise_multi_loads_inlined, sch.mod["main"])


if __name__ == "__main__":
    test_compute_inline_elementwise()
    test_compute_inline_under_loop()
    test_compute_inline_as_dce()
    test_compute_inline_multi_consumer()
    test_compute_inline_fail_multi_writer()
    test_reverse_compute_inline_elementwise()
    test_reverse_compute_inline_under_loop()
    test_reverse_compute_inline_fail_as_dce()
    test_reverse_compute_inline_fail_multi_producer()
    test_reverse_compute_inline_fail_multi_reader()
    test_reverse_compute_multi_reverse_loads()
    test_reverse_compute_fail_multi_reverse_loads()
    test_opaque_access_load()
    test_opaque_access_store()
    test_buffer_matched()
    test_compute_inline_predicate()
    test_compute_inline_multi_loads()
