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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
"""Test Meta Schedule Database"""
import os.path as osp
import sys
import tempfile
from typing import Callable

import pytest

import tvm
from tvm import tir
from tvm.ir.module import IRModule
from tvm.meta_schedule.arg_info import ArgInfo
from tvm.meta_schedule.database import JSONDatabase, TuningRecord
from tvm.script import tir as T
from tvm.tir import Schedule

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument
# fmt: off
@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.ir_module
class MatmulRelu:
    @T.prim_func
    def main(a: T.handle, b: T.handle, d: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (16, 16), "float32")
        B = T.match_buffer(b, (16, 16), "float32")
        D = T.match_buffer(d, (16, 16), "float32")
        C = T.alloc_buffer((16, 16), "float32")
        for i, j, k in T.grid(16, 16, 16):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(16, 16):
            with T.block("relu"):
                vi, vj = T.axis.remap("SS", [i, j])
                D[vi, vj] = T.max(C[vi, vj], 0.0)


# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def _schedule_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    i_tiles = [1, 1, 2, 512]
    j_tiles = [1, 512, 1, 2]
    k_tiles = [256, 4]
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
    k_0, k_1 = sch.split(loop=k, factors=k_tiles)
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def _create_schedule(mod: IRModule, sch_fn: Callable[[Schedule], None]) -> Schedule:
    sch = tir.Schedule(mod=mod, debug_mask="all")
    sch_fn(sch)
    return sch


def _create_tmp_database(tmpdir: str) -> JSONDatabase:
    path_workload = osp.join(tmpdir, "workloads.json")
    path_tuning_record = osp.join(tmpdir, "tuning_records.json")
    return JSONDatabase(path_workload, path_tuning_record)


def _equal_record(a: TuningRecord, b: TuningRecord):
    assert str(a.trace) == str(b.trace)
    assert str(a.run_secs) == str(b.run_secs)
    # AWAIT(@zxybazh): change to export after fixing "(bool)0"
    assert str(a.target) == str(b.target)
    assert tvm.ir.structural_equal(a.workload.mod, b.workload.mod)
    for arg0, arg1 in zip(a.args_info, b.args_info):
        assert str(arg0.as_json()) == str(arg1.as_json())


def test_meta_schedule_tuning_record_round_trip():
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        workload = database.commit_workload(mod)
        record = TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            [1.5, 2.5, 1.8],
            workload,
            tvm.target.Target("llvm"),
            ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
        )
        database.commit_tuning_record(record)
        new_record = TuningRecord.from_json(record.as_json(), workload)
        _equal_record(record, new_record)


def test_meta_schedule_database_create():
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        assert osp.exists(database.path_workload)
        assert osp.exists(database.path_tuning_record)


def test_meta_schedule_database_add_entry():
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        workload = database.commit_workload(mod)
        record = TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            [1.5, 2.5, 1.8],
            workload,
            tvm.target.Target("llvm"),
            ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
        )
        database.commit_tuning_record(record)
        assert len(database) == 1
        (ret,) = database.get_top_k(workload, 3)
        _equal_record(ret, record)


def test_meta_schedule_database_missing():
    mod: IRModule = Matmul
    mod_2: IRModule = MatmulRelu
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        workload = database.commit_workload(mod)
        workload_2 = database.commit_workload(mod_2)
        record = TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            [1.5, 2.5, 1.8],
            workload,
            tvm.target.Target("llvm"),
            ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
        )
        database.commit_tuning_record(record)
        ret = database.get_top_k(workload_2, 3)
        assert len(ret) == 0


def test_meta_schedule_database_sorting():
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        token = database.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        records = [
            TuningRecord(
                trace,
                [7.0, 8.0, 9.0],
                token,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
            TuningRecord(
                trace,
                [1.0, 2.0, 3.0],
                token,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
            TuningRecord(
                trace,
                [4.0, 5.0, 6.0],
                token,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
            TuningRecord(
                trace,
                [1.1, 1.2, 600.0],
                token,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
            TuningRecord(
                trace,
                [1.0, 100.0, 6.0],
                token,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
            TuningRecord(
                trace,
                [4.0, 9.0, 8.0],
                token,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
        ]
        for record in records:
            database.commit_tuning_record(record)
        ret = database.get_top_k(token, 2)
        assert len(ret) == 2
        try:
            _equal_record(ret[0], records[2])
            _equal_record(ret[1], records[1])
        except AssertionError:
            _equal_record(ret[0], records[1])
            _equal_record(ret[1], records[2])


def test_meta_schedule_database_reload():
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        token = database.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        records = [
            TuningRecord(
                trace,
                [7.0, 8.0, 9.0],
                token,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
            TuningRecord(
                trace,
                [1.0, 2.0, 3.0],
                token,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
            TuningRecord(
                trace,
                [4.0, 5.0, 6.0],
                token,
                tvm.target.Target("llvm"),
                ArgInfo.from_prim_func(func=mod["main"]),  # pylint: disable=unsubscriptable-object
            ),
        ]
        for record in records:
            database.commit_tuning_record(record)
        new_database = JSONDatabase(  # pylint: disable=unused-variable
            path_workload=database.path_workload,
            path_tuning_record=database.path_tuning_record,
        )
        token = new_database.commit_workload(mod)
        ret = new_database.get_top_k(token, 2)
        assert len(ret) == 2
        try:
            _equal_record(ret[0], records[2])
            _equal_record(ret[1], records[1])
        except AssertionError:
            _equal_record(ret[0], records[1])
            _equal_record(ret[1], records[2])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
