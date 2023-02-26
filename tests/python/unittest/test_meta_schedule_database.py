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
import tempfile
from typing import Callable, List, Optional

import pytest
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm import relay, tir
from tvm.ir.module import IRModule
from tvm.meta_schedule.database import TuningRecord, Workload
from tvm.script import tir as T
from tvm.target import Target
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


def _create_tmp_database(tmpdir: str, mod_eq: str = "structural") -> ms.database.JSONDatabase:
    path_workload = osp.join(tmpdir, "workloads.json")
    path_tuning_record = osp.join(tmpdir, "tuning_records.json")
    return ms.database.JSONDatabase(path_workload, path_tuning_record, module_equality=mod_eq)


def _equal_record(a: ms.database.TuningRecord, b: ms.database.TuningRecord):
    assert str(a.trace) == str(b.trace)
    assert str(a.run_secs) == str(b.run_secs)
    # AWAIT(@zxybazh): change to export after fixing "(bool)0"
    assert str(a.target) == str(b.target)
    assert tvm.ir.structural_equal(a.workload.mod, b.workload.mod)
    for arg0, arg1 in zip(a.args_info, b.args_info):
        assert str(arg0.as_json()) == str(arg1.as_json())


@ms.utils.derived_object
class PyMemoryDatabaseDefault(ms.database.PyDatabase):
    def __init__(self):
        super().__init__()
        self.tuning_records_: List[TuningRecord] = []
        self.workloads_: List[Workload] = []

    def has_workload(self, mod: IRModule) -> bool:
        for workload in self.workloads_:
            if tvm.ir.structural_equal(mod, workload.mod):
                return True

    def commit_workload(self, mod: IRModule) -> ms.database.Workload:
        if self.has_workload(mod):
            for workload in self.workloads_:
                if tvm.ir.structural_equal(mod, workload.mod):
                    return workload
        else:
            workload = ms.database.Workload(mod)
            self.workloads_.append(workload)
            return workload

    def commit_tuning_record(self, record: TuningRecord) -> None:
        self.tuning_records_.append(record)

    def get_all_tuning_records(self) -> List[TuningRecord]:
        return self.tuning_records_

    def get_top_k(self, workload: ms.database.Workload, top_k: int) -> List[TuningRecord]:
        return sorted(
            list(
                filter(
                    lambda x: tvm.ir.structural_equal(workload.mod, x.workload.mod),
                    self.tuning_records_,
                )
            ),
            key=lambda x: sum(x.run_secs) / len(x.run_secs) if x.run_secs else 1e9,
        )[:top_k]

    def __len__(self) -> int:
        return len(self.tuning_records_)


@ms.utils.derived_object
class PyMemoryDatabaseOverride(ms.database.PyDatabase):
    def __init__(self):
        super().__init__()
        self.tuning_records_: List[TuningRecord] = []
        self.workloads_: List[Workload] = []

    def has_workload(self, mod: IRModule) -> bool:
        for workload in self.workloads_:
            if tvm.ir.structural_equal(mod, workload.mod):
                return True

    def commit_workload(self, mod: IRModule) -> ms.database.Workload:
        if self.has_workload(mod):
            for workload in self.workloads_:
                if tvm.ir.structural_equal(mod, workload.mod):
                    return workload
        else:
            workload = ms.database.Workload(mod)
            self.workloads_.append(workload)
            return workload

    def commit_tuning_record(self, record: TuningRecord) -> None:
        self.tuning_records_.append(record)

    def get_all_tuning_records(self) -> List[TuningRecord]:
        return self.tuning_records_

    def get_top_k(self, workload: ms.database.Workload, top_k: int) -> List[TuningRecord]:
        return sorted(
            list(
                filter(
                    lambda x: tvm.ir.structural_equal(workload.mod, x.workload.mod),
                    self.tuning_records_,
                )
            ),
            key=lambda x: sum(x.run_secs) / len(x.run_secs) if x.run_secs else 1e9,
        )[:top_k]

    def __len__(self) -> int:
        return len(self.tuning_records_)

    def query_tuning_record(
        self, mod: IRModule, target: Target, workload_name: Optional[str] = None
    ) -> Optional[TuningRecord]:
        if self.has_workload(mod):
            records = self.get_top_k(self.commit_workload(mod), 2)
            if len(records) == 1:
                return records[0]
            elif len(records) == 2:
                return records[1]  # return the 2nd best if there are two records
        return None

    def query_schedule(
        self, mod: IRModule, target: Target, workload_name: Optional[str] = None
    ) -> Optional[Schedule]:
        record = self.query_tuning_record(mod, target, workload_name)
        if record is not None:
            sch = Schedule(record.workload.mod)
            record.trace.apply_to_schedule(sch, remove_postproc=False)
            return sch
        return None

    def query_ir_module(
        self, mod: IRModule, target: Target, workload_name: Optional[str] = None
    ) -> Optional[IRModule]:
        record = self.query_tuning_record(mod, target, workload_name)
        if record is not None:
            sch = Schedule(record.workload.mod)
            record.trace.apply_to_schedule(sch, remove_postproc=False)
            return sch.mod
        return None


def test_meta_schedule_tuning_record_round_trip():
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        workload = database.commit_workload(mod)
        record = ms.database.TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            workload,
            [1.5, 2.5, 1.8],
            tvm.target.Target("llvm"),
            ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
        )
        database.commit_tuning_record(record)
        new_record = ms.database.TuningRecord.from_json(record.as_json(), workload)
        _equal_record(record, new_record)


def test_meta_schedule_database_create():
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        assert osp.exists(database.path_workload)
        assert osp.exists(database.path_tuning_record)


def test_meta_schedule_database_has_workload():
    mod: IRModule = Matmul
    missing_mod: IRModule = MatmulRelu
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        workload = database.commit_workload(mod)
        record = ms.database.TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            workload,
            [1.5, 2.5, 1.8],
            tvm.target.Target("llvm"),
            ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
        )
        database.commit_tuning_record(record)
        assert len(database) == 1
        assert database.has_workload(mod)
        assert not database.has_workload(missing_mod)


def test_meta_schedule_database_add_entry():
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        workload = database.commit_workload(mod)
        record = ms.database.TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            workload,
            [1.5, 2.5, 1.8],
            tvm.target.Target("llvm"),
            ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
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
        record = ms.database.TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            workload,
            [1.5, 2.5, 1.8],
            tvm.target.Target("llvm"),
            ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
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
            ms.database.TuningRecord(
                trace,
                token,
                [7.0, 8.0, 9.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [1.0, 2.0, 3.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [4.0, 5.0, 6.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [1.1, 1.2, 600.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [1.0, 100.0, 6.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [4.0, 9.0, 8.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
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
            ms.database.TuningRecord(
                trace,
                token,
                [7.0, 8.0, 9.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [1.0, 2.0, 3.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [4.0, 5.0, 6.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
        ]
        for record in records:
            database.commit_tuning_record(record)
        new_database = ms.database.JSONDatabase(
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


def test_meta_schedule_database_union():
    mod: IRModule = Matmul
    target = tvm.target.Target("llvm")
    arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])
    db_1 = ms.database.MemoryDatabase()
    db_2 = ms.database.MemoryDatabase()
    trace = _create_schedule(mod, _schedule_matmul).trace

    def query(db):  # pylint: disable=invalid-name
        return db.query_tuning_record(mod=mod, target=target, workload_name="main").run_secs

    def commit_record(db, run_sec):  # pylint: disable=invalid-name
        db.commit_tuning_record(
            ms.database.TuningRecord(
                trace,
                workload=db.commit_workload(mod),
                run_secs=[run_sec],
                target=target,
                args_info=arg_info,
            )
        )

    commit_record(db_1, 1.0)
    (run_sec,) = query(db_1)
    assert run_sec.value == 1.0

    commit_record(db_2, 0.5)
    (run_sec,) = query(db_2)
    assert run_sec.value == 0.5

    (run_secs,) = query(ms.database.UnionDatabase(db_1, db_2))
    assert run_secs.value == 0.5

    (run_secs,) = query(ms.database.OrderedUnionDatabase(db_1, db_2))
    assert run_secs.value == 1.0


def test_meta_schedule_pydatabase_default_query():

    mod: IRModule = Matmul
    target = tvm.target.Target("llvm")
    arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])
    db = PyMemoryDatabaseDefault()  # pylint: disable=invalid-name
    sch = _create_schedule(mod, _schedule_matmul)
    trace = sch.trace

    def query(db, mod, target, kind):  # pylint: disable=invalid-name
        return db.query(mod=mod, target=target, workload_name="main", kind=kind)

    def commit_record(trace, db, run_sec):  # pylint: disable=invalid-name
        db.commit_tuning_record(
            ms.database.TuningRecord(
                trace,
                workload=db.commit_workload(mod),
                run_secs=[run_sec],
                target=target,
                args_info=arg_info,
            )
        )

    commit_record(trace, db, 1.0)
    record = query(db, mod, target, "record")
    assert record is not None and record.run_secs[0].value == 1.0
    sch_res = query(db, mod, target, "schedule")
    assert sch_res is not None and tvm.ir.structural_equal(sch_res.mod, sch.mod)
    mod_res = query(db, mod, target, "ir_module")
    assert mod_res is not None and tvm.ir.structural_equal(mod_res, sch.mod)

    commit_record(Schedule(mod).trace, db, 0.2)  # Empty Trace
    record = query(db, mod, target, "record")
    assert record is not None and record.run_secs[0].value == 0.2
    sch_res = query(db, mod, target, "schedule")
    assert sch_res is not None and tvm.ir.structural_equal(sch_res.mod, mod)
    mod_res = query(db, mod, target, "ir_module")
    assert mod_res is not None and tvm.ir.structural_equal(mod_res, mod)


def test_meta_schedule_pydatabase_override_query():

    mod: IRModule = Matmul
    target = tvm.target.Target("llvm")
    arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])
    db = PyMemoryDatabaseOverride()  # pylint: disable=invalid-name
    sch = _create_schedule(mod, _schedule_matmul)
    trace = sch.trace

    def query(db, mod, target, kind):  # pylint: disable=invalid-name
        return db.query(mod=mod, target=target, workload_name="main", kind=kind)

    def commit_record(trace, db, run_sec):  # pylint: disable=invalid-name
        db.commit_tuning_record(
            ms.database.TuningRecord(
                trace,
                workload=db.commit_workload(mod),
                run_secs=[run_sec],
                target=target,
                args_info=arg_info,
            )
        )

    commit_record(trace, db, 1.14)
    record = query(db, mod, target, "record")
    assert record is not None and record.run_secs[0].value == 1.14
    sch_res = query(db, mod, target, "schedule")
    assert sch_res is not None and tvm.ir.structural_equal(sch_res.mod, sch.mod)
    mod_res = query(db, mod, target, "ir_module")
    assert mod_res is not None and tvm.ir.structural_equal(mod_res, sch.mod)

    commit_record(Schedule(mod).trace, db, 0.514)  # Empty Trace
    record = query(db, mod, target, "record")
    assert record is not None and record.run_secs[0].value == 1.14  # Override to 2nd best
    sch_res = query(db, mod, target, "schedule")
    assert sch_res is not None and tvm.ir.structural_equal(sch_res.mod, sch.mod)
    mod_res = query(db, mod, target, "ir_module")
    assert mod_res is not None and tvm.ir.structural_equal(mod_res, sch.mod)


def test_meta_schedule_pydatabase_current():
    db = PyMemoryDatabaseDefault()  # pylint: disable=invalid-name
    with db:  # pylint: disable=not-context-manager
        assert ms.database.Database.current() == db


def call_get_top_k(run_secs_list, database, k):
    mod: IRModule = Matmul
    workload = database.commit_workload(mod)
    for run_secs in run_secs_list:
        record = ms.database.TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            workload,
            run_secs,
            tvm.target.Target("llvm"),
            ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
        )
        database.commit_tuning_record(record)
    return [[v.value for v in record.run_secs] for record in database.get_top_k(workload, k)]


@pytest.mark.parametrize(
    "k,expected",
    [
        (0, []),
        (1, [[0.0, 2.0]]),
        (4, [[0.0, 2.0], [2.0], [1.5, 4.5], [3.0, 1e10]]),
        (5, [[0.0, 2.0], [2.0], [1.5, 4.5], [3.0, 1e10]]),
    ],
)
def test_memory_database_get_top_k(k, expected):
    run_secs_list = [[1.5, 4.5], [], [0.0, 2.0], None, [2.0], [3.0, 1e10], [1e10]]
    database = ms.database.MemoryDatabase()
    result = call_get_top_k(run_secs_list, database, k)
    assert result == expected


@pytest.mark.parametrize(
    "k,expected",
    [
        (0, []),
        (4, [[0.0, 2.0], [2.0], [1.5, 4.5], [3.0, 1e10]]),
        (5, [[0.0, 2.0], [2.0], [1.5, 4.5], [3.0, 1e10]]),
    ],
)
def test_json_database_get_top_k(k, expected):
    run_secs_list = [[1.5, 4.5], [], [0.0, 2.0], None, [2.0], [3.0, 1e10], [1e10]]
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        result = call_get_top_k(run_secs_list, database, k)
    assert result == expected


def MatmulFunc() -> IRModule:
    a = relay.var("a", relay.TensorType((1024, 1024), "float32"))
    b = relay.var("b", relay.TensorType((1024, 1024), "float32"))
    func = relay.Function([a, b], relay.nn.matmul(a, b))
    return tvm.IRModule.from_expr(func)


def MatmulPrimFunc() -> IRModule:
    return Matmul


@pytest.mark.parametrize("f_mod", [MatmulPrimFunc, MatmulFunc])
@pytest.mark.parametrize("mod_eq", ["structural", "ignore-ndarray", "anchor-block"])
def test_json_database_commit_workload(f_mod, mod_eq):
    mod: IRModule = f_mod()
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir, mod_eq)
        database.commit_workload(mod)


@pytest.mark.parametrize("f_mod", [MatmulPrimFunc, MatmulFunc])
@pytest.mark.parametrize("mod_eq", ["structural", "ignore-ndarray", "anchor-block"])
def test_memory_database_commit_workload(f_mod, mod_eq):
    mod: IRModule = f_mod()
    database = ms.database.MemoryDatabase(module_equality=mod_eq)
    database.commit_workload(mod)


if __name__ == "__main__":
    tvm.testing.main()
