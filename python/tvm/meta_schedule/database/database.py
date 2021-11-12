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
"""Tuning record database"""
from typing import Any, List

from tvm._ffi import register_object
from tvm.ir.module import IRModule
from tvm.runtime import Object
from tvm.target import Target
from tvm.tir.schedule import Trace

from .. import _ffi_api
from ..arg_info import ArgInfo
from ..utils import _json_de_tvm, check_override


@register_object("meta_schedule.Workload")
class Workload(Object):
    """A workload, i.e. an IRModule and its structural hash.

    Parameters
    ----------
    mod : IRModule
        The workload's IRModule
    """

    mod: IRModule

    def __init__(self, mod: IRModule) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.Workload,  # type: ignore # pylint: disable=no-member
            mod,
        )

    def as_json(self) -> Any:
        """Export the workload to a JSON string.

        Returns
        -------
        json_str : str
            The JSON string exported.
        """
        return _json_de_tvm(_ffi_api.WorkloadAsJSON(self))  # type: ignore # pylint: disable=no-member

    @staticmethod
    def from_json(json_obj: Any) -> "Workload":
        """Create a workload from a json object.

        Parameters
        ----------
        json_obj : Any
            The json object to parse.

        Returns
        -------
        tuning_record : TuningRecord
            The parsed tuning record.
        """
        return _ffi_api.WorkloadFromJSON(json_obj)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.TuningRecord")
class TuningRecord(Object):
    """The class of tuning records.

    Parameters
    ----------
    trace : tvm.ir.Trace
        The trace of the tuning record.
    run_secs : List[float]
        The run time of the tuning record.
    workload : Workload
        The workload of the tuning record.
    target : Target
        The target of the tuning record.
    args_info : List[ArgInfo]
        The argument information of the tuning record.
    """

    trace: Trace
    run_secs: List[float]
    workload: Workload
    target: Target
    args_info: List[ArgInfo]

    def __init__(
        self,
        trace: Trace,
        run_secs: List[float],
        workload: Workload,
        target: Target,
        args_info: List[ArgInfo],
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.TuningRecord,  # type: ignore # pylint: disable=no-member
            trace,
            run_secs,
            workload,
            target,
            args_info,
        )

    def as_json(self) -> Any:
        """Export the tuning record to a JSON string.

        Returns
        -------
        json_str : str
            The JSON string exported.
        """
        return _json_de_tvm(_ffi_api.TuningRecordAsJSON(self))  # type: ignore # pylint: disable=no-member

    @staticmethod
    def from_json(json_obj: Any, workload: Workload) -> "TuningRecord":
        """Create a tuning record from a json object.

        Parameters
        ----------
        json_obj : Any
            The json object to parse.
        workload : Workload
            The workload.

        Returns
        -------
        tuning_record : TuningRecord
            The parsed tuning record.
        """
        return _ffi_api.TuningRecordFromJSON(json_obj, workload)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.Database")
class Database(Object):
    """The abstract database interface."""

    def commit_workload(self, mod: IRModule) -> Workload:
        """Commit a workload to the database if missing.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be searched for or added.

        Returns
        -------
        workload : Workload
            The workload corresponding to the given IRModule.
        """
        return _ffi_api.DatabaseCommitWorkload(self, mod)  # type: ignore # pylint: disable=no-member

    def commit_tuning_record(self, record: TuningRecord) -> None:
        """Commit a tuning record to the database.

        Parameters
        ----------
        record : TuningRecord
            The tuning record to add.
        """
        _ffi_api.DatabaseCommitTuningRecord(self, record)  # type: ignore # pylint: disable=no-member

    def get_top_k(self, workload: Workload, top_k: int) -> List[TuningRecord]:
        """Get the top K tuning records of given workload from the database.

        Parameters
        ----------
        workload : Workload
            The workload to be searched for.
        top_k : int
            The number of top records to get.

        Returns
        -------
        top_k_records : List[TuningRecord]
            The top K records.
        """
        return _ffi_api.DatabaseGetTopK(self, workload, top_k)  # type: ignore # pylint: disable=no-member

    def __len__(self) -> int:
        """Get the number of records in the database.

        Returns
        -------
        num_records : int
            The number of records in the database
        """
        return _ffi_api.DatabaseSize(self)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.PyDatabase")
class PyDatabase(Database):
    """An abstract Database with customized methods on the python-side."""

    def __init__(self):
        """Constructor."""

        @check_override(self.__class__, Database)
        def f_commit_workload(mod: IRModule) -> Workload:
            return self.commit_workload(mod)

        @check_override(self.__class__, Database)
        def f_commit_tuning_record(record: TuningRecord) -> None:
            self.commit_tuning_record(record)

        @check_override(self.__class__, Database)
        def f_get_top_k(workload: Workload, top_k: int) -> List[TuningRecord]:
            return self.get_top_k(workload, top_k)

        @check_override(self.__class__, Database, func_name="__len__")
        def f_size() -> int:
            return len(self)

        self.__init_handle_by_constructor__(
            _ffi_api.DatabasePyDatabase,  # type: ignore  # pylint: disable=no-member
            f_commit_workload,
            f_commit_tuning_record,
            f_get_top_k,
            f_size,
        )
