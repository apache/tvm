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
"""Relax Tuning Pass API default functions"""
from typing import List, Optional
import logging

from tvm.runtime import Object
from tvm.ir.module import IRModule
from tvm.meta_schedule.utils import _json_de_tvm
from tvm.meta_schedule.database import Workload
from tvm.tir.schedule.trace import JSON_TYPE
from tvm.target import Target
from tvm._ffi import register_object
from .primitives import Trace
from . import _ffi_api

logger = logging.getLogger("TuningAPI")  # pylint: disable=invalid-name


@register_object("relax.tuning_api.TuningRecord")
class TuningRecord(Object):
    """The class of tuning records.

    Parameters
    ----------
    trace : tvm.relax.transform.tuning_api.Trace
        The trace of the tuning record.
    run_secs : Optional[List[float]]
        The run-time of the tuning record.
    """

    trace: Trace
    run_secs: Optional[List[float]]

    def __init__(  # type: ignore # pylint: disable=too-many-arguments
        self,
        trace: Trace,
        run_secs: Optional[List[float]] = None,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.TuningRecord,  # type: ignore # pylint: disable=no-member
            trace,
            run_secs,
        )

    def as_json(self, include_irmod: bool = False) -> JSON_TYPE:
        """Export the tuning record to a JSON string.
        Parameters
        ----------
        include_irmod: bool
            Decides whether to serialize in_mod as well.

        Returns
        -------
        json_str : str
            The JSON string exported.
        """
        return _json_de_tvm(_ffi_api.TuningRecordAsJSON(self, include_irmod))  # type: ignore # pylint: disable=no-member

    @staticmethod
    def from_json(json_obj: JSON_TYPE) -> "TuningRecord":
        """Create a tuning record from a json object.

        Parameters
        ----------
        json_obj : JSON_TYPE
            The json object to parse.

        Returns
        -------
        tuning_record : TuningRecord
            The parsed tuning record.
        """
        return _ffi_api.TuningRecordFromJSON(json_obj)  # type: ignore # pylint: disable=no-member


@register_object("relax.tuning_api.Database")
class Database(Object):
    """The abstract database interface."""

    def has_workload(self, mod: IRModule) -> bool:
        """Check if the database has the given workload.
        Parameters
        ----------
        mod : IRModule
            The IRModule to be searched for.

        Returns
        -------
        result : bool
            Whether the given workload is committed.
        """
        return _ffi_api.DatabaseHasWorkload(self, mod)  # type: ignore # pylint: disable=no-member

    def has_measurement_record(self, workload: Workload, target: Target) -> bool:
        """Check if the database has a measurement record for the given workload and target pair.
        Parameters
        ----------
        workload: Workload
            The workload to be searched for.
        target: Target
            The target to be searched for.

        Returns
        -------
        result : bool
            Whether the given workload and target pair is committed for the measurement record.
        """
        return _ffi_api.DatabaseHasMeasurementRecord(self, workload, target)  # type: ignore # pylint: disable=no-member

    def has_tuning_record(self, workload: Workload, target: Target) -> bool:
        """Check if the database has a tuning record for the given workload and target pair.
        Parameters
        ----------
        workload: Workload
            The workload to be searched for.
        target: Target
            The target to be searched for.

        Returns
        -------
        result : bool
            Whether the given workload and target pair is committed for the tuning record.
        """
        return _ffi_api.DatabaseHasTuningRecord(self, workload, target)  # type: ignore # pylint: disable=no-member

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

    def commit_measurement_record(
        self, workload: Workload, target: Target, run_secs: List[float]
    ) -> None:
        """Commit a measurement record to the database.
        A pair of workload and target will be used as a key.

        Parameters
        ----------
        workload: Workload
            The workload to be searched for.
        target: Target
            The target to be searched for.
        run_secs : Optional[List[float]]
            The measurement record to add.
        """
        _ffi_api.DatabaseCommitMeasurementRecord(self, workload, target, run_secs)  # type: ignore # pylint: disable=no-member

    def commit_tuning_record(
        self, workload: Workload, target: Target, record: TuningRecord
    ) -> None:
        """Commit a tuning record to the database.
        A pair of workload and target will be used as a key.

        Parameters
        ----------
        workload: Workload
            The workload to be searched for.
        target: Target
            The target to be searched for.
        record : TuningRecord
            The tuning record to add.
        """
        _ffi_api.DatabaseCommitTuningRecord(self, workload, target, record)  # type: ignore # pylint: disable=no-member

    def get_measurement_record(self, workload: Workload, target: Target) -> Optional[List[float]]:
        """Get the measurement record of given workload and target from the database.

        Parameters
        ----------
        workload : Workload
            The workload to be searched for.
        target: Target
            The target to be searched for.

        Returns
        -------
        measurement_record : Optional[List[float]]
            Measurement record if exists.
        """
        return _ffi_api.DatabaseGetMeasurementRecord(self, workload, target)  # type: ignore # pylint: disable=no-member

    def get_top_k(self, workload: Workload, target: Target, top_k: int) -> List[TuningRecord]:
        """Get the top K tuning records of given workload and target from the database.

        Parameters
        ----------
        workload : Workload
            The workload to be searched for.
        target: Target
            The target to be searched for.
        top_k : int
            The number of top records to get.

        Returns
        -------
        top_k_records : List[TuningRecord]
            The top K records.
        """
        return _ffi_api.DatabaseGetTopK(self, workload, target, top_k)  # type: ignore # pylint: disable=no-member


@register_object("relax.tuning_api.JSONDatabase")
class JSONDatabase(Database):
    """The class of JSON database.

    Parameters
    ----------
    path_workload : str
        The path to the workload table.
    path_tuning_record : str
        The path to the tuning record table.
        Manages pairs of <Workload (in_mod), TuningRecord>
    path_measurement_record : str
        The path to the path_measurement_record table.
        Manages pairs of <Workload (out_mod), run_secs>
    """

    path_workload: str
    path_tuning_record: str
    path_measurement_record: str

    def __init__(
        self,
        path_workload: str,
        path_tuning_record: str,
        path_measurement_record: str,
        allow_missing: bool = True,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        path_workload : str
            The path to the workload table.
        path_tuning_record : str
            The path to the tuning record table.
        path_measurement_record : str
            The path to the path_measurement_record table.
        allow_missing : bool
            Whether to create new file when the given path is not found.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.DatabaseJSONDatabase,  # type: ignore # pylint: disable=no-member
            path_workload,
            path_tuning_record,
            path_measurement_record,
            allow_missing,
        )
