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
"""A context manager that injects the best tuning record in the database into compilation"""
import logging
from typing import List, Optional, Union

from tvm._ffi import register_object
from tvm.ir import IRModule
from tvm.runtime import Object
from tvm.target import Target

from . import _ffi_api
from .database import Database
from .utils import make_logging_func

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@register_object("meta_schedule.ApplyHistoryBest")
class ApplyHistoryBest(Object):
    """An integration context that allows application of historically best records from a database

    Parameters
    ----------
    database : Database
        The database to be queried from
    """

    database: Database

    def __init__(self, database: Database) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.ApplyHistoryBest, database, make_logging_func(logger)  # type: ignore # pylint: disable=no-member
        )

    def query(
        self,
        task_name: str,
        mod: IRModule,
        target: Target,
        dispatched: Optional[List[IRModule]],
    ) -> Union[IRModule, None]:
        """The entry point of the integration

        Parameters
        ----------
        task_name : str
            The name of the task extracted
        mod : IRModule
            The high-level IR
        target: Target
            Target Info
        dispatched : Optional[List[IRModule]]
            A list of low-level IRs that the high-level IR could potentially dispatch to

        Returns
        -------
        result : IRModule or None
            Currently we only have to return tir::PrimFunc, but we wrap it under IRModule for
            more general future use. None is returned if there is no feedback hint.
        """
        return _ffi_api.ApplyHistoryBestQuery(  # type: ignore # pylint: disable=no-member
            self,
            task_name,
            mod,
            target,
            dispatched,
        )

    @staticmethod
    def current() -> Optional["ApplyHistoryBest"]:
        """The context manager in the current scope

        Returns
        -------
        ctx : Optional[ApplyHistoryBest]
            The ApplyHistoryBest context manager in the current scope.
            None if it's currently not under any ApplyHistoryBest context.
        """
        return _ffi_api.ApplyHistoryBestCurrent()  # type: ignore # pylint: disable=no-member

    def __enter__(self) -> "ApplyHistoryBest":
        """Entering the scope of the context manager"""
        _ffi_api.ApplyHistoryBestEnterScope(self)  # type: ignore # pylint: disable=no-member
        return self

    def __exit__(self, ptype, value, trace) -> None:
        """Exiting the scope of the context manager"""
        _ffi_api.ApplyHistoryBestExitScope(self)  # type: ignore # pylint: disable=no-member
