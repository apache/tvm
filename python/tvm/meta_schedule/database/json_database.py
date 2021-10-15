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
"""The default database that uses a JSON File to store tuning records"""
from tvm._ffi import register_object

from .. import _ffi_api
from .database import Database


@register_object("meta_schedule.JSONDatabase")
class JSONDatabase(Database):
    """The class of tuning records.

    Parameters
    ----------
    path_workload : str
        The path to the workload table.
    path_tuning_record : str
        The path to the tuning record table.
    """

    path_workload: str
    path_tuning_record: str

    def __init__(
        self,
        path_workload: str,
        path_tuning_record: str,
        allow_missing: bool = True,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        path_workload : str
            The path to the workload table.
        path_tuning_record : str
            The path to the tuning record table.
        allow_missing : bool
            Whether to create new file when the given path is not found.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.DatabaseJSONDatabase,  # type: ignore # pylint: disable=no-member
            path_workload,
            path_tuning_record,
            allow_missing,
        )
