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
"""A database consists of multiple databases."""
from tvm._ffi import register_object

from .. import _ffi_api
from .database import Database


@register_object("meta_schedule.UnionDatabase")
class UnionDatabase(Database):
    """A database composed of multiple databases, allowing users to guide IR rewriting using
    combined knowledge of those databases. To each query, it returns the best record among all the
    databases given.

    Examples
    --------
    Examples below demonstrate the usecases of and difference between UnionDatabase and
    OrderDatabase.

    Assumption:
    * db1, db2 do not have tuning records for the target workload.
    * Each of db3, db4, db5 has tuning records r3, r4, r5 for target workload respectively.

    .. code-block:: python

    #### Case 1. `UnionDatabase`:
    merged_db = ms.database.UnionDatabase(
        db1, # no record
        db2, # no record
        db3, # has r3
        db4  # has r4
    )
    # returns the better one between r3 and r4
    merged_db.query_tuning_record(..., target_workload)

    ### Case 2. `OrderedUnionDatabase`
    merged_db = ms.database.OrderedUnionDatabase(
        db1, # no record
        db2, # no record
        db3, # has r3
        db4  # has r4
    )
    # returns r3
    merged_db.query_tuning_record(..., target_workload)

    ### Case 3. Mix-use scenario
    merged_db = ms.database.UnionDatabase(
        db1, # no record
        db2, # no record
        db3, # has r3
        ms.database.OrderedUnionDatabase( # returns r4
            db4,  # has r4
            db5,  # has r5
        )
    )
    # returns the better one between r3 and r4
    merged_db.query_tuning_record(..., target_workload)

    ### Case 4. Another mix-use scenario
    merged_db = ms.database.UnionDatabase(
        db1, # no record
        db2, # no record
        db3, # has r3
        ms.database.UnionDatabase( # returns best one between r4 and r5
            db4,  # has r4
            db5,  # has r5
        )
    )
    # returns the best one among r3, r4 and r5
    merged_db.query_tuning_record(..., target_workload)

    ### Case 5. Yet another mix-use scenario
    merged_db = ms.database.OrderedUnionDatabase(
        db1, # no record
        db2, # no record
        ms.database.UnionDatabase( # returns best one between r3 and r4
            db3, # has r3
            db4,  # has r4
        )
        db5,  # has r5
    )
    # returns the better one between r3 and r4
    merged_db.query_tuning_record(..., target_workload)
    """

    def __init__(self, *databases: Database) -> None:
        """Construct a merged database from multiple databases.

        Parameters
        ----------
        *databases : Database
            The list of databases to combine.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.DatabaseUnionDatabase,  # type: ignore # pylint: disable=no-member
            databases,
        )
