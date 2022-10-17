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
"""Replay Trace Search Strategy"""
from tvm._ffi import register_object

from .. import _ffi_api
from .search_strategy import SearchStrategy


@register_object("meta_schedule.ReplayFunc")
class ReplayFunc(SearchStrategy):
    """
    Replay Func Search Strategy is a search strategy that generates measure candidates by
    calling a design space generator and transform the design space.

    Parameters
    ----------
    num_trials_per_iter : int
        Number of trials per iteration.
    max_trials_per_task : int
        Total number of trials for one task
    """

    def __init__(self):
        """Constructor"""
        self.__init_handle_by_constructor__(
            _ffi_api.SearchStrategyReplayFunc,  # type: ignore # pylint: disable=no-member
        )
