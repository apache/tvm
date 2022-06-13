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
"""A context manager that profiles tuning time cost for different parts."""
from __future__ import annotations

import logging
from typing import Dict

from tvm._ffi import register_object
from tvm.runtime import Object

from . import _ffi_api

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TimeItContext:
    """The context to profile given scope."""

    profiler: Profiler
    name: str

    def __init__(self, profiler: "Profiler", name: str):
        self.profiler = profiler
        self.name = name

    def __enter__(self):
        _ffi_api.ProfilerStartContextTimer(self.profiler, self.name)  # type: ignore # pylint: disable=no-member
        return self

    def __exit__(self, exctype, excinst, exctb):
        _ffi_api.ProfilerEndContextTimer(self.profiler)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.Profiler")
class Profiler(Object):
    """A profiler to count tuning time cost in different parts."""

    def __init__(self) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.Profiler,  # type: ignore # pylint: disable=no-member
        )

    def get(self) -> Dict[str, float]:
        """Get the profiling results in minutes"""
        return _ffi_api.ProfilerGet(self)  # type: ignore # pylint: disable=no-member

    def timeit(self, name: str) -> TimeItContext:
        return TimeItContext(self, name)

    def __enter__(self) -> "Profiler":
        """Entering the scope of the context manager"""
        _ffi_api.ProfilerEnterScope(self)  # type: ignore # pylint: disable=no-member
        return self

    def __exit__(self, ptype, value, trace) -> None:
        """Exiting the scope of the context manager"""
        _ffi_api.ProfilerExitScope(self)  # type: ignore # pylint: disable=no-member
