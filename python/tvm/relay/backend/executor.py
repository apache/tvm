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
# pylint: disable=len-as-condition,no-else-return,invalid-name
"""Executor configuration"""

import tvm
from tvm.runtime import Object

from . import _backend


@tvm._ffi.register_object
class Executor(Object):
    """Executor configuration"""

    flag_registry_name = "executor"

    def __init__(self, name, options=None) -> None:
        if options is None:
            options = {}
        self.__init_handle_by_constructor__(_backend.CreateExecutor, name, options)
        self._init_wrapper()

    # Note:  sometimes the _attrs field is not properly populated,
    # most likely since __new__ is called instead of __init__ in tvm/_ffi/_ctypes/object.py
    def _init_wrapper(self):
        self._attrs = _backend.GetExecutorAttrs(self)
        self._init_wrapper_called = True

    def _check_init_wrapper(self):
        if not (hasattr(self, "_init_wrapper_called") and self._init_wrapper_called):
            self._init_wrapper()

    def __contains__(self, name):
        self._check_init_wrapper()
        return name in self._attrs

    def __getitem__(self, name):
        self._check_init_wrapper()
        return self._attrs[name]

    def __eq__(self, other):
        self._check_init_wrapper()
        return str(other) == str(self) and dict(other._attrs) == dict(self._attrs)

    @staticmethod
    def list_registered():
        """Returns a list of possible executors"""
        return list(_backend.ListExecutors())

    @staticmethod
    def list_registered_options(executor):
        """Returns the dict of available option names and types"""
        return dict(_backend.ListExecutorOptions(str(executor)))
