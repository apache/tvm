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
# pylint: disable=no-member, super-init-not-called

"""Backend-defined execution scope identities."""

from tvm_ffi import register_object

from tvm.ir import Scriptable
from tvm.runtime import Object

from . import _ffi_api


@register_object("tirx.ExecScope")
class ExecScope(Object, Scriptable):
    """A node in one backend's registered execution hierarchy."""

    backend: str
    name: str

    def __init__(self, backend: str, name: str):
        self.__init_handle_by_constructor__(_ffi_api.ExecScope, backend, name)


def register_exec_scope(backend: str, name: str, rank: int):
    """Register a backend's finite scope hierarchy, from outer to inner ranks.

    Backend load hooks register scopes before registering their layout axes.
    """
    _ffi_api.RegisterExecScope(backend, name, rank)
