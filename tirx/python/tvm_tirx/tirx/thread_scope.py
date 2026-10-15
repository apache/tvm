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

"""Canonical enum tags for backend-defined thread hierarchies."""

from tvm_ffi.dataclasses import Enum

from tvm.ir import Scriptable

from . import _ffi_api


class ThreadScope(Enum, Scriptable, type_key="tirx.ThreadScope", init=False):
    """A registered thread scope, queried by its qualified backend.scope name.

    Backend and hierarchy rank are enum attributes. The registration index
    does not determine the hierarchy.
    """

    @classmethod
    def get(cls, name: str) -> "ThreadScope":
        """Look up a registered tag without creating unknown scopes."""
        return _ffi_api.ThreadScopeGet(name)

    @property
    def name(self) -> str:
        """Canonical qualified name, such as cuda.warp."""
        return self._str_index

    @property
    def backend(self) -> str:
        """Backend owning this tag."""
        return _ffi_api.ThreadScopeGetBackend(self)

    @property
    def local_name(self) -> str:
        """Backend-local name, such as warp."""
        return _ffi_api.ThreadScopeGetLocalName(self)

    @property
    def rank(self) -> int:
        """Explicit hierarchy rank; smaller ranks denote wider scopes."""
        return _ffi_api.ThreadScopeGetRank(self)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"ThreadScope({self.name!r})"
