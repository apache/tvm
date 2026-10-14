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
"""Canonical, extensible memory spaces shared by the compiler and runtime."""

from tvm_ffi.dataclasses import Enum

from . import _ffi_api


class MemoryScope(Enum, type_key="runtime.MemoryScope", init=False):
    """A registered memory-space identity.

    ``MemoryScope(name)`` looks up an existing scope and raises ``ValueError``
    for an unknown name. Registration alone does not enable backend support.

    Examples
    --------
    .. code-block:: python

        custom = MemoryScope.register("my_accelerator.scratch")
        assert MemoryScope("my_accelerator.scratch") == custom
    """

    @property
    def name(self) -> str:
        """Return the canonical registered spelling."""
        return self._str_index

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"MemoryScope({self.name!r})"

    @classmethod
    def register(cls, name: str) -> "MemoryScope":
        """Register a new canonical name, raising ValueError for a duplicate."""
        if not isinstance(name, str):
            raise TypeError("memory scope name must be a string")
        return _ffi_api.MemoryScopeRegister(name)
