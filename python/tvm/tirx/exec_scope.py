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

"""Definition of execution scope."""

from tvm_ffi import register_object

from tvm.runtime import Object

from . import _ffi_api

_SCOPE_KIND_TO_NAME = {
    2: "cluster",
    3: "cta",
    4: "warpgroup",
    5: "warp",
    6: "thread",
}


@register_object("tirx.ExecScope")
class ExecScope(Object):
    """An execution scope, identified by one of {cluster, cta, warpgroup, warp,
    thread}. The ctor FATALs on any other name."""

    kind: int

    def __init__(self, name: str):
        self.__init_handle_by_constructor__(_ffi_api.ExecScope, name)

    @property
    def name(self) -> str:
        """Human-readable name of this scope (derived from ``kind``)."""
        return _SCOPE_KIND_TO_NAME[self.kind]
