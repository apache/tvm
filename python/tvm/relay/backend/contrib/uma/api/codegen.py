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
"""Codegen base class of the Universal Modular Accelerator Interface (UMA)"""

import tvm

from typing import Callable


class UMACodegen(object):
    def __init__(self, target_name: str) -> None:
        self.target_name = target_name

    def _register_codegen(self, fmt: str = "c", **kwargs) -> None:
        if fmt == "c":
            self._register_c_codegen(**kwargs)
        else:
            raise RuntimeError(f'Unsupported codegen format "{fmt}"')

    def _register_c_codegen(
        self,
        includes: Callable[[], str] = None,
        replace_call_extern: Callable[[tvm.ir.container.Array], str] = None,
    ) -> None:
        if includes is not None:
            tvm._ffi.register_func(
                "relay.ext.uma.codegen_c_includes_{}".format(self.target_name), includes
            )
        if replace_call_extern is not None:
            tvm._ffi.register_func(
                "relay.ext.uma.codegen_c_replace_call_extern_{}".format(self.target_name),
                replace_call_extern,
            )

    def register(self) -> None:
        pass
