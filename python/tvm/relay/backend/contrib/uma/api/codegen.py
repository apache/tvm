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

from typing import Callable, Optional
import tvm


class UMACodegen(object):
    """
    Codegen base class of the Universal Modular Accelerator Interface (UMA)
    """

    def __init__(self, target_name: str) -> None:
        self.target_name = target_name

    def _register_codegen(
        self, fmt: str = "c", includes: Optional[Callable[[], str]] = None, **kwargs
    ) -> None:
        """Registration codegen in UMA.

        Parameters
        ----------
        fmt: str
            format of codegen. Currently only "c" is supported.
        includes : OptionalCallable[[], str]]
            user-defined function that adds C-#include statement to UMA C-Code.
        """
        if fmt == "c":
            self._register_c_codegen(includes, **kwargs)
        else:
            raise RuntimeError(f'Unsupported codegen format "{fmt}"')

    def _register_c_codegen(self, includes: Optional[Callable[[], str]] = None) -> None:
        """Registration of UMA helper functions, e.g. includes and replace_call_extern.

        Parameters
        ----------
        includes : OptionalCallable[[], str]]
            user-defined function that adds C-#include statement to UMA C-Code.
        """
        if includes is not None:
            tvm._ffi.register_func(
                f"relay.ext.uma.codegen_c_includes_{self.target_name}",
                includes,
                override=True,
            )

    def register(self) -> None:
        pass
