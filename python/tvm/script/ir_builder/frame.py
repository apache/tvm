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
"""Package tvm.script.ir_builder.frame"""

from tvm_ffi import register_object as _register_object

from . import _ffi_api
from .base import IRBuilderFrame


@_register_object("script.ir_builder.IRModuleFrame")
class IRModuleFrame(IRBuilderFrame):
    def __exit__(self, exc_type, exc_value, trace):
        try:
            return super().__exit__(exc_type, exc_value, trace)
        finally:
            # Body and exit-callback failures must not leave an active module scope.
            _ffi_api.IRModuleFrameAbort(self)

    def __getattr__(self, name):
        """Expose a declared function through the native module's reference map.

        Reflected native fields and Python methods resolve before this fallback.
        Source aliases therefore retain this frame and its plain GlobalVars.
        """
        try:
            return self.global_vars[name]
        except KeyError:
            raise AttributeError(f"IRModuleFrame has no attribute {name!r}") from None
