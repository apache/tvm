# License .to the Apache Software Foundation (ASF) under one
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
# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable, invalid-name, redefined-builtin
"""
APIs for a packaging module
"""
from tvm.runtime import _ffi_api

class PackagingModule:
    """The Packaging module"""
    def __init__(self, mod):
        self.mod = mod
        self._get_source = self.mod["get_source"]
        self._get_source_type = self.mod["get_source_type"]
        self._get_metadata = self.mod["get_metadata"]
        self._is_c_source = self.mod["is_c_source"]

    @property
    def source(self):
        """Get the source"""
        return self._get_source()

    @property
    def source_type(self):
        """Get the source type"""
        return self._get_source_type()

    @property
    def metadata(self):
        """Get the metadata"""
        return self._get_metadata()

    def is_c_source(self):
        """Check if the source code is C/C++"""
        return self._is_c_source()


def CSourceModule(code, fmt="c"):
    """Create a C source module"""
    return _ffi_api.CSourceModuleCreate(code, fmt)


def ModuleInitWrapper(metadata, code="", source_type="c"):
    """Create a module initialization wrapper"""
    return _ffi_api.ModuleInitWrapper(metadata, code, source_type)
