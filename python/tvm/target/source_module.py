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
Helper functions and classes for handling source and metdata.
"""
from tvm.runtime import _ffi_api

class SourceMetadataModule:
    """The module used to wrap both source and metadata."""
    def __init__(self, mod):
        self.mod = mod
        self._get_source = self.mod["get_source"]
        self._get_symbol = self.mod["get_symbol"]
        self._get_source_type = self.mod["get_source_type"]
        self._get_variables = self.mod["get_vars"]
        self._get_metadata = self.mod["get_metadata"]

    @property
    def symbol(self):
        """Get the source"""
        return self._get_symbol()

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

    @property
    def variables(self):
        """Get the metadata"""
        return self._get_variables()


    def is_c_source(self):
        """Check if the source code is C/C++"""
        return self.source_type == "c" or self.source_type == "cc"


def CSourceModule(code, fmt="c"):
    """Create a C source module"""
    return _ffi_api.CSourceModuleCreate(code, fmt)
