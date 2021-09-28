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
# pylint: disable=invalid-name
"""The under development unified IR parsing infrastructure."""
from .. import _ffi, Object
from . import _ffi_api


@_ffi.register_object("SourceMap")
class SourceMap(Object):
    def add(self, name, content):
        return _ffi.get_global_func("SourceMapAdd")(self, name, content)


def parse(source, source_name="from_string", init_module=None, init_meta_table=None):
    if init_meta_table is None:
        init_meta_table = {}
    return _ffi_api.ParseModuleInContext(source_name, source, init_module, init_meta_table)


def parse_expr(source):
    return _ffi_api.ParseExpr("string", source)


def fromtext(source, source_name="from_string"):
    return parse(source, source_name)


def SpanCheck():
    """A debugging utility for reporting missing span information."""
    return _ffi_api.SpanCheck()
