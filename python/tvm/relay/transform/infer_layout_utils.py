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
# pylint: disable=invalid-name, unused-argument, missing-docstring, unused-import
"""
Relay infer correct layout pass.
"""
import tvm
from tvm.runtime import Object
from . import _ffi_api


@tvm._ffi.register_object("relay._transform.InferCorrectLayoutOutput")
class InferCorrectLayoutOutput(Object):
    """An output structure to hold results from FInferCorrectLayout calls."""

    def __init__(self, input_layouts, output_layouts, new_attrs):
        self.__init_handle_by_constructor__(
            _ffi_api.InferCorrectLayoutOutput, input_layouts, output_layouts, new_attrs
        )
