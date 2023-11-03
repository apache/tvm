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
"""Wrapping Layouts."""
# pylint: disable=invalid-name, unsupported-binary-operation

from . import _ffi_api
import tvm
from tvm.ir import Node
from tvm.runtime import Scriptable

@tvm._ffi.register_object("tl.Layout")
class Layout(Node):
    def __init__(self, forward_var, forward_index):
        self.__init_handle_by_constructor__(_ffi_api.Layout, forward_var, forward_index)

    @property
    def var(self):
        return _ffi_api.Layout_var(self)

    @property
    def index(self):
        return _ffi_api.Layout_index(self)

    def get_input_shape(self):
        return _ffi_api.Layout_input_shape(self)

    def get_output_shape(self):
        return _ffi_api.Layout_output_shape(self)

    def inverse(self) -> "Layout":
        return _ffi_api.Layout_inverse(self)

@tvm._ffi.register_object("tl.Fragment")
class Fragment(Layout):
    def __init__(self, forward_var, forward_index, forward_thread, thread_replicate):
        self.__init_handle_by_constructor__(_ffi_api.Fragment, forward_var, forward_index, forward_thread, thread_replicate)

    @property
    def thread(self):
        return _ffi_api.Fragment_thread(self)

    @property
    def replicate_var(self):
        return _ffi_api.Fragment_replicate_var(self)

    def get_thread_size(self):
        return _ffi_api.Fragment_thread_size(self)

    def repeat(self, repeats, repeat_on_thread: bool=False) -> "Fragment":
        return _ffi_api.Fragment_repeat(self, repeats, repeat_on_thread)

    def condense_rep_var(self) -> "Fragment":
        return _ffi_api.Fragment_condense_rep_var(self)
