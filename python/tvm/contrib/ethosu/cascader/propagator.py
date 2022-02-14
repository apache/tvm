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
"""Propagator class."""
# pylint: disable=invalid-name
import tvm._ffi

from tvm.runtime import Object

from . import _ffi_api


@tvm._ffi.register_object("contrib.ethosu.cascader.Propagator")
class Propagator(Object):
    """Propagator class"""

    def __init__(self, transform, offset):
        float_transform = list([list(float(v) for v in row) for row in transform])
        self.__init_handle_by_constructor__(_ffi_api.Propagator, float_transform, offset)

    def propagate(self, stripe_config):
        return _ffi_api.PropagatorPropagate(self, stripe_config)

    @property
    def transform(self):
        """Get the transform matrix"""
        new_matrix = []
        for row in self._transform:
            new_row = []
            for v in row:
                new_row.append(v.value)

            new_matrix.append(new_row)

        return new_matrix

    @property
    def offset(self):
        """Get the offset matrix"""
        new_vec = []
        for v in self._offset:
            new_vec.append(v.value)

        return new_vec
