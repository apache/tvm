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
"""Types specific to TIRX."""

import tvm_ffi

from tvm.ir import Type

from . import _ffi_api


@tvm_ffi.register_object("tirx.TensorMapType")
class TensorMapType(Type):
    """TensorMapType used in the low-level TIR.

    Parameters
    ----------
    span : tvm.ir.Span
        The span information.
    """

    def __init__(self, span=None):
        self.__init_handle_by_constructor__(
            _ffi_api.TensorMapType,
            span,  # pylint: disable=no-member
        )
