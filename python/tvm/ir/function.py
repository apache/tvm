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
"""Function defintiions."""
from enum import IntEnum
from .expr import RelayExpr
from . import _ffi_api


class CallingConv(IntEnum):
    """Possible kinds of calling conventions."""
    DEFAULT = 0
    C_PACKED_FUNC = 1
    DEVICE_KERNEL_LAUNCH = 2


class BaseFunc(RelayExpr):
    """Base class of all functions."""
    @property
    def attrs(self):
        """Return the attrs member of the function.
        """
        return _ffi_api.BaseFunc_Attrs(self)
