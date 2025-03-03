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
"""The tir parser"""

from typing import TYPE_CHECKING

from ...ir_builder.tir import *  # pylint: disable=redefined-builtin
from ...ir_builder.tir import ir as _tir
from . import operation as _operation
from . import parser as _parser
from .entry import Buffer, Ptr

if TYPE_CHECKING:
    # pylint: disable=invalid-name
    # Define prim_func and make it type check as static method
    # so most tvmscript won't trigger pylint error here.
    prim_func = staticmethod
else:
    from .entry import macro, prim_func

__all__ = _tir.__all__ + ["Buffer", "Ptr", "bool", "prim_func", "macro"]
