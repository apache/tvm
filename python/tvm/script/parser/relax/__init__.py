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
"""Initial impl of relax parser for sugars"""

from typing import TYPE_CHECKING

from ...ir_builder.relax import *  # pylint: disable=redefined-builtin
from ...ir_builder.relax import ir as _relax
from . import parser as _parser
from .entry import Callable, Object, Prim, Shape, Tensor, Tuple, match_cast

from . import dist
from .dist import *  # pylint: disable=wildcard-import,redefined-builtin

if TYPE_CHECKING:
    # pylint: disable=invalid-name
    # Define prim_func and make it type check as static method
    # so most tvmscript won't trigger pylint error here.
    function = staticmethod
else:
    from .entry import function, macro

__all__ = _relax.__all__ + [
    "dist",
    "Callable",
    "Object",
    "Prim",
    "Shape",
    "Tensor",
    "Tuple",
    "function",
    "macro",
    "match_cast",
]
