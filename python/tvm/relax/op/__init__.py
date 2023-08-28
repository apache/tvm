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
# pylint: disable=wildcard-import, redefined-builtin
"""Relax core operators."""

# Operators
from .base import *
from .binary import *
from .create import *
from .datatype import *
from .index import *
from .linear_algebra import *
from .manipulate import *
from .mask import *
from .op_attrs import *
from .statistical import *
from .search import *
from .set import *
from .ternary import *
from .unary import *
from . import builtin
from . import distributed
from . import grad
from . import image
from . import memory
from . import nn
from . import ccl

# Register operator gradient functions
from . import _op_gradient


def _register_op_make():
    # pylint: disable=import-outside-toplevel
    from . import _ffi_api
    from .. import expr

    expr._op_ffi_api = _ffi_api  # type: ignore


_register_op_make()
