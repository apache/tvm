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
#pylint: disable=wildcard-import, redefined-builtin
"""Relay core operators."""
# operator defs
from .op import get, register, register_schedule, register_compute, register_gradient, \
    register_pattern, register_alter_op_layout, schedule_injective, Op, OpPattern, debug

# Operators
from .reduce import *
from .tensor import *
from .transform import *
from .algorithm import *
from . import nn
from . import annotation
from . import image
from . import vision
from . import contrib
from . import op_attrs


# operator registry
from . import _tensor
from . import _tensor_grad
from . import _transform
from . import _reduce
from . import _algorithm
from ..expr import Expr
from ..base import register_relay_node


def _register_op_make():
    from . import _make
    from .. import expr
    expr._op_make = _make

_register_op_make()
