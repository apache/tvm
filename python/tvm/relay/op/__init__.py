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
"""Relay core operators."""
# operator defs
from .op import (
    get,
    register_compute,
    register_gradient,
    register_pattern,
    register_alter_op_layout,
    register_legalize,
    OpPattern,
    OpStrategy,
    debug,
    register_external_compiler,
    register_fake_quantization_to_integer,
    register_optional_fake_quantization_to_integer,
    register_mixed_precision_conversion,
)
from . import strategy

# Operators
from .reduce import *
from .tensor import *
from .transform import *
from .algorithm import *
from . import vm
from . import nn
from . import annotation
from . import memory
from . import image
from . import vision
from . import op_attrs
from . import random


# operator registry
from . import _tensor
from . import _tensor_grad
from . import _transform
from . import _reduce
from . import _algorithm
from . import _math


def _register_op_make():
    # pylint: disable=import-outside-toplevel
    from . import _make
    from .. import expr

    expr._op_make = _make


_register_op_make()
