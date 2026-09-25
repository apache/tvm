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
"""Concrete TIRx construction operations over the shared native IRBuilder stack."""

from tvm.ir import is_prim_var as is_type_var
from tvm.script.ir_builder.ir import dynamic as dynamic
from tvm.script.parser.protocol_registry import constexpr as constexpr
from tvm.tirx.lang.alloc_pool import SMEMPool as SMEMPool
from tvm.tirx.lang.alloc_pool import TMEMPool as TMEMPool

from . import ir as _native
from . import tirx as tile
from .ir import *
from .ir import __all__ as _ir_exports
from .ir import boolean as bool
from .op import *
from .op import __all__ as _op_exports
from .op import _get_script_namespace as __getattr__
from .parser_protocol import *
from .parser_protocol import __all__ as _protocol_exports
from .tirx import (
    cluster as cluster,
)
from .tirx import (
    cta as cta,
)
from .tirx import (
    thread as thread,
)
from .tirx import (
    warp as warp,
)
from .tirx import (
    warpgroup as warpgroup,
)
from .tirx import (
    wg as wg,
)
from .utils import (
    buffer_indices as buffer_indices,
)
from .utils import (
    frame_scope as frame_scope,
)
from .utils import (
    seq_scope as seq_scope,
)

supports_mutable_declarations = True

__all__ = [
    *_ir_exports,
    *_op_exports,
    *_protocol_exports,
    "is_type_var",
    "dynamic",
    "constexpr",
    "SMEMPool",
    "TMEMPool",
    "bool",
    "tile",
    "cluster",
    "cta",
    "thread",
    "warp",
    "warpgroup",
    "wg",
    "buffer_indices",
    "frame_scope",
    "seq_scope",
    "supports_mutable_declarations",
]
