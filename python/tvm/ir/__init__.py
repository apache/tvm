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
# pylint: disable=unused-import
"""Common data structures across all IR variants."""
from . import diagnostics, instrument, transform
from .adt import Constructor, TypeData
from .affine_type import TensorAffineType, TupleAffineType
from .attrs import Attrs, DictAttrs, make_node
from .base import (
    EnvFunc,
    Node,
    SourceName,
    Span,
    SequentialSpan,
    assert_structural_equal,
    load_json,
    save_json,
    structural_equal,
    structural_hash,
)
from .container import Array, Map
from .expr import BaseExpr, GlobalVar, PrimExpr, Range, RelayExpr
from .function import BaseFunc, CallingConv
from .global_info import GlobalInfo, DummyGlobalInfo, VDevice
from .memory_pools import (
    ConstantMemoryPools,
    ConstantPoolInfo,
    PoolInfo,
    PoolInfoProperties,
    WorkspaceMemoryPools,
    WorkspacePoolInfo,
)
from .module import IRModule
from .op import Op, register_intrin_lowering, register_op_attr
from .tensor_type import TensorType
from .type import (
    FuncType,
    GlobalTypeVar,
    IncompleteType,
    PointerType,
    PrimType,
    RelayRefType,
    TupleType,
    Type,
    TypeConstraint,
    TypeKind,
    TypeVar,
)
from .type_relation import TypeCall, TypeRelation
