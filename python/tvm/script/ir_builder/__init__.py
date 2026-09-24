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
"""Shared TVMScript construction APIs and lazy language variant builders."""

import importlib as _importlib
from typing import Any as _Any

from tvm.ir import GenericConst, Range, StringImm, StringType
from tvm.script.parser.protocol_registry import constexpr

from .base import (
    MISSING,
    AlreadyEmitted,
    IRBuilder,
    annotation_value_,
    at_,
    require_defined,
    with_at_group_,
)
from .frame import IRModuleFrame
from .ir import (
    decl_function,
    def_function,
    ir_module,
    lookup_name,
    meta_var,
    module_attrs,
    module_get_attr,
    module_global_infos,
    module_set_attr,
)
from .parser_protocol import check_well_formed_, module_member_

# Keep source namespaces independent of imported helper modules and lazy builders.
__all__ = [
    "MISSING",
    "AlreadyEmitted",
    "GenericConst",
    "IRBuilder",
    "IRModuleFrame",
    "Range",
    "StringImm",
    "StringType",
    "annotation_value_",
    "at_",
    "check_well_formed_",
    "constexpr",
    "decl_function",
    "def_function",
    "ir_module",
    "lookup_name",
    "meta_var",
    "module_attrs",
    "module_get_attr",
    "module_global_infos",
    "module_member_",
    "module_set_attr",
    "require_defined",
    "with_at_group_",
]


def __getattr__(name: str) -> _Any:
    # Real packages resolve directly; the finder supplies registered external builders.
    from tvm.script import _DIALECT_REGISTRY  # pylint: disable=import-outside-toplevel

    if name in _DIALECT_REGISTRY:
        module = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'tvm.script.ir_builder' has no attribute {name!r}")
