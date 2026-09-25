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
"""Public canonical TVMScript dialect namespace."""

import importlib as _importlib
import sys as _sys
from typing import Any as _Any

_initialized = False
_initializing = False
_ENTRY_EXPORTS = ("jit", "Optional")


def _initialize() -> None:
    global _initialized, _initializing
    if _initialized or _initializing:
        return
    _initializing = True
    try:
        from tvm.script.parser import entry, register_namespace
        from tvm.script.parser.protocol_registry import declaration_kind
        from tvm.tirx.layout import Axis

        from . import ir_builder as builder
        from . import tile
        from .jit import make_jit

        globals().update(
            (name, value) for name, value in vars(builder).items() if not name.startswith("_")
        )
        globals().update(
            bind=builder.bind,
            prim_func=declaration_kind("T.prim_func", "function")(entry.make_decorator(builder)),
            inline=declaration_kind("T.inline", "helper")(
                entry.make_macro_decorator(builder, preserve_return=True, late_binding=True)
            ),
            macro=declaration_kind("T.macro", "helper")(
                entry.make_macro_decorator(builder, preserve_return=False)
            ),
            jit=declaration_kind("T.jit", "function")(make_jit(builder)),
            tile=tile,
        )
        for name in ("cluster", "cta", "thread", "warp", "warpgroup", "wg"):
            globals()[name] = getattr(tile, name)
        namespace = _sys.modules[__name__]
        for alias in ("T", "tir", "tirx"):
            register_namespace(alias, namespace)
        register_namespace("Tx", tile)
        register_namespace("Axis", Axis)
        globals()["__all__"] = sorted(name for name in globals() if not name.startswith("_"))
        _initialized = True
    finally:
        _initializing = False


def __getattr__(name: str) -> _Any:
    if name == "ir_builder":
        return _importlib.import_module(__name__ + ".ir_builder")
    if name == "tile":
        return _importlib.import_module(__name__ + ".tile")
    if name.startswith("_") and name != "__all__":
        raise AttributeError(name)
    _initialize()
    if name in globals():
        return globals()[name]
    return getattr(_importlib.import_module("tvm.tirx.script.ir_builder"), name)


from .jit import OptionalAnnotation as Optional

globals().pop("jit", None)
