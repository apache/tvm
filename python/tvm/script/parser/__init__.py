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
"""Canonical TVMScript AST parser and public construction namespaces."""

import importlib
import sys
from typing import TypeVar

_FRONTEND_EXPORTS = (
    "_NAMESPACES", "from_source", "ir_module", "make_decorator", "make_helper",
    "parse", "pyfunc", "register_namespace",
)
__all__ = [name for name in _FRONTEND_EXPORTS if not name.startswith("_")]
_initialized = False
_initializing = False


def _initialize():
    global _initialized, _initializing
    if _initialized or _initializing:
        return
    _initializing = True
    try:
        from tvm import relax
        from tvm.relax import script as relax_namespace
        from tvm.relax.script import builder as relax_builder
        from tvm.script.ir_builder import construction
        from tvm.tirx import script as tir_namespace
        from tvm.tirx.layout import Axis
        from tvm.tirx.script import builder as tir_builder
        from tvm.tirx.script import tile
        from . import frontend, ir
        from .jit import OptionalAnnotation, make_jit

        for namespace, builder in ((tir_namespace, tir_builder), (relax_namespace, relax_builder)):
            namespace.__dict__.update(
                (name, value) for name, value in vars(builder).items() if not name.startswith("_")
            )
        # Source assignment consumes a receipt; imperative bind returns the Var.
        tir_namespace.bind = tir_builder._native.bind
        tir_namespace.prim_func = frontend.make_decorator(
            tir_builder, option_map={"private": "private", "s_tir": "s_tir", "persistent": "persistent"}
        )
        tir_namespace.jit = make_jit(tir_builder)
        tir_namespace.Optional = OptionalAnnotation
        tir_namespace.inline = frontend.make_helper(tir_builder, preserve_return=True, late_binding=True)
        tir_namespace.macro = frontend.make_helper(tir_builder, preserve_return=False)
        tir_namespace.tile = tile
        for name in ("cluster", "cta", "thread", "warp", "warpgroup", "wg"):
            setattr(tir_namespace, name, getattr(tile, name))
        relax_namespace.function = frontend.make_decorator(
            relax_builder, option_map={"pure": "is_pure", "private": "is_private"}
        )
        relax_namespace.macro = frontend.make_helper(relax_builder, preserve_return=True)
        ir.ir_module = frontend.ir_module
        ir.pyfunc = frontend.pyfunc
        for namespace in (tir_namespace, relax_namespace, ir):
            namespace.__all__ = [name for name in vars(namespace) if not name.startswith("_")]
        frontend._NAMESPACES.update(
            I=ir, ir=ir, T=tir_namespace, tir=tir_namespace, tirx=tir_namespace,
            R=relax_namespace, relax=relax_namespace, Tx=tile, Axis=Axis, TypeVar=TypeVar,
        )

        def opaque(name, function, source, span):
            return relax.ExternFunc(name, span=span).with_attrs({
                "is_pyfunc": True, "function_type": "python", "python_function_name": name,
                "python_source": source, "python_packed_func": function,
            })

        construction.register_opaque_factory(opaque)
        _initialized = True
    finally:
        _initializing = False


def __getattr__(name):
    if name in _FRONTEND_EXPORTS:
        _initialize()
        return getattr(importlib.import_module(f"{__name__}.frontend"), name)
    if name in ("tirx", "tir", "relax", "I", "T", "R", "Tx"):
        _initialize()
        return sys.modules[f"{__name__}.frontend"]._NAMESPACES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
