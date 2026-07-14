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
"""Trainium-owned TIRx modules."""

from importlib import import_module

_LAZY_SUBMODULES = {"layout", "op", "operator", "pipeline", "script", "target_tags", "transform"}


def register_backend():
    """Register Trainium-owned Python semantics."""
    from tvm.tirx import compilation_pipeline  # pylint: disable=import-outside-toplevel
    from tvm.tirx.script.builder import ir as builder_ir  # pylint: disable=import-outside-toplevel

    for name, namespace in script_namespaces().items():
        builder_ir.register_script_namespace(name, namespace)

    import_module(f"{__name__}.operator.tile_primitive")
    trn_pipeline = import_module(f"{__name__}.pipeline")
    import_module(f"{__name__}.target_tags")
    import_module(f"{__name__}.transform")
    compilation_pipeline.register_tir_pipeline("trn", trn_pipeline.trn_pipeline)


def script_namespace(op_wrapper=None):
    """Return the Trainium TVMScript namespace object."""
    from .script import NKINamespace  # pylint: disable=import-outside-toplevel

    return NKINamespace(op_wrapper)


def script_namespaces(op_wrapper=None, **_):
    """Return Trainium-owned TVMScript namespaces."""
    return {"nki": script_namespace(op_wrapper)}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "layout",
    "op",
    "operator",
    "pipeline",
    "register_backend",
    "script",
    "script_namespace",
    "script_namespaces",
    "target_tags",
    "transform",
]
