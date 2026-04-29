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
"""TVM Script APIs of TVM Python Package.

The user-facing surface (``tvm.script.from_source``, ``tvm.script.ir_module``,
``tvm.script.<dialect>``) resolves through a dialect registry plus
``__getattr__`` redirect. Extension dialects (``tvm.tirx``, ``tvm.relax``)
register themselves from ``tvm/__init__.py`` so that ``tvm.script.<name>``
points at ``tvm.<name>.script``. ``tvm.script`` itself stays
dialect-agnostic at module load time, so importing ``tvm.script`` from a
dialect's bootstrap path does not pull the dialect's own modules back in
via a circular import. The IR layer is foundational (``tvm.script``
depends on ``tvm.ir``) and is not registered as a dialect — its parser /
ir_builder live as real submodules under ``tvm.script.parser.ir`` /
``tvm.script.ir_builder.ir``.
"""

import importlib
from typing import Any

_DIALECT_REGISTRY: dict[str, str] = {}


def register_dialect(name: str, module_path: str) -> None:
    """Register a dialect's script package path.

    Parameters
    ----------
    name : str
        The short name exposed under ``tvm.script.<name>`` (e.g. ``"tirx"``).
    module_path : str
        The full module path that owns the dialect's script subpackages, e.g.
        ``"tvm.tirx.script"``. The submodules ``parser``, ``ir_builder`` (and,
        if applicable, ``printer``) under that path are reachable through
        ``tvm.script.parser.<name>``, ``tvm.script.ir_builder.<name>``, etc.
    """
    _DIALECT_REGISTRY[name] = module_path


def __getattr__(name: str) -> Any:
    if name in _DIALECT_REGISTRY:
        module = importlib.import_module(_DIALECT_REGISTRY[name])
        globals()[name] = module
        return module
    if name == "ir":
        # IR is foundational — its parser is a real submodule under
        # tvm.script.parser.ir, exposed here as `tvm.script.ir` for the
        # legacy `from tvm.script import ir as I` pattern.
        ir_parser = importlib.import_module("tvm.script.parser.ir")
        globals()["ir"] = ir_parser
        return ir_parser
    if name in ("from_source", "parse"):
        from .parser._core import parse  # pylint: disable=import-outside-toplevel

        globals()["from_source"] = parse
        globals()["parse"] = parse
        return parse
    if name == "ir_module":
        # ir_module lives in the IR parser at tvm.script.parser.ir; the IR
        # layer is foundational, so we resolve it directly rather than via
        # the dialect registry.
        ir_parser = importlib.import_module("tvm.script.parser.ir")
        ir_module_value = ir_parser.ir_module
        globals()["ir_module"] = ir_module_value
        return ir_module_value
    raise AttributeError(f"module 'tvm.script' has no attribute {name!r}")
