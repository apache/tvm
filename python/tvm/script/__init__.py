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
``tvm.script.<dialect>``) resolves through a dialect registry plus an
import redirect. Extension dialects (``tvm.tirx``, ``tvm.relax``) are
registered from ``tvm/__init__.py`` so that ``tvm.script.<name>`` points
at ``tvm.<name>.script``. ``tvm.script`` itself stays dialect-agnostic at
module load time, so importing ``tvm.script`` from a dialect's bootstrap
path does not pull the dialect's own modules back in via a circular
import. The IR layer is foundational (``tvm.script`` depends on
``tvm.ir``) and is not registered as a dialect — its parser / ir_builder
live as real submodules under ``tvm.script.{parser,ir_builder}.ir``.

Two import paths are supported and resolve to the same module:

* attribute access (``from tvm.script import tirx``, ``tvm.script.tirx``)
  is handled by ``__getattr__`` on the relevant package.
* statement-form import (``import tvm.script.parser.tirx``,
  ``from tvm.script.ir_builder.relax.ir import py_print``) is handled by
  a :class:`_DialectRedirectFinder` registered on ``sys.meta_path``.
  Both forms register the resolved module in ``sys.modules`` under the
  legacy name so subsequent imports skip the redirect.
"""

import importlib
import importlib.util
import sys
from typing import Any

_DIALECT_REGISTRY: dict[str, str] = {}

# Subpackages of `tvm.script` whose per-dialect children are redirected to a
# matching subpackage under `tvm.<dialect>.script`. The values are the
# subpackage name on the dialect side (e.g. `tvm.<dialect>.script.parser`).
_REDIRECTED_SUBPACKAGES = {
    "tvm.script.parser": "parser",
    "tvm.script.ir_builder": "builder",
}


def register_dialect(name: str, module_path: str) -> None:
    """Register a dialect's script package path.

    Parameters
    ----------
    name : str
        The short name exposed under ``tvm.script.<name>`` (e.g. ``"tirx"``).
    module_path : str
        The full module path that owns the dialect's script subpackages, e.g.
        ``"tvm.tirx.script"``. The submodules ``parser``, ``builder`` (and,
        if applicable, ``printer``) under that path are reachable through
        ``tvm.script.parser.<name>``, ``tvm.script.ir_builder.<name>``, etc.
    """
    _DIALECT_REGISTRY[name] = module_path


def _redirect_target(fullname: str) -> str | None:
    """Return the target module path for a redirected ``tvm.script[...]`` name.

    Returns ``None`` if ``fullname`` is not a redirected name.
    """
    if fullname.startswith("tvm.script."):
        # tvm.script.<dialect>[.subpath]
        rest = fullname[len("tvm.script.") :]
        head, _, tail = rest.partition(".")
        if head in _DIALECT_REGISTRY and "." not in head:
            target = _DIALECT_REGISTRY[head]
            return f"{target}.{tail}" if tail else target
        # tvm.script.parser.<dialect>[.subpath] / tvm.script.ir_builder.<dialect>[.subpath]
        for prefix, sub in _REDIRECTED_SUBPACKAGES.items():
            if fullname == prefix or not fullname.startswith(prefix + "."):
                continue
            rest = fullname[len(prefix) + 1 :]
            head, _, tail = rest.partition(".")
            if head in _DIALECT_REGISTRY:
                target = f"{_DIALECT_REGISTRY[head]}.{sub}"
                return f"{target}.{tail}" if tail else target
    return None


class _DialectRedirectFinder:
    """``sys.meta_path`` finder that redirects legacy ``tvm.script.<dialect>``
    paths to the corresponding ``tvm.<dialect>.script`` modules."""

    @classmethod
    def find_spec(cls, fullname, path, target=None):
        redirected = _redirect_target(fullname)
        if redirected is None:
            return None
        # Resolve the target module and alias it under the legacy name.
        module = importlib.import_module(redirected)
        sys.modules[fullname] = module
        return importlib.util.spec_from_loader(fullname, _AliasLoader(module))


class _AliasLoader:
    """Loader that returns an already-resolved module for an alias spec."""

    def __init__(self, module):
        self._module = module

    def create_module(self, spec):
        return self._module

    def exec_module(self, module):
        # Module is already populated by the redirect target.
        return None


# Install the redirect finder once. Re-importing tvm.script (e.g. during a
# pytest reload) must not stack duplicates.
if not any(isinstance(f, _DialectRedirectFinder) for f in sys.meta_path):
    sys.meta_path.append(_DialectRedirectFinder())


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
