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
"""
TVMScript public namespace.

Dialect resolution mechanism
----------------------------

``tvm.script`` is a virtual namespace: dialect names like ``tirx`` and
``relax`` are not bound as static attributes here.  Instead:

- ``register_dialect(name, module_path)`` writes an entry to
  ``_DIALECT_REGISTRY: dict[str, str]``.  Each in-tree dialect's
  ``__init__.py`` calls this on import (e.g., ``tvm.tirx.__init__.py``
  calls ``tvm.script.register_dialect("tirx", "tvm.tirx.script")``).
  Out-of-tree dialects can register themselves the same way.

- ``__getattr__(name)`` (PEP 562) fires on missing attribute access.
  If ``name`` is in ``_DIALECT_REGISTRY``, the listed module is imported
  and cached as a normal module attribute.  Subsequent accesses
  skip ``__getattr__`` (cached in ``globals()``).

- Subpackages ``tvm.script.parser``, ``tvm.script.ir_builder``, etc.
  each define their own ``__getattr__`` that consults the SAME
  ``_DIALECT_REGISTRY`` and appends their suffix.  So
  ``tvm.script.parser.tirx`` resolves to ``tvm.tirx.script.parser`` via
  the dialect registry + ``.parser`` suffix.

- For deep statement-form imports like
  ``from tvm.script.parser.tirx.entry import ObjectProxy``, PEP 562's
  ``__getattr__`` is not enough — it only handles one-level
  ``from X import Y``.  A ``sys.meta_path`` finder (see
  ``_DialectRedirectFinder``) intercepts the import machinery to
  register the real module under the legacy name in ``sys.modules``,
  so subsequent attribute walks resolve correctly.

Each dialect's ``tvm.<dialect>.script`` package MUST expose ``parser``,
``ir_builder``, and (where applicable) ``printer`` as submodules.  This
convention is what makes the suffix-append redirect work uniformly.
IR is foundational (script depends on ir) and is NOT a dialect; its
script handlers live in the shared core, not via this registry.

Bootstrap order
---------------

``python/tvm/__init__.py`` imports ``tvm.script`` BEFORE importing any
dialect package (``tvm.tirx``, ``tvm.relax``, …).  This guarantees that
``tvm.script.register_dialect`` is reachable the moment a dialect's own
``__init__.py`` runs and calls it.  The ``tvm.script`` module itself
stays dialect-agnostic at load time (no dialect submodules are eagerly
imported here), so there is no circular dependency.
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

    Writes ``name -> module_path`` into ``_DIALECT_REGISTRY``.  After
    registration, ``tvm.script.<name>`` resolves to ``module_path`` via
    ``__getattr__``, and ``tvm.script.parser.<name>`` / ``tvm.script.ir_builder.<name>``
    resolve to ``module_path + ".parser"`` / ``module_path + ".builder"`` etc.
    via each subpackage's own ``__getattr__``.  Deep statement-form imports
    (e.g., ``from tvm.script.parser.<name>.entry import X``) are handled
    by ``_DialectRedirectFinder`` on ``sys.meta_path``.

    This function is idempotent — re-registering the same name with the same
    path is harmless.

    Each in-tree dialect calls this from its own ``__init__.py``::

        import tvm.script
        tvm.script.register_dialect("tirx", "tvm.tirx.script")

    Out-of-tree dialects do the same in their own package init without
    editing any in-tree file.

    Parameters
    ----------
    name : str
        The short name exposed under ``tvm.script.<name>`` (e.g. ``"tirx"``).
    module_path : str
        The full dotted module path of the dialect's script package, e.g.
        ``"tvm.tirx.script"``.  That package must expose ``parser`` and
        ``ir_builder`` as submodules (and ``printer`` where applicable) so
        that the suffix-append redirect works uniformly.
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
    """``sys.meta_path`` finder that redirects ``tvm.script.<dialect>`` import paths.

    PEP 562 ``__getattr__`` only handles one-level attribute lookups
    (``from tvm.script import tirx``).  It cannot intercept deep
    statement-form imports such as::

        from tvm.script.parser.tirx.entry import ObjectProxy
        import tvm.script.ir_builder.relax.ir

    This finder is installed on ``sys.meta_path`` to cover those cases.
    When the import machinery asks for a module whose full name starts with
    ``tvm.script.<dialect>`` (or ``tvm.script.parser.<dialect>``, etc.) and
    that dialect is in ``_DIALECT_REGISTRY``, :meth:`find_spec` imports the
    real target module (e.g. ``tvm.tirx.script.parser.entry``) and registers
    it in ``sys.modules`` under the legacy name, so all subsequent imports and
    attribute walks resolve correctly without going through the redirect again.
    """

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
