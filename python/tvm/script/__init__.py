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
"""TVMScript public namespace and registered language variant exports.

All source parsing uses :mod:`tvm.script.parser`; language variant packages expose source
constructors and concrete builders from the same canonical implementation.
The registry permits out-of-tree language variants to expose the same public spellings.
"""

import importlib
import importlib.machinery
import importlib.util
import sys
from collections.abc import Callable
from typing import Any

_DIALECT_REGISTRY: dict[str, str] = {}
_DIALECT_BUILDER_REGISTRY: dict[str, str] = {}
# Language variants register whole-module checks at import, independently of source decorators.
# Callbacks own concrete eligibility; this list retains no source or construction state.
_MODULE_VALIDATORS: list[Callable[[Any], None]] = []

# Public parser and builder compatibility paths consume dialect-owned registrations.
_REDIRECTED_SUBPACKAGES = {
    "tvm.script.parser": _DIALECT_REGISTRY,
    "tvm.script.ir_builder": _DIALECT_BUILDER_REGISTRY,
}


def register_module_validator(validator: Callable[[Any], None], *, prepend: bool = False) -> None:
    """Register an opaque language variant check over a completed module.

    Parameters
    ----------
    validator : Callable[[Any], None]
        Whole-module validator. It owns concrete IR eligibility and raises on
        failure; shared coordination forwards its exception unchanged.
    prepend : bool, optional
        Run this check before existing callbacks. False appends it instead.
        Re-registering the same callback is a no-op and preserves its position.

    Returns
    -------
    None
        Registration changes no IR and returns no source-visible result.

    Notes
    -----
    Register beside language variant initialization so captured and preexisting functions
    are checked even when no corresponding source decorator appears. A callback
    may lazily import its language variant implementation to preserve import order. The
    registry owns static hooks only, never source scopes, frames or IR results.
    """
    if any(existing is validator for existing in _MODULE_VALIDATORS):
        return
    if prepend:
        _MODULE_VALIDATORS.insert(0, validator)
    else:
        _MODULE_VALIDATORS.append(validator)


def register_dialect(name: str, module_path: str, *, builder_path: str | None = None) -> None:
    """Register a dialect's script namespace and canonical builder package.

    Registration is lazy: it records paths without importing either package.
    ``tvm.script.<name>`` and ``tvm.script.parser.<name>`` expose ``module_path``;
    ``tvm.script.ir_builder.<name>`` exposes ``builder_path``. Deep imports under
    these compatibility paths resolve to the same canonical modules, preserving
    their identity and avoiding repeated initialization. Existing physical
    out-of-tree packages under ``tvm.script.ir_builder`` retain normal package
    lookup precedence.

    Each dialect registers from its own package initialization::

        import tvm.script
        tvm.script.register_dialect(
            "example", "example.script", builder_path="example.script.ir_builder"
        )

    Out-of-tree dialects register the same way without editing shared TVMScript
    files. Re-registering the same name and paths is idempotent.

    Parameters
    ----------
    name : str
        Short name exposed under ``tvm.script.<name>``.
    module_path : str
        Full dotted path of the dialect's script package, which owns its public
        construction namespace.
    builder_path : str, optional
        Full dotted path of the dialect-owned imperative builder package,
        normally ``module_path + ".ir_builder"``. When omitted, use
        ``module_path + ".builder"`` to preserve existing out-of-tree
        registrations. No imports or aliases are created at registration time.
    """
    _DIALECT_REGISTRY[name] = module_path
    _DIALECT_BUILDER_REGISTRY[name] = (
        builder_path if builder_path is not None else module_path + ".builder"
    )


def _redirect_target(fullname: str) -> str | None:
    """Return the target module path for a redirected ``tvm.script[...]`` name.

    Returns ``None`` if ``fullname`` is not a redirected name.
    """
    if fullname.startswith("tvm.script."):
        # tvm.script.<language variant>[.subpath]
        rest = fullname[len("tvm.script.") :]
        head, _, tail = rest.partition(".")
        if head in _DIALECT_REGISTRY and "." not in head:
            target = _DIALECT_REGISTRY[head]
            return f"{target}.{tail}" if tail else target
        # Registered language variants may redirect parser and builder subpackages.
        for prefix, registry in _REDIRECTED_SUBPACKAGES.items():
            if fullname == prefix or not fullname.startswith(prefix + "."):
                continue
            rest = fullname[len(prefix) + 1 :]
            head, _, tail = rest.partition(".")
            if head in registry:
                target = registry[head]
                return f"{target}.{tail}" if tail else target
    return None


class _DialectRedirectFinder:
    """``sys.meta_path`` finder that redirects ``tvm.script.<language variant>`` import paths.

    PEP 562 ``__getattr__`` only handles one-level attribute lookups
    (``from tvm.script import tirx``).  It cannot intercept deep
    statement-form imports such as::

        import tvm.script.tirx.tile
        import tvm.script.ir_builder.relax.ir

    This finder is installed on ``sys.meta_path`` to cover those cases.
    When the import machinery asks for a module whose full name starts with
    ``tvm.script.<language variant>`` (or ``tvm.script.parser.<language variant>``, etc.) and
    that language variant is in ``_DIALECT_REGISTRY``, :meth:`find_spec` imports the
    real target module (e.g. ``tvm.tirx.script.tile``) and returns an
    alias spec whose loader hands back that module, so the import machinery
    registers it in ``sys.modules`` under the public alias and all subsequent
    imports and attribute walks resolve without going through the redirect
    again.
    """

    @classmethod
    def find_spec(cls, fullname, path, target=None):
        # Preserve real external builder packages before the registered fallback.
        prefix = "tvm.script.ir_builder."
        if fullname.startswith(prefix):
            package_name = prefix + fullname[len(prefix) :].partition(".")[0]
            parent_path = sys.modules["tvm.script.ir_builder"].__path__
            if importlib.machinery.PathFinder.find_spec(package_name, parent_path) is not None:
                return None
        redirected = _redirect_target(fullname)
        if redirected is None:
            return None
        # Resolve the target module and return an alias spec for it.  Do NOT
        # pre-register ``sys.modules[fullname]`` here: if ``fullname`` is in
        # ``sys.modules`` when find_spec returns, ``_bootstrap._find_spec``
        # discards the returned spec in favor of ``module.__spec__`` — the
        # target's original SourceFileLoader spec — and ``_load_unlocked``
        # then RE-EXECUTES the target module and replaces its canonical
        # ``sys.modules`` entry with the duplicate.  The machinery registers
        # the alias under ``fullname`` itself when loading the spec below.
        module = importlib.import_module(redirected)
        return importlib.util.spec_from_loader(fullname, _AliasLoader(module))


class _AliasLoader:
    """Loader that returns an already-resolved module for an alias spec."""

    def __init__(self, module):
        self._module = module
        # ``module_from_spec`` unconditionally stamps the alias spec onto the
        # module returned by ``create_module``; capture the canonical values
        # so ``exec_module`` can restore them.
        self._spec = getattr(module, "__spec__", None)
        self._loader = getattr(module, "__loader__", None)

    def create_module(self, spec):
        return self._module

    def exec_module(self, module):
        # Module is already populated by the redirect target; just restore
        # the canonical ``__spec__``/``__loader__`` that the import machinery
        # overwrote with the alias spec (a stale alias ``__spec__.parent``
        # breaks relative imports inside the module).
        if self._spec is not None:
            module.__spec__ = self._spec
        if self._loader is not None:
            module.__loader__ = self._loader


# Install the redirect finder once. Re-importing tvm.script (e.g. during a
# pytest reload) must not stack duplicates.
if not any(isinstance(f, _DialectRedirectFinder) for f in sys.meta_path):
    # Handle deep aliases before PathFinder can execute a canonical source file
    # again under its alias package's __path__.
    sys.meta_path.insert(0, _DialectRedirectFinder())


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
        from .parser import parse  # pylint: disable=import-outside-toplevel

        globals()["from_source"] = parse
        globals()["parse"] = parse
        return parse
    if name == "ir_module":
        # ir_module lives in the IR parser at tvm.script.parser.ir; the IR
        # layer is foundational, so we resolve it directly rather than via
        # the language variant registry.
        ir_parser = importlib.import_module("tvm.script.parser.ir")
        ir_module_value = ir_parser.ir_module
        globals()["ir_module"] = ir_module_value
        return ir_module_value
    raise AttributeError(f"module 'tvm.script' has no attribute {name!r}")


# Shared source entry points participate in the same opaque namespace boundary.
from .parser import register_namespace as _register_namespace

_register_namespace("script", sys.modules[__name__])
