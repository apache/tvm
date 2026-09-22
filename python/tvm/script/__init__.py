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
"""TVMScript public namespace and registered dialect exports.

All source parsing uses :mod:`tvm.script.parser`; dialect packages expose source
constructors and concrete builders from the same canonical implementation.
The registry permits out-of-tree dialects to expose the same public spellings.
"""

import importlib
import importlib.util
import sys
from typing import Any

_DIALECT_REGISTRY: dict[str, str] = {}

# Subpackages of `tvm.script` whose per-dialect children are redirected to a
# canonical package under `tvm.<dialect>.script`. An empty suffix names
# the public dialect namespace itself.
_REDIRECTED_SUBPACKAGES = {
    "tvm.script.parser": "",
    "tvm.script.ir_builder": "builder",
}


def register_dialect(name: str, module_path: str) -> None:
    """Register a dialect's script package path.

    Writes ``name -> module_path`` into ``_DIALECT_REGISTRY``.  After
    registration, ``tvm.script.<name>`` resolves to ``module_path`` via
    ``__getattr__``, and ``tvm.script.parser.<name>`` / ``tvm.script.ir_builder.<name>``
    resolve to the same script namespace and ``module_path + ".builder"``
    respectively. Deep builder imports are handled by
    ``_DialectRedirectFinder`` on ``sys.meta_path``.

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
        ``"tvm.tirx.script"``. That package exposes its public construction
        entry points directly and its imperative APIs in a ``builder`` submodule.
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
                target = _DIALECT_REGISTRY[head] + (f".{sub}" if sub else "")
                return f"{target}.{tail}" if tail else target
    return None


class _DialectRedirectFinder:
    """``sys.meta_path`` finder that redirects ``tvm.script.<dialect>`` import paths.

    PEP 562 ``__getattr__`` only handles one-level attribute lookups
    (``from tvm.script import tirx``).  It cannot intercept deep
    statement-form imports such as::

        import tvm.script.tirx.tile
        import tvm.script.ir_builder.relax.ir

    This finder is installed on ``sys.meta_path`` to cover those cases.
    When the import machinery asks for a module whose full name starts with
    ``tvm.script.<dialect>`` (or ``tvm.script.parser.<dialect>``, etc.) and
    that dialect is in ``_DIALECT_REGISTRY``, :meth:`find_spec` imports the
    real target module (e.g. ``tvm.tirx.script.tile``) and returns an
    alias spec whose loader hands back that module, so the import machinery
    registers it in ``sys.modules`` under the public alias and all subsequent
    imports and attribute walks resolve without going through the redirect
    again.
    """

    @classmethod
    def find_spec(cls, fullname, path, target=None):
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
        from .parser import parse  # pylint: disable=import-outside-toplevel

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
