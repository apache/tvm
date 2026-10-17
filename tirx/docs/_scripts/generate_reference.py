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
"""Generate the Python reference from source exports and live native-backed APIs.

Namespace classification is explicit and exhaustive. Source declarations and
import aliases set order; star imports expand recursively. Runtime insertion
order handles generated exports without declarations, then lexical order handles
names advertised only by __all__. __all__ controls membership, never ordering.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import importlib
import importlib.util
import inspect
import itertools
import json
import os
import types
from pathlib import Path

os.environ["TVM_FFI_BUILD_DOCS"] = "1"
DOCS = Path(__file__).resolve().parents[1]
SOURCE = DOCS.parent / "python"
REFERENCE = DOCS / "reference" / "python"
HEADER = (DOCS / "index.rst").read_text().split("tvm-tirx\n")[0]
POLICY = json.loads(Path(__file__).with_name("namespace_policy.json").read_text())
LEGACY = json.loads(Path(__file__).with_name("legacy_anchors.json").read_text())
TOP = [
    "runtime",
    "ir",
    "ir.transform",
    "script",
    "sym",
    "target",
    "tirx",
    "tirx.transform",
    "codegen",
    "backend",
    "support",
    "testing",
]
PRIORITY = [
    "ir",
    "runtime",
    "ir.prim",
    "ir.transform",
    "sym",
    "target",
    "codegen",
    "tirx",
]
HIDDEN = {"tvm_tirx.ir." + n for n in ("ExprWithOp", "Scriptable", "ExprOperand")}
GROUPED = {
    "tvm_tirx.ir.ForKind": "tvm_tirx.ir.For",
    "tvm_tirx.ir.CallEffectKind": "tvm_tirx.ir.Call",
    "tvm_tirx.ir.CallingConv": "tvm_tirx.ir.BaseFunc",
    "tvm_tirx.ir.UNKNOWN_LOC": "tvm_tirx.ir.Location",
}
EXCLUDED_OBJECTS = {
    "tvm_tirx.tvm_wrap_excepthook",
    "tvm_tirx.codegen.Array",
    "tvm_tirx.codegen.Target",
    "tvm_tirx.codegen.get_global_func",
    "tvm_tirx.tirx.op.ExprOp",
    "tvm_tirx.tirx.script.jit.make_jit",
    "tvm_tirx.tirx.tile.views.ArgProperty",
    "tvm_tirx.script.parser.protocol_registry.DefinitionKind",
    "tvm_tirx.tirx.transform.RemoveNoOpConfig",
    "tvm_tirx.tirx.transform.UnrollLoopConfig",
    "tvm_tirx.backend.cuda.launch.register_launch_ops",
    "tvm_tirx.backend.cuda.launch.register_launch_helpers",
}
PREFERRED_CANONICAL = {"tvm_tirx.backend.cuda.thread_scope.ThreadScope"} | {
    f"tvm_tirx.tirx.script.nvshmem.{operation}.{scope}"
    for operation in ("getmem_nbi", "putmem_nbi", "putmem_signal_nbi")
    for scope in ("block", "warp")
}
PRIMITIVE_HELPERS = {
    "const",
    "convert",
    "expr_deep_equal",
    "infinity",
    "max_value",
    "min_value",
}


def qualify(name):
    """Expand a policy namespace to its package name."""
    return name if name == "tvm_tirx" else "tvm_tirx." + name


def module_files():
    """Inventory every non-private source module, including excluded modules."""
    result = {}
    for path in sorted((SOURCE / "tvm_tirx").rglob("*.py")):
        parts = list(path.relative_to(SOURCE).with_suffix("").parts)
        if any(n.startswith("_") and n != "__init__" for n in parts[1:]):
            continue
        if parts[-1] == "__init__":
            parts.pop()
        result[".".join(parts)] = path
    classified = {qualify(n) for n in POLICY["public"] + list(POLICY["excluded"])}
    missing = result.keys() - classified
    obsolete = classified - result.keys()
    if missing or obsolete:
        raise ValueError(
            f"Update namespace_policy.json: unclassified={sorted(missing)}, "
            f"missing source={sorted(obsolete)}"
        )
    return result


def source_order(module, files, seen=()):
    """Follow declarations and import aliases without alphabetizing reexports."""
    if module in seen or module not in files:
        return []
    path = files[module]
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    names = []

    def statements(nodes):
        for node in nodes:
            yield node
            if isinstance(node, ast.If | ast.Try | ast.For | ast.While | ast.With):
                yield from statements(node.body)
                yield from statements(getattr(node, "orelse", []))
                yield from statements(getattr(node, "finalbody", []))
                for handler in getattr(node, "handlers", []):
                    yield from statements(handler.body)

    for node in statements(ast.parse(path.read_text()).body):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.append(node.name)
        elif isinstance(node, ast.ImportFrom):
            target = (
                importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                if node.level
                else node.module
            )
            for alias in node.names:
                if alias.name == "*":
                    names.extend(source_order(target, files, (*seen, module)))
                else:
                    names.append(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            names.extend(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.extend(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))
    # The script facade copies its builder's globals inside lazy initialization.
    if module == "tvm_tirx.tirx.script":
        names = source_order(module + ".ir_builder", files, (*seen, module)) + names
    return list(dict.fromkeys(names))


def dynamic_objects():
    """Expand lazy namespaces once, retaining objects to prevent identity reuse."""
    from tvm_tirx.backend.cuda import script as cuda
    from tvm_tirx.backend.cuda.ptx.engine import _InstrChain
    from tvm_tirx.tirx import script as script

    script._initialize()  # pylint: disable=protected-access
    roots = {
        "tvm_tirx.backend.cuda.script." + name: getattr(cuda, name)
        for name in ("tile", "wgmma", "tcgen05", "iket")
    }
    roots.update(
        {
            "tvm_tirx.tirx.script." + n: getattr(script, n)
            for n in ("nvshmem", "ptx", "ptx_legacy", "nki", "trn", "metal", "webgpu")
        }
    )
    result = {}

    def walk(prefix, obj, depth=0):
        if depth > 4:
            raise ValueError(f"Unexpected recursive dynamic namespace: {prefix}")
        # These namespaces expose generated members via __dir__, not __dict__.
        # Their stable lexical order is the documented source-less fallback.
        for name in sorted(dir(obj)):
            if name.startswith("_"):
                continue
            child = getattr(obj, name)
            path = prefix + "." + name
            if callable(child):
                result[path] = child
                if (
                    isinstance(child, _InstrChain)
                    or inspect.isroutine(child)
                    or inspect.isclass(child)
                ):
                    continue
            if type(child).__name__.endswith("Namespace") or isinstance(
                child, types.SimpleNamespace
            ):
                walk(path, child, depth + 1)

    for prefix, obj in roots.items():
        walk(prefix, obj)
    for name in (
        "__shfl_sync",
        "__shfl_up_sync",
        "__shfl_down_sync",
        "__shfl_xor_sync",
        "__activemask",
    ):
        result["tvm_tirx.backend.cuda.script." + name] = getattr(cuda, name)
    return result


def inventory(files):
    """Collect supported live exports, canonical identities and class members."""
    from sphinx.ext.autodoc.mock import mock

    # Initialize dialect exports before observing any package's membership.
    dynamic = dynamic_objects()
    rows = []
    modules = {}
    objects = {}
    for short in POLICY["public"]:
        name = qualify(short)
        with mock(["torch"]):
            module = importlib.import_module(name)
        if Path(module.__file__).resolve() != files[name].resolve():
            raise ValueError(f"Reference imported a different checkout: {name}")
        modules[name] = module
        tree = ast.parse(files[name].read_text())
        defined = {
            n.name
            for n in tree.body
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        }
        explicit = getattr(module, "__all__", None)
        ordered = list(
            dict.fromkeys(source_order(name, files) + list(vars(module)) + sorted(explicit or []))
        )
        for key in ordered:
            if key.startswith("_"):
                continue
            if not hasattr(module, key):
                continue
            obj = getattr(module, key)
            if explicit is not None and key not in explicit:
                partial = POLICY.get("partial_all", {}).get(short)
                if not partial or not (
                    key in defined
                    or (partial["include_generated"] and getattr(obj, "__module__", None) == name)
                ):
                    continue
            if isinstance(obj, types.ModuleType):
                continue
            origin = getattr(obj, "__module__", "") or ""
            imported = any(
                isinstance(n, ast.ImportFrom)
                and (n.level or (n.module or "").startswith("tvm_tirx"))
                and any((a.asname or a.name) == key for a in n.names)
                for n in tree.body
            )
            public = (
                explicit is not None
                or key in defined
                or origin == name
                or (
                    imported
                    and any(
                        isinstance(n, ast.ImportFrom) and any(a.asname == key for a in n.names)
                        for n in tree.body
                    )
                )
                or (
                    files[name].name == "__init__.py"
                    and (
                        origin.startswith(name + ".")
                        or imported
                        or (
                            name in {"tvm_tirx", "tvm_tirx.ir", "tvm_tirx.runtime"}
                            and origin.startswith(("tvm_tirx", "tvm_ffi"))
                        )
                    )
                )
            )
            if public:
                objects[name + "." + key] = obj
    objects.update(dynamic)
    excluded = {}
    hidden_ids = {id(obj) for name, obj in objects.items() if name in HIDDEN}
    for name, obj in objects.items():
        kind = "class" if inspect.isclass(obj) else "function" if callable(obj) else "data"
        module, _, short = name.rpartition(".")
        if name in EXCLUDED_OBJECTS or id(obj) in hidden_ids:
            excluded[name] = (
                "Internal helper or registration implementation; "
                "compatibility helper classes are orphan pages."
            )
            continue
        if (
            kind == "data"
            and short
            not in {
                "UNKNOWN_LOC",
                "MISSING",
                "R",
                "S",
                "let",
                "supports_mutable_declarations",
            }
            and type(obj).__name__ not in {"Axis", "_AxisNamespace"}
        ):
            excluded[name] = (
                "Generated namespace documented through its operations."
                if type(obj).__name__.endswith("Namespace")
                else "Implementation metadata or registration table, not a construction API."
            )
            continue
        if module == "tvm_tirx.backend.cuda.ptx" and short not in {
            "PTXNamespace",
            "instruction_forms",
            "instruction_name",
        }:
            excluded[name] = "PTX registration implementation; use public instruction families."
            continue
        rows.append(
            {
                "name": name,
                "module": module,
                "short": short,
                "kind": kind,
                "dynamic": name in dynamic,
                "object": obj,
            }
        )
    by_id = collections.defaultdict(list)
    for row in rows:
        by_id[id(row["object"])].append(row["name"])

    def rank(name):
        module = name.rpartition(".")[0].removeprefix("tvm_tirx.")
        return (
            (
                PRIORITY.index(module)
                if module in PRIORITY
                else len(PRIORITY) + (5 if module == "tvm_tirx" else 0)
            ),
            name.count("."),
            name,
        )

    canonical = {}
    for names in by_id.values():
        owner = min(names, key=lambda n: (n not in PREFERRED_CANONICAL, rank(n)))
        canonical.update(dict.fromkeys(names, owner))
    # PTX families are lazily allocated; each family gets its own canonical page.
    for row in rows:
        if row["dynamic"] and row["module"] == "tvm_tirx.tirx.script.ptx":
            canonical[row["name"]] = row["name"]
    return rows, canonical, modules, {name: objects[name] for name in HIDDEN}, excluded


def heading(title, marker="-"):
    """Render an RST section heading."""
    width = max(len(title), 8) if marker == "=" else len(title)
    return title + "\n" + marker * width + "\n\n"


def page_name(module):
    """Preserve the established namespace URLs."""
    return "index" if module == "tvm_tirx" else module.removeprefix("tvm_tirx.")


def generate():
    """Return generated pages and a JSON-serializable audit manifest."""
    files = module_files()
    rows, canonical, modules, hidden, excluded = inventory(files)
    by_name = {r["name"]: r for r in rows}
    by_module = {name: [] for name in modules}
    for row in rows:
        by_module.setdefault(row["module"], []).append(row)
    for name in list(by_module):
        parent = name.rpartition(".")[0]
        while parent and parent != "tvm_tirx":
            by_module.setdefault(parent, [])
            parent = parent.rpartition(".")[0]
    # Stable source-first namespace order, with explicitly prominent entry points.
    children = collections.defaultdict(list)
    for name in by_module:
        if name != "tvm_tirx":
            children[name.rpartition(".")[0]].append(name)
    prominent = {
        "tvm_tirx": [qualify(n) for n in TOP],
        "tvm_tirx.ir": ["tvm_tirx.ir.prim", "tvm_tirx.ir.transform"],
        "tvm_tirx.script": ["tvm_tirx.script.parser", "tvm_tirx.script.ir_builder"],
        "tvm_tirx.tirx": [
            "tvm_tirx.tirx." + n
            for n in (
                "op",
                "analysis",
                "transform",
                "layout",
                "tile",
                "compilation_pipeline",
                "script",
            )
        ],
        "tvm_tirx.backend": ["tvm_tirx.backend.cuda"],
    }
    for parent, names in children.items():
        order = source_order(parent, files)
        names.sort(
            key=lambda n: (
                (
                    order.index(n.rsplit(".", 1)[-1])
                    if n.rsplit(".", 1)[-1] in order
                    else len(order)
                ),
                n,
            )
        )
        first = prominent.get(parent, [])
        if parent.startswith("tvm_tirx.backend.") and parent + ".op" in names:
            first = [parent + ".op"]
        children[parent] = list(dict.fromkeys(first + names))

    cases = collections.defaultdict(list)
    for row in rows:
        if canonical[row["name"]] == row["name"]:
            cases[row["name"].casefold()].append(row)
    filename_map = {}
    for values in cases.values():
        if len(values) > 1:
            for row in values:
                filename_map[row["name"]] = row["name"] + "_" + row["kind"]
    # Preserve already published disambiguated filenames even without a collision.
    filename_map["tvm_tirx.runtime.Device"] = "tvm_tirx.runtime.Device_class"

    tables = {}

    def summary(module, title, names):
        names = [n for n in names if n not in GROUPED]
        if not names:
            return ""
        tables.setdefault(module, []).append({"title": title, "names": names})
        # Separate runs, not separate global alias tables, preserve source ordering.
        text = heading(title)
        for is_alias, group in itertools.groupby(names, lambda n: canonical[n] != n):
            group = list(group)
            if is_alias:
                text += (
                    ".. list-table::\n   :header-rows: 1\n   :widths: 35 65\n\n"
                    "   * - Alias\n     - Canonical API\n"
                )
                for name in group:
                    text += (
                        f"   * - ``{by_name[name]['short']}``\n     - :obj:`{canonical[name]}`\n"
                    )
                text += "\n"
            else:
                text += ".. autosummary::\n   :toctree: generated/\n\n"
                text += "".join("   " + by_name[n]["short"] + "\n" for n in group) + "\n"
        return text

    pages = {}
    for module, items in by_module.items():
        text = HEADER + heading(module, "=")
        if module in modules:
            text += f".. automodule:: {module}\n   :no-members:\n"
            if module in by_name:
                text += "   :no-index:\n"
            text += "\n"
        else:
            owner = module
            while owner not in modules:
                owner = owner.rpartition(".")[0]
            text += f"Generated namespace exposed by :mod:`{owner}`.\n\n"
        text += f".. currentmodule:: {module}\n\n"
        if module == "tvm_tirx.backend.cuda.transform":
            for anchor in (
                "tvm-tirx-backend-cuda-transforms",
                "module-tvm_tirx.backend.cuda.transforms",
            ):
                text += f'.. raw:: html\n\n   <span id="{anchor}"></span>\n\n'

        classes = [r for r in items if r["kind"] == "class"]
        functions = [r["name"] for r in items if r["kind"] == "function"]
        if module == "tvm_tirx.ir":
            import tvm_tirx.ir as ir

            assigned = set()
            for title in ("Location", "Type", "Expr", "Stmt", "IRModule"):
                family = [
                    r["name"]
                    for r in classes
                    if issubclass(r["object"], getattr(ir, title))
                    or (title == "Location" and r["short"] == "SourceName")
                ]
                root = module + "." + title
                family = [root] + [n for n in family if n != root]
                assigned.update(family)
                text += summary(module, title, family)
            text += summary(
                module,
                "Supporting classes",
                [r["name"] for r in classes if r["name"] not in assigned],
            )
            text += summary(module, "IR helpers", functions)
        elif module == "tvm_tirx.tirx.transform":
            decorators = [module + ".FunctionPass", module + ".function_pass"]
            text += summary(module, "Pass decorator helpers", decorators)
            text += summary(module, "Passes", [n for n in functions if n not in decorators])
        else:
            text += summary(module, "Classes", [r["name"] for r in classes])
            if module == "tvm_tirx.ir.prim":
                text += summary(
                    module,
                    "Operator functions",
                    [n for n in functions if n.rpartition(".")[2] not in PRIMITIVE_HELPERS],
                )
                text += summary(
                    module,
                    "Helpers and constant constructors",
                    [n for n in functions if n.rpartition(".")[2] in PRIMITIVE_HELPERS],
                )
            else:
                text += summary(module, "Functions", functions)
        text += summary(
            module,
            "Constants and construction helpers",
            [r["name"] for r in items if r["kind"] == "data"],
        )
        if children[module]:
            text += heading("Namespaces")
            text += ".. list-table::\n   :header-rows: 1\n\n   * - Namespace\n     - Purpose\n"
            for child in children[module]:
                intro = (
                    inspect.getdoc(modules[child]) if child in modules else "Generated operations."
                )
                intro = (intro or "Namespace reference.").splitlines()[0].replace("`", "")
                text += f"   * - :doc:`{child} <{page_name(child)}>`\n     - {intro}\n"
            text += "\n.. toctree::\n   :hidden:\n   :maxdepth: 2\n\n"
            text += "".join("   " + page_name(c) + "\n" for c in children[module]) + "\n"
        for anchor in LEGACY.get(page_name(module), []):
            if anchor == module.replace(".", "-").replace("_", "-"):
                continue
            text += f'.. raw:: html\n\n   <span id="{anchor}"></span>\n\n'
        pages[page_name(module) + ".rst"] = text.rstrip() + "\n"

    inherited = {}
    public_types = {r["object"] for r in rows if r["kind"] == "class"}
    for row in rows + [
        {"name": name, "object": obj, "kind": "class"} for name, obj in hidden.items()
    ]:
        name, obj = row["name"], row["object"]
        if name in GROUPED or canonical.get(name, name) != name:
            continue
        text = HEADER + (":orphan:\n\n" if name in hidden else "") + heading(name, "=")
        directive = {
            "class": "autoclass",
            "function": "autofunction",
            "data": "autodata",
        }[row["kind"]]
        text += f".. {directive}:: {name}\n"
        if row["kind"] == "data":
            text += "   :no-value:\n"
        text += "\n"
        compatibility = []
        if name.startswith("tvm_tirx.backend.cuda.transform."):
            compatibility.append(name.replace(".transform.", ".transforms."))
        if name == "tvm_tirx.runtime.hexagon":
            compatibility.append("tvm_tirx.hexagon")
        for anchor in compatibility:
            text += f'.. raw:: html\n\n   <span id="{anchor}"></span>\n\n'
        if name == "tvm_tirx.ir.Var":
            text += '.. raw:: html\n\n   <span id="tvm_tirx.tirx.Var"></span>\n\n'
        if row["kind"] == "class" and name not in hidden:
            members = []
            for member in dict.fromkeys(n for cls in obj.__mro__ for n in vars(cls)):
                if member.startswith("_"):
                    continue
                owner = next(cls for cls in obj.__mro__ if member in vars(cls))
                value = vars(owner)[member]
                if (
                    owner not in public_types
                    and owner.__module__.startswith("tvm_tirx")
                    and (
                        callable(value)
                        or isinstance(value, property | staticmethod | classmethod)
                        or type(value).__name__ in {"FFIProperty", "_tuplegetter"}
                    )
                ):
                    members.append(member)
            if members:
                inherited[name] = members
                text += heading("Inherited methods and properties")
                for member in members:
                    value = inspect.getattr_static(obj, member)
                    directive = (
                        "autoproperty"
                        if isinstance(value, property)
                        else (
                            "autoattribute"
                            if type(value).__name__ in {"FFIProperty", "_tuplegetter"}
                            else "automethod"
                        )
                    )
                    text += f".. {directive}:: {name}.{member}\n\n"
        if row["kind"] == "data" and type(obj).__name__ == "_AxisNamespace":
            text += f".. automethod:: {name}.get\n\n"
        for grouped, owner in GROUPED.items():
            if name == owner:
                text += heading(grouped.rpartition(".")[2])
                if grouped.endswith("UNKNOWN_LOC"):
                    text += (
                        "The shared immutable, non-null :class:`tvm_tirx.ir.UnknownLoc` sentinel "
                        "represents an unspecified location in compiler mode. Runtime-only "
                        "imports expose ``None``.\n\n"
                    )
                    text += f".. autodata:: {grouped}\n   :no-value:\n\n"
                else:
                    text += f".. autoclass:: {grouped}\n\n"
        pages["generated/" + filename_map.get(name, name) + ".rst"] = text.rstrip() + "\n"

    # Preserve old implementation/category URLs with links and their historical anchors.
    for old_name in sorted(set(LEGACY) | {"ir.base"}):
        old = Path(old_name + ".rst")
        if old.name in pages or old.stem == "config":
            continue
        if old.stem == "backend.cuda.transforms":
            continue
        text = HEADER + ":orphan:\n\n" + heading(old.stem, "=")
        target = "ir" if old.stem == "ir.base" else "index"
        text += f":doc:`Namespace reference <{target}>`\n\n"
        for anchor in LEGACY.get(old.stem, []):
            text += f'.. raw:: html\n\n   <span id="{anchor}"></span>\n\n'
            destination = next(
                (
                    m
                    for m in by_module
                    if m.replace(".", "-").replace("_", "-")
                    == anchor.replace("-transforms", "-transform")
                ),
                None,
            )
            if destination:
                text += f":doc:`{destination} <{page_name(destination)}>`\n\n"

        pages[old.name] = text.rstrip() + "\n"

    implementation_aliases = {}
    for row in rows:
        obj = row["object"]
        owner = getattr(obj, "__module__", None)
        qualname = getattr(obj, "__qualname__", None)
        if owner and qualname and "<locals>" not in qualname:
            implementation_aliases[owner + "." + qualname] = canonical[row["name"]]

    manifest = {
        "namespace_pages": {m: page_name(m) for m in by_module},
        "source_sha256": {m: hashlib.sha256(p.read_bytes()).hexdigest() for m, p in files.items()},
        "module_classification": {
            m: ("public" if m in modules else POLICY["excluded"][m.removeprefix("tvm_tirx.")])
            for m in files
        },
        "objects": [{k: v for k, v in r.items() if k != "object"} for r in rows],
        "canonical": canonical,
        "implementation_aliases": implementation_aliases,
        "filename_map": filename_map,
        "tables": tables,
        "children": dict(children),
        "grouped": GROUPED,
        "inherited": inherited,
        "hidden": sorted(HIDDEN),
        "dynamic": [r["name"] for r in rows if r["dynamic"]],
        "legacy_anchors": LEGACY,
        "object_exclusions": excluded,
        "dynamic_exclusions": {
            "tvm_tirx.tirx.script.s_tir": "Printer-roundtrip surface; no hand-authored syntax.",
            "ptx_modifier_permutations": (
                "One canonical page per lazy family; modifiers are "
                "documented by PTXNamespace and instruction_forms."
            ),
        },
    }
    return pages, manifest


def build_reference(check=False):
    """Write generated sources and return the manifest for Sphinx configuration."""
    pages, manifest = generate()
    changes = []
    for name, text in pages.items():
        path = REFERENCE / name
        if not path.exists() or path.read_text() != text:
            changes.append(name)
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
    if check and changes:
        raise SystemExit("Reference is stale: " + ", ".join(changes))
    if not check:
        for path in (REFERENCE / "generated").glob("*.rst"):
            if str(path.relative_to(REFERENCE)) not in pages:
                path.unlink()
        obsolete = REFERENCE / "backend.cuda.transforms.rst"
        obsolete.unlink(missing_ok=True)
        output = DOCS / "_build" / "reference-manifest.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"{len(manifest['namespace_pages'])} namespaces, {len(manifest['objects'])} exports, "
        f"{len(set(manifest['canonical'].values()))} canonical APIs; {len(changes)} changed pages"
    )

    return manifest


def main():
    """Regenerate the reference, or check that its sources are current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    build_reference(check=args.check)


if __name__ == "__main__":
    main()
