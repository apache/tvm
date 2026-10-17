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
"""Audit generated reference coverage, source ordering and rendered HTML targets."""

import ast
import hashlib
import importlib
import inspect
import json
import zlib
from html.parser import HTMLParser
from urllib.parse import unquote

from generate_reference import DOCS, GROUPED, HIDDEN, SOURCE, module_files, source_order


class Targets(HTMLParser):
    """Collect actual HTML targets and navigation links."""

    def __init__(self, text):
        super().__init__()
        self.ids = set()
        self.links = set()
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.add(unquote(attrs["href"]))


def main():
    """Fail on missing public targets, hierarchy edges, ordering or members."""
    manifest = json.loads((DOCS / "_build/reference-manifest.json").read_text())
    html = DOCS / "_build/html"
    data = (html / "objects.inv").read_bytes().split(b"\n", 4)[4]
    inventory = {}
    for line in zlib.decompress(data).decode().splitlines():
        name, kind, _, uri, _ = line.split(" ", 4)
        if kind.startswith("py:"):
            inventory[name] = uri[:-1] + name if uri.endswith("$") else uri
    unresolved = json.loads((DOCS / "_build/unresolved-references.json").read_text())
    public_aliases = set(manifest["canonical"]) | set(manifest["implementation_aliases"])
    assert not (unresolved.keys() & public_aliases), unresolved.keys() & public_aliases
    # Public surfaces whose __all__ intentionally advertises only lazy/bootstrap names.
    # These direct APIs must remain covered alongside star-import exports.
    for module, names in {
        "tvm_tirx.tirx.layout": ("Layout", "Axis", "Iter", "TileLayout", "ComposeLayout"),
        "tvm_tirx.backend.opencl": ("nd_mem_alloc_with_scope", "texture2d_load", "texture2d_store"),
    }.items():
        for name in names:
            assert module + "." + name in manifest["canonical"], (module, name)
    expected = set(manifest["canonical"].values()) | HIDDEN
    missing = expected - inventory.keys()
    assert not missing, f"Missing canonical object targets: {sorted(missing)}"
    parsed = {}

    def targets(relative):
        if relative not in parsed:
            parsed[relative] = Targets((html / relative).read_text())
        return parsed[relative]

    # Check real anchors, not just inventory names or successful Sphinx exit.
    for name in expected:
        page, _, anchor = inventory[name].partition("#")
        assert anchor in targets(page).ids, (name, page, anchor)
    namespaces = manifest["namespace_pages"]
    visited = set()

    def visit(module):
        if module in visited:
            return
        visited.add(module)
        parent = "reference/python/" + namespaces[module] + ".html"
        for child in manifest["children"][module]:
            child_page = namespaces[child] + ".html"
            assert child_page in targets(parent).links, (module, child)
            visit(child)

    visit("tvm_tirx")
    assert visited == namespaces.keys(), f"Unreachable namespaces: {namespaces.keys() - visited}"
    for module in (
        "ir.prim",
        "tirx.op",
        "tirx.analysis",
        "tirx.transform",
        "tirx.layout",
        "tirx.tile",
        "tirx.compilation_pipeline",
        "tirx.script.ir_builder",
        "script.parser",
        "script.ir_builder",
        "script.ir_builder.parser_protocol",
        "backend.cuda.op",
        "backend.cuda.transform",
        "backend.trn.op",
        "backend.metal.op",
        "tirx.script.nki",
        "tirx.script.trn.tile",
        "tirx.script.metal",
    ):
        assert "tvm_tirx." + module in namespaces, module
    for old, anchors in manifest["legacy_anchors"].items():
        if old == "backend.cuda.transforms":
            continue
        for anchor in anchors:
            assert anchor in targets("reference/python/" + old + ".html").ids, (
                old,
                anchor,
            )
    assert "tvm_tirx.tirx.Var" in targets("reference/python/generated/tvm_tirx.ir.Var.html").ids
    for old, owner in GROUPED.items():
        filename = "reference/python/generated/" + old + ".html"
        assert (html / filename).exists(), filename
        assert inventory[old].split("#")[0] == inventory[owner].split("#")[0]
    for name in manifest["dynamic"]:
        assert manifest["canonical"][name] in inventory, name
    for name, members in manifest["inherited"].items():
        for member in members:
            assert name + "." + member in inventory, (name, member)
    files = module_files()
    for name, path in files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["source_sha256"][name], (
            name
        )
    by_name = {r["name"]: r for r in manifest["objects"]}
    for module, tables in manifest["tables"].items():
        order = source_order(module, files)
        for table in tables:
            assert not (set(table["names"]) & HIDDEN), table
            known = [by_name[n]["short"] for n in table["names"] if by_name[n]["short"] in order]
            if module == "tvm_tirx.ir" and table["title"] in {
                "Location",
                "Type",
                "Expr",
                "Stmt",
                "IRModule",
            }:
                known = [n for n in known if n != table["title"] and n != "SourceName"]
            if table["title"] == "Pass decorator helpers":
                assert known == ["FunctionPass", "function_pass"]
            else:
                indices = [order.index(n) for n in known]
                assert indices == sorted(indices), (module, table["title"], known)
    transform = manifest["tables"]["tvm_tirx.tirx.transform"]
    assert [t["title"] for t in transform] == ["Pass decorator helpers", "Passes"]
    # Independently derive the expected pass order from the implementation AST.
    tree = ast.parse((SOURCE / "tvm_tirx/tirx/transform/transform.py").read_text())
    passes = [
        n.name for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
    ]
    assert [n.rpartition(".")[2] for n in transform[1]["names"]] == passes
    # Own public methods/properties must be documented; inherited public bases link normally.
    member_count = 0
    for row in manifest["objects"]:
        if row["kind"] != "class" or manifest["canonical"][row["name"]] != row["name"]:
            continue
        from sphinx.ext.autodoc.mock import mock

        with mock(["torch"]):
            obj = getattr(importlib.import_module(row["module"]), row["short"])
        for name, value in vars(obj).items():
            if name.startswith("_") or not (
                inspect.isroutine(value) or isinstance(value, property | staticmethod | classmethod)
            ):
                continue
            target = row["name"] + "." + name
            assert target in inventory, f"Undocumented declared member: {target}"
            member_count += 1
    print(
        json.dumps(
            {
                "namespaces": len(namespaces),
                "exports": len(manifest["objects"]),
                "canonical_targets": len(expected),
                "dynamic_exports": len(manifest["dynamic"]),
                "declared_members": member_count,
                "inherited_members": sum(map(len, manifest["inherited"].values())),
                "html_pages_checked": len(parsed),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
