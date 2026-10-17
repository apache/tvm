<!---
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# Building the Python reference

Use a Python environment that can import this checkout's `tvm_tirx`, its pinned
TVM FFI dependency, and the matching built native compiler libraries. A
runtime-only installation cannot inventory compiler APIs. Install documentation
dependencies with `python -m pip install -r tirx/docs/requirements.txt`.

From the repository root:

```bash
python -m sphinx -W --keep-going -b html tirx/docs tirx/docs/_build/html
python tirx/docs/_scripts/generate_reference.py --check
python tirx/docs/_scripts/audit_reference.py
python tirx/docs/_scripts/parser_examples.py
```

Sphinx regenerates the reference from the imported source before reading it.
The generator is also directly executable. Commit the namespace pages; per-object
stubs and build artifacts are ignored. The build writes
`_build/reference-manifest.json` with namespace classifications, canonical aliases,
source-ordered tables, dynamic APIs and selectively inherited members. The audit
checks the built object inventory, HTML targets, navigation, source order and
compatibility anchors. Imported modules must come from this checkout; recorded
source hashes reject a stale artifact. No publication step is part of the build.

For a complete nitpicky diagnostic, rebuild every source with a fresh Sphinx
environment:

```bash
python -m sphinx -E -a -n -b dummy tirx/docs tirx/docs/_build/nitpicky
```

Each build records `_build/unresolved-references.json` for the pages it processes;
an incremental build replaces that file with only its processed-page results.
External types, private implementation annotations and free-form type prose may
remain unresolved.
The audit rejects unresolved references to known public aliases. The configuration
does not suppress these diagnostics.

`_scripts/namespace_policy.json` classifies every non-private source module.
Adding or removing one requires an explicit policy update; a missing classification
fails generation. Common implementation files are represented through their public
parent exports. The policy records why implementation-only namespaces are omitted.
`ir.prim`, public dialect extensions, parser/builder protocols and backend operator
namespaces retain their own pages. CLI configuration is excluded. Printer-only
`s_tir` and combinatorial PTX modifier permutations are internal surfaces; one page
per lazy PTX instruction family documents the supported grammar. Backend pattern
registrations and code-emission implementation files are classified separately from
public tile and code-generation interfaces.

Tables follow Python declarations and import-alias order. Star imports expand the
referenced source's order; `__all__` constrains membership without reordering it; the policy records the
partial `tirx.layout.__all__`, which advertises lazy aliases alongside public
class definitions, and `backend.opencl.__all__`, which advertises registration
alongside generated native operators.
Generated APIs without declarations use runtime insertion order, followed by a
lexical fallback for advertised-only names. Generated instance namespaces use
stable `dir()` order. Classes precede functions. IR families appear in Location,
Type, Expr, Stmt and IRModule order, with each base first. Supporting enums and
`UNKNOWN_LOC` share their owning class pages. Pass decorator helpers precede
source-ordered transformation passes.

Aliases link to one canonical object page. Internal expression helper classes have
orphan compatibility pages, while public subclasses explicitly document useful
methods inherited from undocumented helper bases, including builder frames and
launch configuration records. Other inherited methods are available
through the linked public base classes. The small Apache configuration preserves
legacy `.Var.html` URLs on servers that would otherwise treat `.var` as a type map.
