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

import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

os.environ["TVM_FFI_BUILD_DOCS"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

project = "tvm-tirx"
html_title = project
author = "Apache TVM Community"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_reredirects",
]
autosummary_generate = False
exclude_patterns = ["_build"]
html_theme = "sphinx_book_theme"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_preserve_defaults = True
napoleon_use_rtype = False
napoleon_use_ivar = True
napoleon_custom_sections = ["Usage"]

# Match the FFI reference's optional PyTorch import convention.
autodoc_mock_imports = ["torch"]

html_theme_options = {
    "repository_url": "https://github.com/apache/tvm",
    "repository_branch": "main",
    "path_to_docs": "tirx/docs",
    "use_repository_button": True,
    "show_toc_level": 2,
    "max_navbar_depth": 4,
    "collapse_navbar": True,
}

# Generate from the exact imported source, including lazy namespace APIs.
sys.path.insert(0, str(Path(__file__).with_name("_scripts")))
from generate_reference import build_reference  # noqa: E402

reference_manifest = build_reference()
autosummary_filename_map = reference_manifest["filename_map"]


redirects = {
    "reference/python/generated/tvm_tirx.tirx.script.Let": (
        "tvm_tirx.tirx.script.Let_function.html"
    ),
    "reference/python/generated/tvm_tirx.ir.CallingConv": "tvm_tirx.ir.BaseFunc.html",
    "reference/python/generated/tvm_tirx.ir.CallEffectKind": "tvm_tirx.ir.Call.html",
    "reference/python/generated/tvm_tirx.ir.ForKind": "tvm_tirx.ir.For.html",
    "reference/python/generated/tvm_tirx.tirx.Var": "tvm_tirx.ir.Var.html",
    "reference/python/generated/tvm_tirx.runtime.Tensor": "tvm_tirx.runtime.Tensor_class.html",
    "reference/python/generated/tvm_tirx.ir.UNKNOWN_LOC": ("tvm_tirx.ir.Location.html"),
    "reference/python/generated/tvm_tirx.hexagon": "tvm_tirx.runtime.hexagon.html",
    "reference/python/backend.cuda.transforms": "backend.cuda.transform.html",
}

for api in reference_manifest["canonical"]:
    if api.startswith("tvm_tirx.backend.cuda.transform."):
        redirects["reference/python/generated/" + api.replace(".transform.", ".transforms.")] = (
            api + ".html"
        )


_unresolved_references = Counter()
_reference_aliases = {
    **reference_manifest["implementation_aliases"],
    **reference_manifest["canonical"],
}


def _resolve_reference(app, env, node, contnode):
    """Resolve aliases through canonical ownership without duplicating objects."""
    target = node.get("reftarget", "")
    candidate = _reference_aliases.get(target)
    if candidate is None:
        owner, separator, member = target.rpartition(".")
        if separator and owner in _reference_aliases:
            candidate = _reference_aliases[owner] + "." + member
    if node.get("refdomain") == "py" and candidate and candidate != target:
        resolved = env.domains["py"].resolve_xref(
            env, node["refdoc"], app.builder, "obj", candidate, node, contnode
        )
        if resolved is not None:
            return resolved
    _unresolved_references[target] += 1
    return None


def _copy_apache_config(app, exception):
    # Record diagnostics, including external type names; never suppress warnings.
    output = Path(app.outdir).parent / "unresolved-references.json"
    output.write_text(json.dumps(dict(sorted(_unresolved_references.items())), indent=2) + "\n")
    if exception is None and app.builder.format == "html":
        shutil.copyfile(Path(__file__).with_name("htaccess.in"), Path(app.outdir) / ".htaccess")


def setup(app):
    app.connect("missing-reference", _resolve_reference)
    app.connect("build-finished", _copy_apache_config)
