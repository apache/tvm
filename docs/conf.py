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

import os
import sys
from pathlib import Path
from types import ModuleType

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
]
autosummary_generate = True
templates_path = ["_templates"]
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

# Optional kernel dependencies are not needed to render their Python APIs.
autodoc_mock_imports = ["torch", "triton"]
# The Triton adapter checks this minimum version at import time. Its remaining
# imports are handled by autodoc's mocks; no Triton code executes during a build.
_triton = ModuleType("triton")
_triton.__version__ = "3.3.0"
_triton.__path__ = []
sys.modules.setdefault("triton", _triton)

html_theme_options = {
    "repository_url": "https://github.com/apache/tvm",
    "repository_branch": "tirx-standalone-root",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "show_toc_level": 2,
    "max_navbar_depth": 4,
    "collapse_navbar": True,
}

# Case-insensitive filesystems need distinct class/function stub names.
autosummary_filename_map = {
    "tvm_tirx.ir.Var": "tvm_tirx.ir.Var_class",
    "tvm_tirx.runtime.Tensor": "tvm_tirx.runtime.Tensor_class",
    "tvm_tirx.runtime.tensor": "tvm_tirx.runtime.tensor_function",
    "tvm_tirx.ir.prim.Div": "tvm_tirx.ir.prim.Div_class",
    "tvm_tirx.ir.prim.div": "tvm_tirx.ir.prim.div_function",
    "tvm_tirx.ir.prim.FloorDiv": "tvm_tirx.ir.prim.FloorDiv_class",
    "tvm_tirx.ir.prim.floordiv": "tvm_tirx.ir.prim.floordiv_function",
    "tvm_tirx.ir.prim.FloorMod": "tvm_tirx.ir.prim.FloorMod_class",
    "tvm_tirx.ir.prim.floormod": "tvm_tirx.ir.prim.floormod_function",
    "tvm_tirx.ir.prim.Max": "tvm_tirx.ir.prim.Max_class",
    "tvm_tirx.ir.prim.max": "tvm_tirx.ir.prim.max_function",
    "tvm_tirx.ir.prim.Min": "tvm_tirx.ir.prim.Min_class",
    "tvm_tirx.ir.prim.min": "tvm_tirx.ir.prim.min_function",
    "tvm_tirx.support.utils.FileLock": "tvm_tirx.support.utils.FileLock_class",
    "tvm_tirx.support.utils.filelock": "tvm_tirx.support.utils.filelock_function",
    "tvm_tirx.runtime.Device": "tvm_tirx.runtime.Device_class",
}
