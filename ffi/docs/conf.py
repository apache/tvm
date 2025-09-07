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
# -*- coding: utf-8 -*-
import os
import sys

import tomli


os.environ["TVM_FFI_BUILD_DOCS"] = "1"

build_exhale = os.environ.get("BUILD_CPP_DOCS", "0") == "1"


# -- General configuration ------------------------------------------------

# Load version from pyproject.toml
with open("../pyproject.toml", "rb") as f:
    pyproject_data = tomli.load(f)
__version__ = pyproject_data["project"]["version"]

project = "tvm-ffi"

version = __version__
release = __version__

# -- Extensions and extension configurations --------------------------------

extensions = [
    "breathe",
    "myst_parser",
    "nbsphinx",
    "autodocsumm",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.ifconfig",
    "sphinx_copybutton",
    "sphinx_reredirects",
    "sphinx_tabs.tabs",
    "sphinx_toolbox.collapse",
    "sphinxcontrib.httpdomain",
    "sphinxcontrib.mermaid",
]

if build_exhale:
    extensions.append("exhale")

breathe_default_project = "tvm-ffi"

breathe_projects = {"tvm-ffi": "./_build/doxygen/xml"}

exhaleDoxygenStdin = """
INPUT = ../include
PREDEFINED  += TVM_FFI_DLL= TVM_FFI_INLINE= TVM_FFI_EXTRA_CXX_API= __cplusplus=201703

EXCLUDE_SYMBOLS   += *details*  *TypeTraits* std \
                         *use_default_type_traits_v* *is_optional_type_v* *operator* \

EXCLUDE_PATTERNS   += *details.h
ENABLE_PREPROCESSING   = YES
MACRO_EXPANSION        = YES
"""

exhaleAfterTitleDescription = """
This page contains the full API index for the C++ API.
"""

# Setup the exhale extension
exhale_args = {
    "containmentFolder": "reference/cpp/generated",
    "rootFileName": "index.rst",
    "doxygenStripFromPath": "../include",
    "rootFileTitle": "Full API Index",
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": exhaleDoxygenStdin,
    "afterTitleDescription": exhaleAfterTitleDescription,
}
nbsphinx_allow_errors = True
nbsphinx_execute = "never"

autosectionlabel_prefix_document = True
nbsphinx_allow_directives = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "html_image",
    "linkify",
    "attrs_block",
    "substitution",
]

myst_heading_anchors = 3
myst_ref_domains = ["std", "py"]
myst_all_links_external = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12", None),
    "typing_extensions": ("https://typing-extensions.readthedocs.io/en/latest", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

autodoc_mock_imports = ["torch"]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
    "member-order": "bysource",
}

# -- Other Options --------------------------------------------------------

templates_path = []

redirects = {}

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

html_theme = "sphinx_book_theme"
html_title = project
html_copy_source = True
html_last_updated_fmt = ""

html_favicon = "https://tvm.apache.org/images/logo/tvm-logo-square.png"


footer_dropdown = {
    "name": "ASF",
    "items": [
        ("ASF Homepage", "https://apache.org/"),
        ("License", "https://www.apache.org/licenses/"),
        ("Sponsorship", "https://www.apache.org/foundation/sponsorship.html"),
        ("Security", "https://tvm.apache.org/docs/reference/security.html"),
        ("Thanks", "https://www.apache.org/foundation/thanks.html"),
        ("Events", "https://www.apache.org/events/current-event"),
    ],
}


footer_copyright = "Copyright © 2025, Apache Software Foundation"
footer_note = (
    "Apache TVM, Apache, the Apache feather, and the Apache TVM project "
    + "logo are either trademarks or registered trademarks of the Apache Software Foundation."
)


def footer_html():
    # Create footer HTML with two-line layout
    # Generate dropdown menu items
    dropdown_items = ""
    for item_name, item_url in footer_dropdown["items"]:
        dropdown_items += f'<li><a class="dropdown-item" href="{item_url}" target="_blank" style="font-size: 0.9em;">{item_name}</a></li>\n'

    footer_dropdown_html = f"""
  <div class="footer-container" style="margin: 5px 0; font-size: 0.9em; color: #6c757d;">
      <div class="footer-line1" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
          <div class="footer-copyright-short">
              {footer_copyright}
          </div>
          <div class="footer-dropdown">
              <div class="dropdown">
                  <button class="btn btn-link dropdown-toggle" type="button" id="footerDropdown" data-bs-toggle="dropdown"
                  aria-expanded="false" style="font-size: 0.9em; color: #6c757d; text-decoration: none; padding: 0; border: none; background: none;">
                      {footer_dropdown['name']}
                  </button>
                  <ul class="dropdown-menu" aria-labelledby="footerDropdown" style="font-size: 0.9em;">
{dropdown_items}                  </ul>
              </div>
          </div>
      </div>
      <div class="footer-line2" style="font-size: 0.9em; color: #6c757d;">
          {footer_note}
      </div>
  </div>
  """
    return footer_dropdown_html


html_theme_options = {
    "repository_url": "https://github.com/apache/tvm",
    "use_repository_button": True,
    "extra_footer": footer_html(),
}

html_context = {
    "display_github": True,
    "github_user": "apache",
    "github_version": "main",
    "conf_py_path": "/ffi/docs/",
}
