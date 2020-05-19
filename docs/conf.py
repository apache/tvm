# -*- coding: utf-8 -*-

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

#
# documentation build configuration file, created by
# sphinx-quickstart on Thu Jul 23 19:40:08 2015.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.
import sys
import inspect
import os, subprocess
import shlex
import recommonmark
import sphinx_gallery
from recommonmark.parser import CommonMarkParser
from recommonmark.transform import AutoStructify

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../python/'))
sys.path.insert(0, os.path.join(curr_path, '../topi/python'))
sys.path.insert(0, os.path.join(curr_path, '../vta/python'))

# -- General configuration ------------------------------------------------

# General information about the project.
project = u'tvm'
author = u'Apache Software Foundation'
copyright = u'2020, %s' % author
github_doc_root = 'https://github.com/apache/incubator-tvm/tree/master/docs/'

# add markdown parser
CommonMarkParser.github_doc_root = github_doc_root
source_parsers = {
    '.md': CommonMarkParser
}
os.environ['TVM_BUILD_DOC'] = '1'
# Version information.
import tvm
from tvm import te
version = tvm.__version__
release = tvm.__version__

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_gallery.gen_gallery',
    'autodocsumm'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = ['.rst', '.md']

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# generate autosummary even if no references
autosummary_generate = True

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# The reST default role (used for this markup: `text`) to use for all
# documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# The theme is set by the make target
html_theme = os.environ.get('TVM_THEME', 'rtd')

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
# only import rtd theme and set it if want to build docs locally
if not on_rtd and html_theme == 'rtd':
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'analytics_id': 'UA-75982049-2',
    'logo_only': True,
}

html_logo = "_static/img/tvm-logo-small.png"

html_favicon = "_static/img/tvm-logo-square.png"


# Output file base name for HTML help builder.
htmlhelp_basename = project + 'doc'

# -- Options for LaTeX output ---------------------------------------------
latex_elements = {
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  (master_doc, '%s.tex' % project, project,
   author, 'manual'),
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'matplotlib': ('https://matplotlib.org/', None),
}

from sphinx_gallery.sorting import ExplicitOrder

examples_dirs = ["../tutorials/", "../vta/tutorials/"]
gallery_dirs = ["tutorials", "vta/tutorials"]

subsection_order = ExplicitOrder(
    ['../tutorials/frontend',
     '../tutorials/language',
     '../tutorials/optimize',
     '../tutorials/autotvm',
     '../tutorials/dev',
     '../tutorials/topi',
     '../tutorials/deployment',
     '../vta/tutorials/frontend',
     '../vta/tutorials/optimize',
     '../vta/tutorials/autotvm'])

sphinx_gallery_conf = {
    'backreferences_dir': 'gen_modules/backreferences',
    'doc_module': ('tvm', 'numpy'),
    'reference_url': {
        'tvm': None,
        'matplotlib': 'https://matplotlib.org/',
        'numpy': 'https://numpy.org/doc/stable'
    },
    'examples_dirs': examples_dirs,
    'gallery_dirs': gallery_dirs,
    'subsection_order': subsection_order,
    'filename_pattern': os.environ.get("TVM_TUTORIAL_EXEC_PATTERN", ".py"),
    'find_mayavi_figures': False,
    'download_all_examples': False,
    "min_reported_time": 60,
    'expected_failing_examples': []
}

autodoc_default_options = {
    'member-order': 'bysource',
}

# Maps the original namespace to list of potential modules
# that we can import alias from.
tvm_alias_check_map = {
    "tvm.te": ["tvm.tir"],
    "tvm.tir": ["tvm.ir", "tvm.runtime"],
    "tvm.relay": ["tvm.ir", "tvm.tir"],
}

def update_alias_docstring(name, obj, lines):
    """Update the docstring of alias functions.

    This function checks if the obj is an alias of another documented object
    in a different module.

    If it is an alias, then it will append the alias information to the docstring.

    Parameters
    ----------
    name : str
        The full name of the object in the doc.

    obj : object
        The original object.

    lines : list
        The docstring lines, need to be modified inplace.
    """
    arr = name.rsplit(".", 1)
    if len(arr) != 2:
        return
    target_mod, target_name = arr

    if target_mod not in tvm_alias_check_map:
        return
    if not hasattr(obj, "__module__"):
        return
    obj_mod = obj.__module__

    for amod in tvm_alias_check_map[target_mod]:
        if not obj_mod.startswith(amod):
            continue

        if hasattr(sys.modules[amod], target_name):
            obj_type = ":py:func" if callable(obj) else ":py:class"
            lines.append(
                ".. rubric:: Alias of %s:`%s.%s`" % (obj_type, amod, target_name))


def process_docstring(app, what, name, obj, options, lines):
    """Sphinx callback to process docstring"""
    if callable(obj) or inspect.isclass(obj):
        update_alias_docstring(name, obj, lines)


def setup(app):
    app.connect('autodoc-process-docstring', process_docstring)
    app.add_css_file('css/tvm_theme.css')
    app.add_config_value('recommonmark_config', {
        'url_resolver': lambda url: github_doc_root + url,
        'auto_doc_ref': True
            }, True)
    app.add_transform(AutoStructify)
