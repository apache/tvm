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
"""Methods and data structures to support dumping HalideIR to Hybrid Script.
This allows users to do quick hack to generated HalideIR and cast it back to
TVM modules.

To enable this feature, you need to build with -DUSE_HYBRID_DUMP=ON.
"""

import ast

from tvm.contrib import utils
from .utils import _internal_assert
from .utils import _is_tvm_arg_types
from .parser import source_to_op


class HybridModule(object):
    """The usage of Hybrid Module is very similar to conventional TVM module,
    but conventional TVM module requires a function body which is already fully
    lowered. This contradicts to the fact that Hybrid Module is originally a text
    format for Phase 0 HalideIR. Thus, a totally separated module is defined."""

    def __init__(self, src=None, name=None):
        """The constructor of this a hybrid module

        Parameters
        ----------
        src : str
            The source code of this module

        name : str
            The name of this module
        """
        self.src_ = self.name = self.func_ = self.root_ = None
        if src is not None:
            temp = utils.tempdir()
            dst = temp.relpath("script.py")
            with open(dst, "w") as f:
                f.write(f"import tvm\n@tvm.te.hybrid.script\n{src}")

            if name is not None:
                self.name = name
            self.load(dst)

    def __call__(self, *args):
        if _is_tvm_arg_types(args):
            return source_to_op(self.root_, args, globals(), {})
        return self.func_(*args)

    def get_source(self):
        return self.src_

    def save(self, path):
        if not path.endswith(".py"):
            path = path + ".py"
        with open(path, "w") as f:
            f.write(self.src_)

    def load(self, path):
        """Load the module from a python file

        Parameters
        ----------
        path : str
            Path to the given python file
        """
        with open(path, "r") as f:
            self.src_ = f.read()

        src = self.src_

        class FindFunc(ast.NodeVisitor):
            """Find the function in module to be loaded module."""

            # pylint: disable=invalid-name
            def __init__(self):
                self.name = None
                self.root = None

            def visit_FunctionDef(self, node):
                _internal_assert(self.name is None, "For now, only one function supported!")
                self.name = node.name
                _internal_assert(self.root is None, "For now, only one function supported!")
                self.root = node

        root = ast.parse(src)
        finder = FindFunc()
        finder.visit(root)
        _internal_assert(finder.name is not None and finder.root is not None, "No function found!")
        if self.name is None:
            self.name = finder.name
        self.root_ = finder.root

        _, local_ = {}, {}
        exec(self.src_, _, local_)  # pylint: disable=exec-used
        local_.pop("tvm")
        assert len(local_) == 1
        self.func_ = list(local_.values())[0]
