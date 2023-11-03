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
"""Determines the declaration, r/w status, and last use of each variable"""

import ast
import sys
from .runtime import HYBRID_GLOBALS
from .utils import _internal_assert


class PyVariableUsage(ast.NodeVisitor):
    """The vistor class to determine the declaration, r/w status, and last use of each variable"""

    # pylint: disable=invalid-name
    # pylint: disable=missing-docstring
    def __init__(self, args, symbols, closure_vars):
        self.status = {}
        self.scope_level = []
        self._args = {}
        self.args = args
        self.aug_assign_ = False
        self.symbols = symbols
        self.closure_vars = closure_vars

    def visit_FunctionDef(self, node):
        self.scope_level.append(node)
        _internal_assert(
            len(node.args.args) == len(self.args),
            "#arguments passed should be the same as #arguments defined",
        )
        for idx, arg in enumerate(node.args.args):
            _attr = "id" if sys.version_info[0] < 3 else "arg"  # To make py2 and 3 compatible
            self._args[getattr(arg, _attr)] = self.args[idx]
        for i in node.body:
            self.visit(i)

    def visit_For(self, node):
        _internal_assert(isinstance(node.target, ast.Name), "For's iterator should be an id")
        self.visit(node.iter)
        self.scope_level.append(node)
        for i in node.body:
            self.visit(i)
        self.scope_level.pop()

    def visit_Call(self, node):
        # No function pointer supported so far
        _internal_assert(isinstance(node.func, ast.Name), "Function call should be an id")
        func_id = node.func.id
        _internal_assert(
            func_id
            in list(HYBRID_GLOBALS.keys())
            + ["range", "max", "min", "len"]
            + list(self.symbols.keys()),
            "Function call id " + func_id + " not in intrinsics' list",
        )
        for elem in node.args:
            self.visit(elem)

    def visit_AugAssign(self, node):
        self.aug_assign_ = True
        self.generic_visit(node)
        self.aug_assign_ = False

    def visit_Name(self, node):
        # If it is True or False, we do not worry about it!
        if sys.version_info[0] == 2 and node.id in ["True", "False"]:
            return
        # If it is from the argument list or loop variable, we do not worry about it!
        if node.id in self._args.keys():
            return
        fors = [loop.target.id for loop in self.scope_level if isinstance(loop, ast.For)]
        if node.id in fors:
            return
        # The loop variable cannot be overwritten when iteration
        _internal_assert(
            not isinstance(node.ctx, ast.Store) or node.id not in fors,
            "Iter var cannot be overwritten",
        )

        if node.id not in self.status.keys():
            # It is a captured value in closure
            if node.id in self.closure_vars:
                try:
                    ast.literal_eval(str(self.closure_vars[node.id]))
                except ValueError:
                    raise ValueError("Only support capturing constant values in closure")
                return

            _internal_assert(isinstance(node.ctx, ast.Store), f"Undeclared variable {node.id}")
            if self.aug_assign_:
                raise ValueError('"First store" cannot be an AugAssign')
            self.status[node.id] = (node, self.scope_level[-1], set())
        else:
            decl, loop, usage = self.status[node.id]
            usage.add(type(node.ctx))
            _internal_assert(
                loop in self.scope_level, f"{node.id} is used out of the scope it is defined!"
            )
            self.status[node.id] = (decl, loop, usage)


def determine_variable_usage(root, args, symbols, closure_vars):
    """The helper function for calling the dedicated visitor."""
    visitor = PyVariableUsage(args, symbols, closure_vars)
    visitor.visit(root)
    return visitor.status
