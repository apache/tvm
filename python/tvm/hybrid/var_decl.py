"""Determines the declaration, r/w status, and last use of each variable"""

import ast
import sys
from .intrin import HYBRID_GLOBALS


class PyVariableUsage(ast.NodeVisitor):
    """The vistor class to determine the declaration, r/w status, and last use of each variable"""
    #pylint: disable=invalid-name
    #pylint: disable=missing-docstring
    def __init__(self, args):
        self.status = {}
        self.scope_level = []
        self._args = {}
        self.args = args


    def visit_FunctionDef(self, node):
        self.scope_level.append(node)
        if len(node.args.args) != len(self.args):
            raise ValueError('#arguments passed should be the same as #arguments defined')
        for idx, arg in enumerate(node.args.args):
            _attr = 'id' if sys.version_info[0] < 3 else 'arg' # To make py2 and 3 compatible
            self._args[getattr(arg, _attr)] = self.args[idx]
        for i in node.body:
            self.visit(i)


    def visit_For(self, node):
        if not isinstance(node.target, ast.Name):
            raise ValueError("For's iterator should be an id")
        self.visit(node.iter)
        self.scope_level.append(node)
        for i in node.body:
            self.visit(i)
        self.scope_level.pop()


    def visit_Call(self, node):
        #No function pointer supported so far
        if not isinstance(node.func, ast.Name):
            raise ValueError("Function call should be an id")
        func_id = node.func.id
        if func_id not in list(HYBRID_GLOBALS.keys()) + ['range', 'max', 'min']:
            raise ValueError("Function call id not in intrinsics' list")
        for elem in node.args:
            self.visit(elem)


    def visit_Name(self, node):
        # If it is from the argument list or loop variable, we do not worry about it!
        if node.id in self._args.keys():
            return
        fors = [loop.target.id for loop in self.scope_level if isinstance(loop, ast.For)]
        if node.id in fors:
            return
        # The loop variable cannot be overwritten when iteration
        if isinstance(node.ctx, ast.Store) and node.id in fors:
            raise ValueError("Iter var cannot be overwritten")

        if node.id not in self.status.keys():
            if not isinstance(node.ctx, ast.Store):
                raise ValueError('In Python, "first store" indicates "declaration"')
            self.status[node.id] = (node, self.scope_level[-1], set())
        else:
            decl, loop, usage = self.status[node.id]
            usage.add(type(node.ctx))
            self.status[node.id] = (decl, loop, usage)


def determine_variable_usage(root, args):
    """The helper function for calling the dedicated visitor."""
    visitor = PyVariableUsage(args)
    visitor.visit(root)
    return visitor.status
