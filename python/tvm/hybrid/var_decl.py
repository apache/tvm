"""Determines the declaration, r/w status, and last use of each variable"""
import ast

class PyVariableUsage(ast.NodeVisitor):
    """The vistor class to determine the declaration, r/w status, and last use of each variable"""
    #pylint: disable=invalid-name
    #pylint: disable=missing-docstring
    def __init__(self, args):
        self.status = {}
        self.scope_level = []
        self.args = {}
        for elem in args:
            self.args[elem.name] = elem

    def visit_FunctionDef(self, node):
        self.scope_level.append(node)
        for i in node.body:
            self.visit(i)

    def visit_For(self, node):
        assert isinstance(node.target, ast.Name)
        self.scope_level.append(node)

        for i in node.body:
            self.visit(i)

        self.scope_level.pop()

    def visit_Name(self, node):
        # If it is from the argument list or loop variable, we do not worry about it!
        if node.id in self.args.keys():
            return
        fors = filter(lambda x: isinstance(x, ast.For), self.scope_level)
        fors = list(map(lambda x: x.target.id, fors))
        if node.id in fors:
            return
        # The loop variable cannot be overwritten when iteration
        if isinstance(node.ctx, ast.Store):
            assert node.id not in fors

        if node.id not in self.status.keys():
            # In Python, "first store" indicates "declaration"
            assert isinstance(node.ctx, ast.Store)
            self.status[node.id] = (node, self.scope_level[-1], set())
        else:
            decl, loop, usage = self.status[node.id]
            loop = self.scope_level[-1]
            usage.add(type(node.ctx))
            self.status[node.id] = (decl, loop, usage)

def determine_variable_usage(root, args):
    """The helper function for calling the dedicated visitor."""
    visitor = PyVariableUsage(args)
    visitor.visit(root)
    return visitor.status
