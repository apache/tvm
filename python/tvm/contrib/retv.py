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
"""Relay Expression Terminal Visualization (RETV), visualizing Relay Expression on Terminal"""
from tvm import relay
from tvm import ir
from tvm.relay import Tuple


class Node:
    """Base unit of a relay IR visualization node.

    Parameters
    ----------
    expr : expr
        Relay IR expression.

    name : str
        The name of the relay IR node.

    parent : Node
        The parent node of the relay IR node.

    is_last : bool
        Whether the node is the last within same level nodes.
    """

    def __init__(self, expr, name, parent, is_last):
        self.expr = expr
        self.name = name
        self.parent = parent
        self.is_last = is_last
        self.children = []
        self.prefix = ""


@ir.transform.module_pass(opt_level=1)
class ASTVisualization:
    """To visualize the relay IR graph on terminal."""

    def __init__(self):
        self.output = []

    def get_output(self):
        """
        Returns
        -------
        output: str
          The graph.
        """
        output = "== The AST view of the IRModule is ==\n"
        for subout in self.output[1:]:
            output += subout + "\n"
        output += self.output[0] + "\n"  # "main" function
        return output

    def transform_module(self, mod, ctx):
        """A module pass"""

        class ASTVisitor(relay.ExprVisitor):
            """
            A visitor over Expr.

            It traverses the AST recursively, and each node information into a sequence.
            """

            def __init__(self):
                super(ASTVisitor, self).__init__()
                self.sequence = []
                self.parent_stack = []
                self.last_stack = []
                self.current_subgraph = ""

            def seen_node(self, new_node, expr):
                """Record those seen expression"""
                self.sequence.append(new_node)
                self.parent_stack.append(new_node)
                for expr_child in self.memo_map[expr].children:
                    new_node = Node(
                        expr=expr_child,
                        name=self.memo_map[expr_child].name,
                        parent=self.parent_stack[-1],
                        is_last=self.memo_map[expr_child].is_last,
                    )
                    self.seen_node(new_node, expr_child)
                self.parent_stack.pop()

            def visit(self, expr):
                if expr in self.memo_map:
                    new_node = Node(
                        expr=expr,
                        name=self.memo_map[expr].name,
                        parent=self.parent_stack[-1],
                        is_last=self.last_stack[-1],
                    )
                    self.seen_node(new_node, expr)
                else:
                    super(ASTVisitor, self).visit(expr)

            def visit_tuple(self, tup):
                node = Node(
                    expr=tup,
                    name="(tuple)",
                    parent=self.parent_stack[-1],
                    is_last=self.last_stack[-1],
                )
                self.sequence.append(node)
                node.parent.children.append(tup)
                self.parent_stack.append(node)
                for i, x in enumerate(tup.fields):
                    if i == len(tup.fields) - 1:
                        self.last_stack.append(True)
                    else:
                        self.last_stack.append(False)
                    self.visit(x)
                    self.last_stack.pop()
                self.parent_stack.pop()
                return node

            def visit_var(self, var):
                node = Node(
                    expr=var,
                    name=var.name_hint,
                    parent=self.parent_stack[-1],
                    is_last=self.last_stack[-1],
                )
                self.sequence.append(node)
                node.parent.children.append(var)
                return node

            def visit_function(self, fn):
                if len(self.sequence) == 0:  # entry function call
                    layer_name = "@" + self.current_subgraph + "(" + str(fn.params) + ")"
                    self.parent_stack = [None]
                    self.last_stack = [True]
                else:
                    layer_name = "Function_" + str(fn.__hash__()) + "(" + str(fn.params) + ")"

                node = Node(
                    expr=fn,
                    name=layer_name,
                    parent=self.parent_stack[-1],
                    is_last=self.last_stack[-1],
                )
                self.sequence.append(node)
                if node.parent is not None:
                    node.parent.children.append(fn)

                is_last = True
                self.last_stack.append(is_last)
                self.parent_stack.append(node)
                self.visit(fn.body)
                self.parent_stack.pop()
                self.last_stack.pop()
                return node

            def visit_call(self, call):
                layer_name = "(call)"
                node = Node(
                    expr=call,
                    name=layer_name,
                    parent=self.parent_stack[-1],
                    is_last=self.last_stack[-1],
                )
                self.sequence.append(node)
                node.parent.children.append(call)
                self.parent_stack.append(node)
                self.last_stack.append(len(call.args) == 0)
                self.visit(call.op)
                self.last_stack.pop()

                for i, arg in enumerate(call.args):
                    is_last = i == len(call.args) - 1
                    self.last_stack.append(is_last)
                    self.visit(arg)
                    self.last_stack.pop()
                self.parent_stack.pop()
                return node

            def visit_constant(self, const):
                node = Node(
                    expr=const,
                    name=const,
                    parent=self.parent_stack[-1],
                    is_last=self.last_stack[-1],
                )
                self.sequence.append(node)
                node.parent.children.append(const)
                return node

            def visit_if(self, i):
                layer_name = "if(cond, true, false)"
                node = Node(
                    expr=i,
                    name=layer_name,
                    parent=self.parent_stack[-1],
                    is_last=self.last_stack[-1],
                )
                node.parent.children.append(node)
                self.sequence.append(node)
                self.parent_stack.append(node)
                self.last_stack.append(False)
                self.visit(i.cond)
                self.last_stack[-1] = False
                self.visit(i.true_branch)
                self.last_stack[-1] = True
                self.visit(i.false_branch)
                self.last_stack.pop()
                self.parent_stack.pop()
                return node

            def visit_let(self, let):
                layer_name = "let(var, val, body)"
                node = Node(
                    expr=let,
                    name=layer_name,
                    parent=self.parent_stack[-1],
                    is_last=self.last_stack[-1],
                )
                self.sequence.append(node)
                node.parent.children.append(let)
                self.parent_stack.append(node)
                self.last_stack.append(False)
                self.visit(let.var)
                self.last_stack[-1] = False
                self.visit(let.value)
                self.last_stack[-1] = True
                self.visit(let.body)
                self.last_stack.pop()
                self.parent_stack.pop()
                return node

            def visit_global_var(self, gv):
                layer_name = "@" + str(gv.name_hint)
                node = Node(
                    expr=gv,
                    name=layer_name,
                    parent=self.parent_stack[-1],
                    is_last=self.last_stack[-1],
                )
                self.sequence.append(node)
                node.parent.children.append(gv)
                return node

            def visit_op(self, op):
                node = Node(
                    expr=op,
                    name=str(op.name),
                    parent=self.parent_stack[-1],
                    is_last=self.last_stack[-1],
                )
                self.sequence.append(node)
                node.parent.children.append(op)
                return node

            def prettyprint(self):
                """Prettyprint the result"""

                if len(self.sequence) <= 1:
                    raise RuntimeError("It is an empty IRmodule")
                res = ""
                res += self.sequence[0].name + "\n"
                for node in self.sequence[1:]:
                    if node.parent is None:
                        part_a = ""
                        part_b = ""
                    else:
                        part_a = node.parent.prefix[:-3]
                        part_b = " " * 3 if node.parent.is_last else "|  "
                    part_c = "`--" if node.is_last else "|--"
                    if isinstance(node.expr, Tuple):
                        name = ""
                        for child in node.children:
                            name += str(self.memo_map[child].name) + ", "
                        name = "(" + name[:-2] + ")"
                        node.name = name
                    node.prefix = part_a + part_b + part_c
                    res += node.prefix + str(node.name) + "\n"
                return res

        printer = ASTVisitor()
        printer.current_subgraph = "main"
        printer.visit(mod["main"])
        self.output.append(printer.prettyprint())
        for subgraph in mod.get_global_vars():
            name = subgraph.name_hint
            if name != "main":
                printer.sequence = []
                printer.parent_stack = []
                printer.last_stack = []
                printer.current_subgraph = name
                printer.visit(mod[name])
                self.output.append(printer.prettyprint())
        return mod
