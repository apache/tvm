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
"""The base parser for ir module"""

from ...ir_builder import ir as I
from .._core import Parser, dispatch, doc


@dispatch.register(token="ir", type_name="ClassDef")
def _visit_class_def(self: Parser, node: doc.ClassDef) -> None:
    """The class definition visiting method for ir module.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.ClassDef
        The doc AST class definition node.
    """

    with self.var_table.with_frame():
        with I.ir_module():
            with self.with_dispatch_token("ir"):
                for stmt in node.body:
                    if not isinstance(stmt, doc.FunctionDef):
                        self.visit(stmt)
            for stmt in node.body:
                if isinstance(stmt, doc.FunctionDef):
                    self.visit_tvm_declare_function(stmt)
            with self.with_dispatch_token("ir"):
                for stmt in node.body:
                    if isinstance(stmt, doc.FunctionDef):
                        self.visit(stmt)


@dispatch.register(token="ir", type_name="Assign")
def _visit_assign(_self: Parser, _node: doc.Assign) -> None:
    """The assign visiting method for ir module.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.ClassDef
        The doc AST assign node.
    """


@dispatch.register(token="ir", type_name="Expr")
def _visit_expr(self: Parser, node: doc.Expr) -> None:
    """The expression visiting method for ir module.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.ClassDef
        The doc AST expression node.
    """
    self.eval_expr(node.value)


@dispatch.register(token="default", type_name="Assign")
def visit_assign(self: Parser, node: doc.Assign) -> None:
    if len(node.targets) != 1:
        self.report_error(node, "Consequential assignments like 'a = b = c' are not supported.")
    lhs = node.targets[0]
    rhs = self.eval_expr(node.value)
    self.eval_assign(
        target=lhs, source=rhs, bind_value=lambda _a, _b, _c, value: value, allow_shadowing=True
    )
