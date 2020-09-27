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
"""Unparse meta AST node into a dict"""
# pylint: disable=invalid-name

from typed_ast import ast3 as ast


class MetaUnparser(ast.NodeVisitor):
    """Python AST Visitor to unparse meta AST node into a dict"""

    def visit_Dict(self, node):
        keys = [self.visit(key) for key in node.keys]
        values = [self.visit(value) for value in node.values]
        return dict(zip(keys, values))

    def visit_Tuple(self, node):
        return tuple(self.visit(element) for element in node.elts)

    def visit_List(self, node):
        return [self.visit(element) for element in node.elts]

    def visit_keyword(self, node):
        return node.arg, self.visit(node.value)

    def visit_NameConstant(self, node):
        return node.value

    def visit_Constant(self, node):
        return node.value

    def visit_Num(self, node):
        return node.n

    def visit_Str(self, node):
        return node.s
