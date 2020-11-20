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

from synr import Transformer


class MetaUnparser(Transformer):
    """Python AST Visitor to unparse meta AST node into a dict"""

    def transform(self, node):
        method = "transform_" + node.__class__.__name__
        visitor = getattr(self, method, None)
        if visitor is None:
            self.error(f"Unexpected node type {type(node)} when parsing __tvm_meta__", node.span)
        return visitor(node)

    def transform_DictLiteral(self, node):
        keys = [self.visit(key) for key in node.keys]
        values = [self.visit(value) for value in node.values]
        return dict(zip(keys, values))

    def transform_Tuple(self, node):
        return tuple(self.visit(element) for element in node.elts)

    def transform_ArrayLiteral(self, node):
        return [self.visit(element) for element in node.elts]

    def transform_Constant(self, node):
        return node.value
