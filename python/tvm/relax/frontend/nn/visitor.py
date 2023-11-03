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
"""The visitor and mutator infra for nn.Module."""
from typing import Any

from . import core as nn


class Mutator:
    """The mutator for nn.Module transform. Users can override the `visit_*` methods
    to apply transform in different structures, or even override the `visit` method
    to change the logic of traversal."""

    def visit_module(self, name: str, node: nn.Module) -> Any:
        """The base visiting method for mutation of nn.Module nodes.

        Parameters
        ----------
        name : str
            The name of the current node in parent's attribute.

        node : nn.Module
            The current node of nn.Module to mutate.

        Returns
        ------
        ret_node: Any
            The new node to replace current node.
        """
        return self.visit(name, node)

    def visit_effect(self, name: str, node: nn.Parameter) -> Any:
        """The base visiting method for mutation of nn.Parameter nodes.

        Parameters
        ----------
        name : str
            The name of the current node in parent's attribute.

        node : nn.Parameter
            The current node of nn.Parameter to mutate.

        Returns
        ------
        ret_node: Any
            The new node to replace current node.
        """
        return self.visit(name, node)

    def visit_param(self, name: str, node: nn.Effect) -> Any:
        """The base visiting method for mutation of nn.Effect nodes.

        Parameters
        ----------
        name : str
            The name of the current node in parent's attribute.

        node : nn.Effect
            The current node of nn.Effect to mutate.

        Returns
        ------
        ret_node: Any
            The new node to replace current node.
        """
        return self.visit(name, node)

    def visit_modulelist(self, name: str, node: nn.ModuleList) -> Any:
        """The base visiting method for mutation of nn.ModuleList nodes.

        Parameters
        ----------
        name : str
            The name of the current node in parent's attribute.

        node : nn.ModuleList
            The current node of nn.MoModuleListdule to mutate.

        Returns
        ------
        ret_node: Any
            The new node to replace current node.
        """
        return self.visit(name, node)

    def visit(self, name: str, node: Any) -> Any:
        """The base dispatching method for visiting of all nodes.

        Parameters
        ----------
        name : str
            The name of the current node in parent's attribute.

        node : Any
            The current node to visit.

        Returns
        ------
        ret_node: Any
            The new node to replace current node.
        """

        def _get_child_name(parent: str, child: str) -> str:
            """Get the name of the child node/key given the parent's name."""
            if parent == "":
                # in the top level of the module
                return child
            else:
                return f"{parent}.{child}"

        if isinstance(node, nn.ModuleList):
            for i in range(len(node)):
                if isinstance(node[i], nn.ModuleList):
                    node[i] = self.visit_modulelist(f"{name}.{i}", node[i])
                elif isinstance(node[i], nn.Module):
                    node[i] = self.visit_module(f"{name}.{i}", node[i])
                elif isinstance(node[i], nn.Effect):
                    node[i] = self.visit_effect(f"{name}.{i}", node[i])
                elif isinstance(node[i], nn.Parameter):
                    node[i] = self.visit_param(f"{name}.{i}", node[i])
        else:
            for key, value in node.__dict__.items():
                if isinstance(value, nn.ModuleList):
                    setattr(node, key, self.visit_modulelist(_get_child_name(name, key), value))
                elif isinstance(value, nn.Module):
                    setattr(node, key, self.visit_module(_get_child_name(name, key), value))
                elif isinstance(value, nn.Effect):
                    setattr(node, key, self.visit_effect(_get_child_name(name, key), value))
                elif isinstance(value, nn.Parameter):
                    setattr(node, key, self.visit_param(_get_child_name(name, key), value))
        return node
