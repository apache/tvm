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
# pylint: disable=no-else-return, unidiomatic-typecheck
"""The base node types for the Relay language."""
from __future__ import absolute_import as _abs
from .._ffi.node import NodeBase, register_node as _register_tvm_node
from . import _make
from . import _expr
from . import _base

NodeBase = NodeBase

def register_relay_node(type_key=None):
    """Register a Relay node type.

    Parameters
    ----------
    type_key : str or cls
        The type key of the node.
    """
    if not isinstance(type_key, str):
        return _register_tvm_node(
            "relay." + type_key.__name__)(type_key)
    return _register_tvm_node(type_key)


def register_relay_attr_node(type_key=None):
    """Register a Relay attribute node.

    Parameters
    ----------
    type_key : str or cls
        The type key of the node.
    """
    if not isinstance(type_key, str):
        return _register_tvm_node(
            "relay.attrs." + type_key.__name__)(type_key)
    return _register_tvm_node(type_key)


class RelayNode(NodeBase):
    """Base class of all Relay nodes."""
    def astext(self, show_meta_data=True, annotate=None):
        """Get the text format of the expression.

        Parameters
        ----------
        show_meta_data : bool
            Whether to include meta data section in the text
            if there is meta data.

        annotate: Optional[relay.Expr->str]
            Optional annotate function to provide additional
            information in the comment block.

        Note
        ----
        The meta data section is necessary to fully parse the text format.
        However, it can contain dumps that are big (e.g constant weights),
        so it can be helpful to skip printing the meta data section.

        Returns
        -------
        text : str
            The text format of the expression.
        """
        return _expr.AsText(self, show_meta_data, annotate)

    def set_span(self, span):
        _base.set_span(self, span)

    def __str__(self):
        return self.astext(show_meta_data=False)


@register_relay_node
class Span(RelayNode):
    """Specifies a location in a source program."""

    def __init__(self, source, lineno, col_offset):
        self.__init_handle_by_constructor__(_make.Span, source, lineno, col_offset)

@register_relay_node
class SourceName(RelayNode):
    """A identifier for a source location"""

    def __init__(self, name):
        self.__init_handle_by_constructor__(_make.SourceName, name)

@register_relay_node
class Id(NodeBase):
    """Unique identifier(name) used in Var.
       Guaranteed to be stable across all passes.
    """
    def __init__(self):
        raise RuntimeError("Cannot directly construct Id")
