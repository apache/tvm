"""Node is the base class of all TVM AST.

Normally user do not need to touch this api.
"""
# pylint: disable=unused-import
from __future__ import absolute_import as _abs
from ._ffi.node import NodeBase, register_node

Node = NodeBase
