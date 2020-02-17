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
# pylint: disable=no-else-return, unidiomatic-typecheck, unused-import
"""The base node types for the Relay language."""
import os
import tvm._ffi

from tvm.runtime import Object
from tvm.ir import SourceName, Span, Node as RelayNode
from . import _make
from . import _expr
from . import _base


__STD_PATH__ = os.path.join(os.path.dirname(os.path.realpath(__file__)), "std")

@tvm._ffi.register_func("tvm.relay.std_path")
def _std_path():
    return __STD_PATH__


def register_relay_node(type_key=None):
    """Register a Relay node type.

    Parameters
    ----------
    type_key : str or cls
        The type key of the node.
    """
    if not isinstance(type_key, str):
        return tvm._ffi.register_object(
            "relay." + type_key.__name__)(type_key)
    return tvm._ffi.register_object(type_key)


def register_relay_attr_node(type_key=None):
    """Register a Relay attribute node.

    Parameters
    ----------
    type_key : str or cls
        The type key of the node.
    """
    if not isinstance(type_key, str):
        return tvm._ffi.register_object(
            "relay.attrs." + type_key.__name__)(type_key)
    return tvm._ffi.register_object(type_key)


@register_relay_node
class Id(Object):
    """Unique identifier(name) used in Var.
       Guaranteed to be stable across all passes.
    """
    def __init__(self):
        raise RuntimeError("Cannot directly construct Id")
