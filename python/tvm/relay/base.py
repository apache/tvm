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
from tvm.ir import Node as RelayNode
from tvm.ir import SourceName, Span, SequentialSpan
from tvm.runtime import Object

from . import _ffi_api

__STD_PATH__ = os.path.join(os.path.dirname(os.path.realpath(__file__)), "std")


def pretty_print(obj: Object) -> None:
    """Pretty print the object."""
    return _ffi_api.PrettyPrint(obj)  # type: ignore # pylint: disable=no-member


def astext(obj: Object, show_meta_data=True, annotate=None):
    """Get the text format of the expression.

    Parameters
    ----------
    obj : Object
        The object to be printed.
    show_meta_data : bool
        Whether to include meta data section in the text
        if there is meta data.
    annotate: Optional[Object->str]
        Optionally annotate function to provide additional
        information in the comment block.

    Returns
    -------
    text : str
        The text format of the expression.

    Notes
    -----
    The meta data section is necessary to fully parse the text format.
    However, it can contain dumps that are big (e.g constant weights),
    so it can be helpful to skip printing the meta data section.
    """
    return _ffi_api.AsText(obj, show_meta_data, annotate)  # type: ignore # pylint: disable=no-member


@tvm._ffi.register_func("tvm.relay.std_path")
def _std_path():
    return __STD_PATH__


@tvm._ffi.register_object("relay.Id")
class Id(Object):
    """Unique identifier(name) used in Var.
    Guaranteed to be stable across all passes.
    """

    def __init__(self):
        raise RuntimeError("Cannot directly construct Id")
