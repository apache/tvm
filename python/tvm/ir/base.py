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
"""Common base structures."""
import tvm._ffi

import tvm.error
import tvm.runtime._ffi_node_api
from tvm.runtime import Object

from . import _ffi_api
from . import json_compact

class Node(Object):
    """Base class of all IR Nodes, implements astext function."""
    def astext(self, show_meta_data=True, annotate=None):
        """Get the text format of the expression.

        Parameters
        ----------
        show_meta_data : bool
            Whether to include meta data section in the text
            if there is meta data.

        annotate: Optional[Object->str]
            Optionally annotate function to provide additional
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
        return _ffi_api.AsText(self, show_meta_data, annotate)

    def __str__(self):
        return self.astext(show_meta_data=False)


@tvm._ffi.register_object("relay.SourceName")
class SourceName(Object):
    """A identifier for a source location.

    Parameters
    ----------
    name : str
        The name of the source.
    """
    def __init__(self, name):
        self.__init_handle_by_constructor__(_ffi_api.SourceName, name)


@tvm._ffi.register_object("relay.Span")
class Span(Object):
    """Specifies a location in a source program.

    Parameters
    ----------
    source : SourceName
        The source name.

    lineno : int
        The line number.

    col_offset : int
        The column offset of the location.
    """
    def __init__(self, source, lineno, col_offset):
        self.__init_handle_by_constructor__(
            _ffi_api.Span, source, lineno, col_offset)


@tvm._ffi.register_object
class EnvFunc(Object):
    """Environment function.

    This is a global function object that can be serialized by its name.
    """
    def __call__(self, *args):
        return _ffi_api.EnvFuncCall(self, *args)

    @property
    def func(self):
        return _ffi_api.EnvFuncGetPackedFunc(self)

    @staticmethod
    def get(name):
        """Get a static env function

        Parameters
        ----------
        name : str
            The name of the function.
        """
        return _ffi_api.EnvFuncGet(name)


def load_json(json_str):
    """Load tvm object from json_str.

    Parameters
    ----------
    json_str : str
        The json string

    Returns
    -------
    node : Object
        The loaded tvm node.
    """

    try:
        return tvm.runtime._ffi_node_api.LoadJSON(json_str)
    except tvm.error.TVMError:
        json_str = json_compact.upgrade_json(json_str)
        return tvm.runtime._ffi_node_api.LoadJSON(json_str)


def save_json(node):
    """Save tvm object as json string.

    Parameters
    ----------
    node : Object
        A TVM object to be saved.

    Returns
    -------
    json_str : str
        Saved json string.
    """
    return tvm.runtime._ffi_node_api.SaveJSON(node)
