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

import tvm_ffi
from tvm_ffi import get_global_func, register_object
from tvm_ffi.serialization import from_json_graph_str, to_json_graph_str

from tvm.runtime import Object

from ..libinfo import __version__
from . import _ffi_api, json_compact


class Node(Object):
    """Base class of all IR Nodes."""

    def __repr__(self) -> str:
        from tvm.runtime.script_printer import _script

        try:
            return _script(self, None)
        except Exception:
            return super().__repr__()


@register_object("ir.SourceMap")
class SourceMap(Object):
    def add(self, name, content):
        return get_global_func("SourceMapAdd")(self, name, content)


@register_object("ir.SourceName")
class SourceName(Object):
    """A identifier for a source location.

    Parameters
    ----------
    name : str
        The name of the source.
    """

    def __init__(self, name):
        self.__init_handle_by_constructor__(_ffi_api.SourceName, name)  # type: ignore # pylint: disable=no-member


@register_object("ir.Span")
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

    def __init__(self, source_name, line, end_line, column, end_column):
        self.__init_handle_by_constructor__(
            _ffi_api.Span,
            source_name,
            line,
            end_line,
            column,
            end_column,  # type: ignore # pylint: disable=no-member
        )


@register_object("ir.SequentialSpan")
class SequentialSpan(Object):
    """A sequence of source spans

    This span is specific for an expression, which is from multiple expressions
    after an IR transform.

    Parameters
    ----------
    spans : Array
        The array of spans.
    """

    def __init__(self, spans):
        self.__init_handle_by_constructor__(_ffi_api.SequentialSpan, spans)


@register_object("ir.EnvFunc")
class EnvFunc(Object):
    """Environment function.

    This is a global function object that can be serialized by its name.
    """

    def __call__(self, *args):
        return _ffi_api.EnvFuncCall(self, *args)  # type: ignore # pylint: disable=no-member

    @property
    def func(self):
        return _ffi_api.EnvFuncGetFunction(self)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def get(name):
        """Get a static env function

        Parameters
        ----------
        name : str
            The name of the function.
        """
        return _ffi_api.EnvFuncGet(name)  # type: ignore # pylint: disable=no-member


def load_json(json_str) -> Object:
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

    json_str = json_compact.upgrade_json(json_str)
    return from_json_graph_str(json_str)


def save_json(node) -> str:
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
    return to_json_graph_str(node, {"tvm_version": __version__})


def assert_structural_equal(lhs, rhs, map_free_vars=False):
    """Assert lhs and rhs are structurally equal to each other.

    Parameters
    ----------
    lhs : Object
        The left operand.

    rhs : Object
        The left operand.

    map_free_vars : bool
        Whether or not shall we map free vars that does
        not bound to any definitions as equal to each other.

    Raises
    ------
    ValueError : if assertion does not hold.

    See Also
    --------
    tvm_ffi.structural_equal
    """
    first_mismatch = tvm_ffi.get_first_structural_mismatch(lhs, rhs, map_free_vars)
    if first_mismatch is not None:
        from tvm.runtime.script_printer import (  # pylint: disable=import-outside-toplevel
            PrinterConfig,
            _script,
        )

        lhs_path, rhs_path = first_mismatch
        lhs_script = _script(lhs, PrinterConfig(syntax_sugar=False, path_to_underline=[lhs_path]))
        rhs_script = _script(rhs, PrinterConfig(syntax_sugar=False, path_to_underline=[rhs_path]))
        raise ValueError(
            f"StructuralEqual check failed, caused by lhs at {lhs_path}:\n"
            f"{lhs_script}\n"
            f"and rhs at {rhs_path}:\n"
            f"{rhs_script}"
        )


def deprecated(
    method_name: str,
    new_method_name: str,
):
    """A decorator to indicate that a method is deprecated

    Parameters
    ----------
    method_name : str
        The name of the method to deprecate
    new_method_name : str
        The name of the new method to use instead
    """
    import functools  # pylint: disable=import-outside-toplevel
    import warnings  # pylint: disable=import-outside-toplevel

    def _deprecate(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            warnings.warn(
                f"{method_name} is deprecated, use {new_method_name} instead",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return _wrapper

    return _deprecate
