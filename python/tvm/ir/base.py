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
from tvm_ffi import register_object
from tvm_ffi.serialization import from_json_graph_str, to_json_graph_str

from tvm.runtime import Object

from ..libinfo import __version__
from . import _ffi_api, json_compact


class Scriptable:
    """Marker for display methods installed by the TVMScript owner."""


class Node(Object):
    """Base class of all IR Nodes."""


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
        lhs_path, rhs_path = first_mismatch
        # Diagnostics use the same display policy as Object.script(), including
        # dialect selection and commented imports. The unbound method also
        # accepts IR objects that do not inherit the convenience mixin.
        script = getattr(Scriptable, "script", None)
        lhs_script = script(lhs, path_to_underline=[lhs_path]) if script else repr(lhs)
        rhs_script = script(rhs, path_to_underline=[rhs_path]) if script else repr(rhs)
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
