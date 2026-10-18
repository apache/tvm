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
"""Configuration of TVMScript printer"""

from collections.abc import Sequence

import tvm_ffi
from tvm_ffi import register_object
from tvm_ffi.access_path import AccessPath

from tvm_tirx.ir.base import register_script_printer

from . import _ffi_api


@register_object("script.PrinterConfig")
class PrinterConfig(tvm_ffi.Object):
    """Configuration of TVMScript printer.

    The ``int_dtype`` and ``float_dtype`` arguments populate the string-valued
    ``ir.int_dtype`` and ``ir.float_dtype`` entries in ``extra_config``.
    Explicit entries in ``extra_config`` override those arguments.
    """

    binding_names: Sequence[str]
    show_meta: bool
    verbose_expr: bool
    indent_spaces: int
    print_line_numbers: bool
    num_context_lines: int
    path_to_underline: list[AccessPath] | None
    path_to_annotate: dict[AccessPath, str] | None
    obj_to_underline: list[tvm_ffi.Object] | None
    obj_to_annotate: dict[tvm_ffi.Object, str] | None
    extra_config: dict

    def __init__(
        self,
        *,
        name: str | None = None,
        show_meta: bool = False,
        int_dtype: str = "int32",
        float_dtype: str = "void",
        verbose_expr: bool = False,
        indent_spaces: int = 4,
        print_line_numbers: bool = False,
        num_context_lines: int | None = None,
        path_to_underline: list[AccessPath] | None = None,
        path_to_annotate: dict[AccessPath, str] | None = None,
        obj_to_underline: list[tvm_ffi.Object] | None = None,
        obj_to_annotate: dict[tvm_ffi.Object, str] | None = None,
        extra_config: dict | None = None,
    ) -> None:
        if num_context_lines is None:
            num_context_lines = -1
        cfg: dict = {
            "show_meta": show_meta,
            "ir.int_dtype": int_dtype,
            "ir.float_dtype": float_dtype,
            "verbose_expr": verbose_expr,
            "indent_spaces": indent_spaces,
            "print_line_numbers": print_line_numbers,
            "num_context_lines": num_context_lines,
            "path_to_underline": path_to_underline,
            "path_to_annotate": path_to_annotate,
            "obj_to_underline": obj_to_underline,
            "obj_to_annotate": obj_to_annotate,
        }

        if name is not None:
            cfg["name"] = name
        if extra_config is not None:
            cfg["extra_config"] = extra_config
        self.__init_handle_by_constructor__(
            _ffi_api.PrinterConfig,
            cfg,  # type: ignore # pylint: disable=no-member
        )


def _script(obj: tvm_ffi.Object, config: PrinterConfig) -> str:
    return _ffi_api.Script(obj, config)  # type: ignore # pylint: disable=no-member


class _ScriptableMethods:
    """A base class that enables the script() and show() method."""

    def script(
        self,
        *,
        name: str | None = None,
        show_meta: bool = False,
        int_dtype: str = "int32",
        float_dtype: str = "void",
        verbose_expr: bool = False,
        indent_spaces: int = 4,
        print_line_numbers: bool = False,
        num_context_lines: int = -1,
        path_to_underline: list[AccessPath] | None = None,
        path_to_annotate: dict[AccessPath, str] | None = None,
        obj_to_underline: list[tvm_ffi.Object] | None = None,
        obj_to_annotate: dict[tvm_ffi.Object, str] | None = None,
        extra_config: dict | None = None,
    ) -> str:
        """Print TVM IR into TVMScript text format

        Parameters
        ----------
        name : Optional[str] = None
            The name of the object
        show_meta : bool = False
            Whether to print the meta data of the object
        int_dtype : str = "int32"
            The default data type of integer
        float_dtype : str = "void"
            The default data type of float
        verbose_expr : bool = False
            Whether to print the detailed definition of each variable in the expression
        indent_spaces : int = 4
            The number of spaces for indentation
        print_line_numbers : bool = False
            Whether to print line numbers
        num_context_lines : int = -1
            The number of lines of context to print before and after the line to underline.
        path_to_underline : Optional[List[AccessPath]] = None
            Object path to be underlined
        path_to_annotate : Optional[Dict[AccessPath, str]] = None
            Object path to be annotated
        obj_to_underline : Optional[List[tvm_ffi.Object]] = None
            Object to be underlined
        obj_to_annotate : Optional[Dict[tvm_ffi.Object, str]] = None
            Object to be annotated
        extra_config : Optional[dict] = None
            Dialect-specific configuration passed through to PrinterConfig.extra_config,
            including ``ir.prefix`` for the IR namespace alias (default ``"I"``).
            Keys are conventionally namespaced as "<dialect>.<knob>", e.g.
            ``{"tirx.prefix": "T"}``.

        Returns
        -------
        script : str
            The TVM Script of the given TVM IR

        """
        merged_extra: dict = {}
        if extra_config is not None:
            merged_extra.update(extra_config)
        # Keep the historical interactive display header. Direct printer calls
        # retain executable imports for standalone parser round-trips.
        merged_extra.setdefault("ir.comment_imports", True)

        return _script(
            self,
            PrinterConfig(
                name=name,
                show_meta=show_meta,
                int_dtype=int_dtype,
                float_dtype=float_dtype,
                verbose_expr=verbose_expr,
                indent_spaces=indent_spaces,
                print_line_numbers=print_line_numbers,
                num_context_lines=num_context_lines,
                path_to_underline=path_to_underline,
                path_to_annotate=path_to_annotate,
                obj_to_underline=obj_to_underline,
                obj_to_annotate=obj_to_annotate,
                extra_config=merged_extra,
            ),
        )

    def show(
        self,
        style: str | None = None,
        *,
        name: str | None = None,
        show_meta: bool = False,
        int_dtype: str = "int32",
        float_dtype: str = "void",
        verbose_expr: bool = False,
        indent_spaces: int = 4,
        print_line_numbers: bool = False,
        num_context_lines: int = -1,
        path_to_underline: list[AccessPath] | None = None,
        path_to_annotate: dict[AccessPath, str] | None = None,
        obj_to_underline: list[tvm_ffi.Object] | None = None,
        obj_to_annotate: dict[tvm_ffi.Object, str] | None = None,
        extra_config: dict | None = None,
    ) -> None:
        """Print highlighted TVMScript.

        Parameters
        ----------
        style : str, optional
            Pygmentize printing style, auto-detected if None.  See
            `tvm_tirx.script.printer.highlight.cprint` for more details.

        name : Optional[str] = None
            The name of the object
        show_meta : bool = False
            Whether to print the meta data of the object
        int_dtype : str = "int32"
            The default data type of integer
        float_dtype : str = "void"
            The default data type of float
        verbose_expr : bool = False
            Whether to print the detailed definition of each variable in the expression
        indent_spaces : int = 4
            The number of spaces for indentation
        print_line_numbers : bool = False
            Whether to print line numbers
        num_context_lines : int = -1
            The number of lines of context to print before and after the line to underline.
        path_to_underline : Optional[List[AccessPath]] = None
            Object path to be underlined
        path_to_annotate : Optional[Dict[AccessPath, str]] = None
            Object path to be annotated
        obj_to_underline : Optional[List[tvm_ffi.Object]] = None
            Object to be underlined
        obj_to_annotate : Optional[Dict[tvm_ffi.Object, str]] = None
            Object to be annotated
        extra_config : Optional[dict] = None
            Dialect-specific configuration passed through to PrinterConfig.extra_config,
            including ``ir.prefix`` for the IR namespace alias (default ``"I"``).

        """
        from tvm_tirx.script.printer.highlight import (
            cprint,  # pylint: disable=import-outside-toplevel
        )

        cprint(
            self.script(
                name=name,
                show_meta=show_meta,
                int_dtype=int_dtype,
                float_dtype=float_dtype,
                verbose_expr=verbose_expr,
                indent_spaces=indent_spaces,
                print_line_numbers=print_line_numbers,
                num_context_lines=num_context_lines,
                path_to_underline=path_to_underline,
                path_to_annotate=path_to_annotate,
                obj_to_underline=obj_to_underline,
                obj_to_annotate=obj_to_annotate,
                extra_config=extra_config,
            ),
            style=style,
        )


def _repr(self) -> str:
    try:
        return _script(
            self,
            PrinterConfig(extra_config={"ir.comment_imports": True}),
        )
    except Exception:
        return tvm_ffi.Object.__repr__(self)


# The printer supplies its explicit display capability to existing IR wrappers.
register_script_printer(
    script=_ScriptableMethods.script, show=_ScriptableMethods.show, node_repr=_repr
)
