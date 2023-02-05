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
from typing import Optional

from tvm._ffi import register_object
from tvm.runtime import Object

from . import _ffi_node_api
from .object_path import ObjectPath


@register_object("node.PrinterConfig")
class PrinterConfig(Object):
    """Configuration of TVMScript printer"""

    ir_prefix: str
    tir_prefix: str
    relax_prefix: str
    buffer_dtype: str
    int_dtype: str
    float_dtype: str
    verbose_expr: bool
    indent_spaces: int
    print_line_numbers: bool
    num_context_lines: int
    path_to_underline: Optional[ObjectPath]
    syntax_sugar: bool

    def __init__(
        self,
        *,
        ir_prefix: str = "I",
        tir_prefix: str = "T",
        relax_prefix: str = "R",
        buffer_dtype: str = "float32",
        int_dtype: str = "int32",
        float_dtype: str = "void",
        verbose_expr: bool = False,
        indent_spaces: int = 4,
        print_line_numbers: bool = False,
        num_context_lines: Optional[int] = None,
        path_to_underline: Optional[ObjectPath] = None,
        syntax_sugar: bool = True,
    ) -> None:
        if num_context_lines is None:
            num_context_lines = -1
        self.__init_handle_by_constructor__(
            _ffi_node_api.PrinterConfig,  # type: ignore # pylint: disable=no-member
            {
                "ir_prefix": ir_prefix,
                "tir_prefix": tir_prefix,
                "relax_prefix": relax_prefix,
                "buffer_dtype": buffer_dtype,
                "int_dtype": int_dtype,
                "float_dtype": float_dtype,
                "verbose_expr": verbose_expr,
                "indent_spaces": indent_spaces,
                "print_line_numbers": print_line_numbers,
                "num_context_lines": num_context_lines,
                "path_to_underline": path_to_underline,
                "syntax_sugar": syntax_sugar,
            },
        )


def _script(obj: Object, config: PrinterConfig) -> str:
    return _ffi_node_api.TVMScriptPrinterScript(obj, config)  # type: ignore # pylint: disable=no-member


class Scriptable:
    """A base class that enables the script() and show() method."""

    def script(
        self,
        *,
        ir_prefix: str = "I",
        tir_prefix: str = "T",
        relax_prefix: str = "R",
        buffer_dtype: str = "float32",
        int_dtype: str = "int32",
        float_dtype: str = "void",
        verbose_expr: bool = False,
        indent_spaces: int = 4,
        print_line_numbers: bool = False,
        num_context_lines: int = -1,
        path_to_underline: Optional[ObjectPath] = None,
        syntax_sugar: bool = True,
    ) -> str:
        """Print TVM IR into TVMScript text format

        Parameters
        ----------
        ir_prefix : str = "I"
            The prefix of AST nodes from tvm.ir
        tir_prefix : str = "T"
            The prefix of AST nodes from tvm.tir
        relax_prefix : str = "R"
            The prefix of AST nodes from tvm.relax
        buffer_dtype : str = "float32"
            The default data type of buffer
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
        path_to_underline : Optional[ObjectPath] = None
            Object path to be underlined
        syntax_sugar: bool = True
             Whether to output with syntax sugar, set false for complete printing.

        Returns
        -------
        script : str
            The TVM Script of the given TVM IR
        """
        return _script(
            self,
            PrinterConfig(
                ir_prefix=ir_prefix,
                tir_prefix=tir_prefix,
                relax_prefix=relax_prefix,
                buffer_dtype=buffer_dtype,
                int_dtype=int_dtype,
                float_dtype=float_dtype,
                verbose_expr=verbose_expr,
                indent_spaces=indent_spaces,
                print_line_numbers=print_line_numbers,
                num_context_lines=num_context_lines,
                path_to_underline=path_to_underline,
                syntax_sugar=syntax_sugar,
            ),
        )

    def show(
        self,
        style: Optional[str] = None,
        black_format: bool = True,
        *,
        ir_prefix: str = "I",
        tir_prefix: str = "T",
        relax_prefix: str = "R",
        buffer_dtype: str = "float32",
        int_dtype: str = "int32",
        float_dtype: str = "void",
        verbose_expr: bool = False,
        indent_spaces: int = 4,
        print_line_numbers: bool = False,
        num_context_lines: int = -1,
        path_to_underline: Optional[ObjectPath] = None,
        syntax_sugar: bool = True,
    ) -> None:
        """A sugar for print highlighted TVM script.

        Parameters
        ----------
        style : str, optional
            Pygmentize printing style, auto-detected if None.  See
            `tvm.script.highlight.cprint` for more details.
        black_format: bool
            If true (default), use the formatter Black to format the TVMScript
        ir_prefix : str = "I"
            The prefix of AST nodes from tvm.ir
        tir_prefix : str = "T"
            The prefix of AST nodes from tvm.tir
        relax_prefix : str = "R"
            The prefix of AST nodes from tvm.relax
        buffer_dtype : str = "float32"
            The default data type of buffer
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
        path_to_underline : Optional[ObjectPath] = None
            Object path to be underlined
        syntax_sugar: bool = True
             Whether to output with syntax sugar, set false for complete printing.
        """
        from tvm.script.highlight import (  # pylint: disable=import-outside-toplevel
            cprint,
        )

        cprint(
            self.script(
                ir_prefix=ir_prefix,
                tir_prefix=tir_prefix,
                relax_prefix=relax_prefix,
                buffer_dtype=buffer_dtype,
                int_dtype=int_dtype,
                float_dtype=float_dtype,
                verbose_expr=verbose_expr,
                indent_spaces=indent_spaces,
                print_line_numbers=print_line_numbers,
                num_context_lines=num_context_lines,
                path_to_underline=path_to_underline,
                syntax_sugar=syntax_sugar,
            ),
            style=style,
            black_format=black_format,
        )
