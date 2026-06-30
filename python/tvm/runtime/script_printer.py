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

import os
from collections.abc import Sequence

from tvm_ffi import get_global_func, register_object
from tvm_ffi.access_path import AccessPath

from tvm.runtime import Object

from . import _ffi_node_api


@register_object("script.PrinterConfig")
class PrinterConfig(Object):
    """Configuration of TVMScript printer"""

    binding_names: Sequence[str]
    show_meta: bool
    ir_prefix: str
    module_alias: str
    buffer_dtype: str
    int_dtype: str
    float_dtype: str
    verbose_expr: bool
    indent_spaces: int
    print_line_numbers: bool
    num_context_lines: int
    syntax_sugar: bool
    show_object_address: bool
    render_invisible_path_info: bool
    extra_config: dict
    path_to_underline: list[AccessPath] | None
    visible_paths: list[AccessPath | None]
    path_to_annotate: dict[AccessPath, str] | None
    obj_to_underline: list[AccessPath] | None
    obj_to_annotate: dict[AccessPath, str] | None

    def __init__(
        self,
        *,
        name: str | None = None,
        show_meta: bool = False,
        ir_prefix: str = "I",
        module_alias: str = "cls",
        buffer_dtype: str = "float32",
        int_dtype: str = "int32",
        float_dtype: str = "void",
        verbose_expr: bool = False,
        indent_spaces: int = 4,
        print_line_numbers: bool = False,
        num_context_lines: int | None = None,
        syntax_sugar: bool = True,
        show_object_address: bool = False,
        render_invisible_path_info: bool = False,
        show_all_ty: bool = True,
        extra_config: dict | None = None,
        path_to_underline: list[AccessPath] | None = None,
        path_to_annotate: dict[AccessPath, str] | None = None,
        obj_to_underline: list[Object] | None = None,
        obj_to_annotate: dict[Object, str] | None = None,
    ) -> None:
        if num_context_lines is None:
            num_context_lines = -1
        cfg: dict = {
            "show_meta": show_meta,
            "ir_prefix": ir_prefix,
            "module_alias": module_alias,
            "buffer_dtype": buffer_dtype,
            "int_dtype": int_dtype,
            "float_dtype": float_dtype,
            "verbose_expr": verbose_expr,
            "indent_spaces": indent_spaces,
            "print_line_numbers": print_line_numbers,
            "num_context_lines": num_context_lines,
            "syntax_sugar": syntax_sugar,
            "show_object_address": show_object_address,
            "render_invisible_path_info": render_invisible_path_info,
            "path_to_underline": path_to_underline,
            "path_to_annotate": path_to_annotate,
            "obj_to_underline": obj_to_underline,
            "obj_to_annotate": obj_to_annotate,
            # Dialect-specific config via dotted keys in extra_config
            "relax.show_all_ty": show_all_ty,
        }

        if name is not None:
            cfg["name"] = name
        if extra_config is not None:
            cfg["extra_config"] = extra_config
        self.__init_handle_by_constructor__(
            _ffi_node_api.PrinterConfig,
            cfg,  # type: ignore # pylint: disable=no-member
        )


def _script(
    obj: Object,
    config: PrinterConfig | None,
) -> str | tuple[str, list[AccessPath | None]]:
    result = _ffi_node_api.TVMScriptPrinterScript(obj, config)  # type: ignore # pylint: disable=no-member
    if config is not None and config.render_invisible_path_info:
        script, visible_paths = result
        return script, list(visible_paths)
    return result


def _relax_script(obj: Object, config: PrinterConfig) -> str:
    func = get_global_func("script.printer.ReprPrintRelax")
    return func(obj, config)


class Scriptable:
    """A base class that enables the script() and show() method."""

    def script(
        self,
        *,
        name: str | None = None,
        show_meta: bool = False,
        ir_prefix: str = "I",
        module_alias: str = "cls",
        int_dtype: str = "int32",
        float_dtype: str = "void",
        verbose_expr: bool = False,
        indent_spaces: int = 4,
        print_line_numbers: bool = False,
        num_context_lines: int = -1,
        syntax_sugar: bool = True,
        show_object_address: bool = False,
        render_invisible_path_info: bool = False,
        show_all_ty: bool = True,
        extra_config: dict | None = None,
        path_to_underline: list[AccessPath] | None = None,
        path_to_annotate: dict[AccessPath, str] | None = None,
        obj_to_underline: list[Object] | None = None,
        obj_to_annotate: dict[Object, str] | None = None,
    ) -> str | tuple[str, list[AccessPath | None]]:
        """Print TVM IR into TVMScript text format

        Parameters
        ----------
        name : Optional[str] = None
            The name of the object
        show_meta : bool = False
            Whether to print the meta data of the object
        ir_prefix : str = "I"
            The prefix of AST nodes from tvm.ir
        module_alias : str = "cls"
            The alias of the current module at cross-function call,
            Directly use module name if it's empty.
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
        syntax_sugar: bool = True
            Whether to output with syntax sugar, set false for complete printing.
        show_object_address: bool = False
            Whether to include the object's address as part of the TVMScript name
        render_invisible_path_info: bool = False
            Whether to return the rendered script together with the visible
            AccessPath selected for each requested underline path.  The visible
            path may be a prefix of the requested access path when the requested
            field is not rendered in TVMScript.
        show_all_ty: bool = True
            If True (default), annotate all variable bindings with the struct
            info of that variable.  If False, only add annotations where
            required for unambiguous round-trip of Relax -> TVMScript -> Relax.
        extra_config : Optional[dict] = None
            Dialect-specific configuration passed through to PrinterConfig.extra_config.
            Keys are conventionally namespaced as "<dialect>.<knob>", e.g.
            ``{"tirx.prefix": "T"}``.
        path_to_underline : Optional[List[AccessPath]] = None
            Object path to be underlined
        path_to_annotate : Optional[Dict[AccessPath, str]] = None
            Object path to be annotated
        obj_to_underline : Optional[List[Object]] = None
            Object to be underlined
        obj_to_annotate : Optional[Dict[Object, str]] = None
            Object to be annotated

        Returns
        -------
        script : str
            The TVM Script of the given TVM IR.  When
            render_invisible_path_info=True, returns a tuple of script and
            visible paths.

        """
        # Auto-switch to tirx (`T`/`tirx`) flavor only when explicitly
        # printing a PrimFunc / IRModule that has no s_tir-tagged content.
        # Free objects (Buffer, BufferRegion, ...) keep the default `T`/`tir`
        # flavor -- they have no enclosing function to indicate tirx vs s_tir.
        merged_extra: dict = {}
        if extra_config is not None:
            merged_extra.update(extra_config)

        # Only auto-switch if the caller has not already set a tirx.prefix override.
        if "tirx.prefix" not in merged_extra:
            from tvm.ir import IRModule  # pylint: disable=import-outside-toplevel
            from tvm.tirx import PrimFunc  # pylint: disable=import-outside-toplevel

            switch_to_tirx = False
            if isinstance(self, PrimFunc):
                attrs = getattr(self, "attrs", None)
                if attrs is None or not attrs.get("s_tir", False):
                    switch_to_tirx = True
            elif isinstance(self, IRModule):
                any_prim = False
                any_s_tir = False
                for _, base_func in self.functions.items():
                    if isinstance(base_func, PrimFunc):
                        any_prim = True
                        if getattr(base_func, "attrs", None) and base_func.attrs.get(
                            "s_tir", False
                        ):
                            any_s_tir = True
                            break
                if any_prim and not any_s_tir:
                    switch_to_tirx = True
            if switch_to_tirx:
                merged_extra["tirx.prefix"] = "T"

        return _script(
            self,
            PrinterConfig(
                name=name,
                show_meta=show_meta,
                ir_prefix=ir_prefix,
                module_alias=module_alias,
                int_dtype=int_dtype,
                float_dtype=float_dtype,
                verbose_expr=verbose_expr,
                indent_spaces=indent_spaces,
                print_line_numbers=print_line_numbers,
                num_context_lines=num_context_lines,
                syntax_sugar=syntax_sugar,
                show_object_address=show_object_address,
                render_invisible_path_info=render_invisible_path_info,
                show_all_ty=show_all_ty,
                extra_config=merged_extra if merged_extra else None,
                path_to_underline=path_to_underline,
                path_to_annotate=path_to_annotate,
                obj_to_underline=obj_to_underline,
                obj_to_annotate=obj_to_annotate,
            ),
        )

    def _relax_script(
        self,
        *,
        name: str | None = None,
        show_meta: bool = False,
        ir_prefix: str = "I",
        module_alias: str = "cls",
        int_dtype: str = "int32",
        float_dtype: str = "void",
        verbose_expr: bool = False,
        indent_spaces: int = 4,
        print_line_numbers: bool = False,
        num_context_lines: int = -1,
        syntax_sugar: bool = True,
        show_object_address: bool = False,
        extra_config: dict | None = None,
        path_to_underline: list[AccessPath] | None = None,
        path_to_annotate: dict[AccessPath, str] | None = None,
        obj_to_underline: list[Object] | None = None,
        obj_to_annotate: dict[Object, str] | None = None,
    ) -> str:
        return _relax_script(
            self,
            PrinterConfig(
                name=name,
                show_meta=show_meta,
                ir_prefix=ir_prefix,
                module_alias=module_alias,
                int_dtype=int_dtype,
                float_dtype=float_dtype,
                verbose_expr=verbose_expr,
                indent_spaces=indent_spaces,
                print_line_numbers=print_line_numbers,
                num_context_lines=num_context_lines,
                syntax_sugar=syntax_sugar,
                show_object_address=show_object_address,
                extra_config=extra_config,
                path_to_underline=path_to_underline,
                path_to_annotate=path_to_annotate,
                obj_to_underline=obj_to_underline,
                obj_to_annotate=obj_to_annotate,
            ),
        )

    def show(
        self,
        style: str | None = None,
        black_format: bool | None = None,
        *,
        name: str | None = None,
        show_meta: bool = False,
        ir_prefix: str = "I",
        module_alias: str = "cls",
        int_dtype: str = "int32",
        float_dtype: str = "void",
        verbose_expr: bool = False,
        indent_spaces: int = 4,
        print_line_numbers: bool = False,
        num_context_lines: int = -1,
        syntax_sugar: bool = True,
        show_object_address: bool = False,
        show_all_ty: bool = True,
        extra_config: dict | None = None,
        path_to_underline: list[AccessPath] | None = None,
        path_to_annotate: dict[AccessPath, str] | None = None,
        obj_to_underline: list[Object] | None = None,
        obj_to_annotate: dict[Object, str] | None = None,
    ) -> None:
        """A sugar for print highlighted TVM script.

        Parameters
        ----------
        style : str, optional
            Pygmentize printing style, auto-detected if None.  See
            `tvm.script.highlight.cprint` for more details.

        black_format: Optional[bool]

            If true, use the formatter Black to format the TVMScript.
            If false, do not apply the auto-formatter.

            If None (default), determine the behavior based on the
            environment variable "TVM_BLACK_FORMAT".  If this
            environment variable is unset, set to the empty string, or
            set to the integer zero, black auto-formatting will be
            disabled.  If the environment variable is set to a
            non-zero integer, black auto-formatting will be enabled.

            Note that the "TVM_BLACK_FORMAT" environment variable only
            applies to the `.show()` method, and not the underlying
            `.script()` method.  The `.show()` method is intended for
            human-readable output based on individual user
            preferences, while the `.script()` method is intended to
            provided a consistent output regardless of environment.

        name : Optional[str] = None
            The name of the object
        show_meta : bool = False
            Whether to print the meta data of the object
        ir_prefix : str = "I"
            The prefix of AST nodes from tvm.ir
        module_alias : str = "cls"
            The alias of the current module at cross-function call,
            Directly use module name if it's empty.
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
        syntax_sugar: bool = True
            Whether to output with syntax sugar, set false for complete printing.
        show_object_address: bool = False
            Whether to include the object's address as part of the TVMScript name
        show_all_ty: bool = True
            If True (default), annotate all variable bindings with the struct
            info of that variable.  If False, only add annotations where
            required for unambiguous round-trip of Relax -> TVMScript -> Relax.
        extra_config : Optional[dict] = None
            Dialect-specific configuration passed through to PrinterConfig.extra_config.
        path_to_underline : Optional[List[AccessPath]] = None
            Object path to be underlined
        path_to_annotate : Optional[Dict[AccessPath, str]] = None
            Object path to be annotated
        obj_to_underline : Optional[List[Object]] = None
            Object to be underlined
        obj_to_annotate : Optional[Dict[Object, str]] = None
            Object to be annotated

        """
        from tvm.script.highlight import cprint  # pylint: disable=import-outside-toplevel

        if black_format is None:
            env = os.environ.get("TVM_BLACK_FORMAT")
            black_format = env and int(env)

        cprint(
            self.script(
                name=name,
                show_meta=show_meta,
                ir_prefix=ir_prefix,
                module_alias=module_alias,
                int_dtype=int_dtype,
                float_dtype=float_dtype,
                verbose_expr=verbose_expr,
                indent_spaces=indent_spaces,
                print_line_numbers=print_line_numbers,
                num_context_lines=num_context_lines,
                syntax_sugar=syntax_sugar,
                show_object_address=show_object_address,
                show_all_ty=show_all_ty,
                extra_config=extra_config,
                path_to_underline=path_to_underline,
                path_to_annotate=path_to_annotate,
                obj_to_underline=obj_to_underline,
                obj_to_annotate=obj_to_annotate,
            ),
            style=style,
            black_format=black_format,
        )
