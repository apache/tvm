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
# pylint: disable=redefined-builtin, abstract-method, arguments-differ
"""
Utility script for printing Relax modules as AST diagrams,
only intended to show how the AST is put together.
It is not a pretty-printer and, in fact, is more of an ugly-printer,
but it can be useful for tutorials and debugging.
"""
from typing import Iterable
import tvm
from tvm import relax
from tvm.ir.expr import PrimExpr
from tvm.relax import ExprFunctor


def wrap_quotes(text: str) -> str:
    """
    Wraps the text in quotes.
    """
    return f'"{text}"'


class ASTPrinter(ExprFunctor):
    """
    Class for recursing down ASTs and printing them in a very simple format,
    mainly for instructive purposes and, perhaps, debugging.
    """

    def __init__(
        self,
        indent_str="    ",
        include_struct_info_annotations=True,
        include_type_annotations=False,
        include_call_attrs=True,
    ):
        self.indent_str = indent_str
        self.include_type_annotations = include_type_annotations
        self.include_struct_info_annotations = include_struct_info_annotations
        self.include_call_attrs = include_call_attrs

    def visit_expr(self, expr: relax.Expr) -> str:
        # extend so we also dispatch to bindings and binding blocks,
        # a little silly but IRFunctor hasn't been ported to Python
        if isinstance(expr, relax.DataflowBlock):
            return self.visit_dataflow_block_(expr)
        if isinstance(expr, relax.BindingBlock):
            return self.visit_binding_block_(expr)
        if isinstance(expr, relax.Binding):
            return self.visit_binding_(expr)
        return super().visit_expr(expr)

    def indent(self, text: str) -> str:
        """
        Indent all lines of the input.
        """
        if text == "":
            return ""
        lines = text.split("\n")
        return self.indent_str + f"\n{self.indent_str}".join(lines)

    def build_ast_node(self, nodename: str, force_newline=False, **kwargs: str) -> str:
        """
        Returns 'nodename(..., fields[i][0]=fields[i][1], ...)'
        with appropriate indentation
        """
        return self.build_list(
            map(lambda field: f"{field[0]}={field[1]}", kwargs.items()),
            open_tok=f"{nodename}(",
            close_tok=")",
            force_newline=force_newline,
        )

    def build_expr(self, node: relax.Expr, nodename: str, force_newline=False, **kwargs: str):
        """
        Renders a Relax expression as a string using `build_ast_node`.
        Handles whether to include the checked_type_ and struct_info fields.
        """
        fields = kwargs.copy()
        if node.struct_info_ and self.include_struct_info_annotations:
            fields["struct_info"] = self.visit_struct_info_(node.struct_info)
        if node._checked_type_ and self.include_type_annotations:
            fields["checked_type_"] = self.visit_type_(node.checked_type)
        return self.build_ast_node(nodename, force_newline=force_newline, **fields)

    def build_list(
        self, members: Iterable[str], open_tok="[", close_tok="]", force_newline=False
    ) -> str:
        """
        Builds a list of the members given, appropriately indented,
        with each field on a line.
        (special case: if there is only one field, then we do not put it on a new line
        unless that field contains a newline or `force_newline` is set to true).
        `open_tok` and `close_tok` are used to open and close the list, respectively.
        """
        mem_list = list(members)
        if not mem_list:
            return f"{open_tok}{close_tok}"
        if len(mem_list) == 1 and not force_newline and "\n" not in mem_list[0]:
            return f"{open_tok}{mem_list[0]}{close_tok}"
        member_lines = ",\n".join(map(self.indent, mem_list))
        return f"{open_tok}\n{member_lines}\n{close_tok}"

    def visit_constant_(self, op: relax.Constant) -> str:
        # simple rule of thumb: keep scalars inline, but anything larger goes on a new one
        force_newline = len(op.data.shape) > 0
        return self.build_expr(op, "Constant", force_newline=force_newline, data=str(op.data))

    def visit_tuple_(self, op: relax.Tuple) -> str:
        return self.build_expr(op, "Tuple", fields=self.build_list(map(self.visit_expr, op.fields)))

    def visit_dataflow_var_(self, op: relax.DataflowVar) -> str:
        return self.build_expr(op, "DataflowVar", name_hint=wrap_quotes(op.name_hint))

    def visit_var_(self, op: relax.Var) -> str:
        return self.build_expr(op, "Var", name_hint=wrap_quotes(op.name_hint))

    def visit_shape_expr_(self, op: relax.ShapeExpr) -> str:
        return self.build_expr(
            op, "ShapeExpr", values=self.build_list(map(self.visit_prim_expr_, op.values))
        )

    def visit_extern_func_(self, op: relax.ExternFunc) -> str:
        # ExternFunc does not inherit from relax.Expr either,
        # so it doesn't have checked_type_ or struct_info fields and we don't use build_expr
        return self.build_ast_node("ExternFunc", global_symbol=wrap_quotes(op.global_symbol))

    def visit_global_var_(self, op: relax.GlobalVar) -> str:
        return self.build_expr(op, "GlobalVar", name_hint=wrap_quotes(op.name_hint))

    def visit_function_(self, op: relax.Function) -> str:
        fields = {
            "params": self.build_list(map(self.visit_expr, op.params)),
            "body": self.visit_expr(op.body),
            "ret_struct_info": self.visit_struct_info_(op.ret_struct_info),
            "is_pure": op.is_pure,
        }
        if op.attrs:
            fields["attrs"] = self.build_list(
                map(
                    lambda kv: f"{wrap_quotes(str(kv[0]))}: {wrap_quotes(str(kv[1]))}",
                    op.attrs.items(),
                ),
                open_tok="{",
                close_tok="}",
            )
        return self.build_expr(op, "Function", **fields)

    def visit_call_(self, op: relax.Call) -> str:
        fields = {
            "op": self.visit_expr(op.op),
            "args": self.build_list(map(self.visit_expr, op.args)),
        }
        if op.sinfo_args:
            fields["sinfo_args"] = self.build_list(map(self.visit_struct_info_, op.sinfo_args))
        if op.attrs and self.include_call_attrs:

            def display_attrs(attr_key):
                attr_val = op.attrs[attr_key]
                # attrs can be strings but also other types;
                # we want to wrap strings in quotes
                # (__repr__ would work but it uses single quotes)
                attr_str = wrap_quotes(attr_val) if isinstance(attr_val, str) else str(attr_val)
                return f"{wrap_quotes(attr_key)}: {attr_str}"

            fields["attrs"] = self.build_list(
                map(display_attrs, op.attrs.keys()),
                open_tok="{",
                close_tok="}",
            )
        return self.build_expr(op, "Call", **fields)

    def visit_seq_expr_(self, op: relax.SeqExpr) -> str:
        return self.build_expr(
            op,
            "SeqExpr",
            blocks=self.build_list(map(self.visit_binding_block_, op.blocks)),
            body=self.visit_expr(op.body),
        )

    def visit_if_(self, op: relax.If) -> str:
        return self.build_expr(
            op,
            "If",
            cond=self.visit_expr(op.cond),
            true_branch=self.visit_expr(op.true_branch),
            false_branch=self.visit_expr(op.false_branch),
        )

    def visit_prim_value_(self, op: relax.PrimValue) -> str:
        return self.build_expr(op, "PrimValue", value=self.visit_prim_expr_(op.value))

    def visit_string_imm_(self, op: relax.StringImm) -> str:
        return self.build_expr(op, "StringImm", value=wrap_quotes(op.value))

    def visit_data_type_imm_(self, op: relax.DataTypeImm) -> str:
        return self.build_expr(op, "DataTypeImm", value=op.value)

    def visit_op_(self, op: tvm.ir.Op) -> str:
        # TODO: List other attributes?
        # op is not actually a Relax expr and does not have checked_type_
        # or struct_info fields, so we don't use build_expr here
        return self.build_ast_node("Op", name=wrap_quotes(op.name))

    def visit_prim_expr_(self, prim_expr: PrimExpr) -> str:
        # TODO: We may want to print PrimExpr ASTs, but this is a simplification for now
        return self.build_ast_node("PrimExpr", value=f"`{str(prim_expr)}`")

    def visit_tuple_getitem_(self, op: relax.TupleGetItem) -> str:
        return self.build_expr(
            op,
            "TupleGetItem",
            tuple_value=self.visit_expr(op.tuple_value),
            index=str(op.index),
        )

    def visit_type_(self, type_node: relax.Type) -> str:
        """
        Recurse down types and print their ASTs too
        """
        if isinstance(type_node, relax.ShapeType):
            return self.build_ast_node("ShapeType", ndim=str(type_node.ndim))
        if isinstance(type_node, relax.ObjectType):
            return self.build_ast_node("ObjectType")
        if isinstance(type_node, relax.PackedFuncType):
            return self.build_ast_node("PackedFuncType")
        if isinstance(type_node, tvm.ir.PrimType):
            return self.build_ast_node("PrimType", dtype=type_node.dtype)
        if isinstance(type_node, relax.DynTensorType):
            fields = {}
            if type_node.ndim is not None:
                fields["ndim"] = str(type_node.ndim)
            if type_node.dtype != "":
                fields["dtype"] = type_node.dtype
            return self.build_ast_node("DynTensorType", **fields)
        if isinstance(type_node, relax.TupleType):
            return self.build_ast_node(
                "TupleType", fields=self.build_list(map(self.visit_type_, type_node.fields))
            )
        if isinstance(type_node, relax.FuncType):
            return self.build_ast_node(
                "FuncType",
                arg_types=self.build_list(map(self.visit_type_, type_node.arg_types)),
                ret_type=self.visit_type_(type_node.ret_type),
                # TODO: skipping type params and type constraints
            )
        raise ValueError(f"Invalid Relax Type {type_node} ({type(type_node)})")

    def visit_struct_info_(self, struct_info_node: relax.StructInfo) -> str:
        """
        Recurse down struct info and print their ASTs too
        """
        if isinstance(struct_info_node, relax.ShapeStructInfo):
            fields = {}
            fields["ndim"] = str(struct_info_node.ndim)
            if struct_info_node.values is not None:
                fields["values"] = self.build_list(
                    map(self.visit_prim_expr_, struct_info_node.values)
                )
            return self.build_ast_node("ShapeStructInfo", **fields)
        elif isinstance(struct_info_node, relax.ObjectStructInfo):
            return self.build_ast_node("ObjectStructInfo")
        elif isinstance(struct_info_node, relax.PrimStructInfo):
            return self.build_ast_node("PrimStructInfo", dtype=struct_info_node.dtype)
        elif isinstance(struct_info_node, relax.TensorStructInfo):
            fields = {}
            fields["dtype"] = struct_info_node.dtype
            if struct_info_node.shape:
                fields["shape"] = self.visit_expr(struct_info_node.shape)
            else:
                fields["ndim"] = str(struct_info_node.ndim)
            return self.build_ast_node("TensorStructInfo", **fields)
        elif isinstance(struct_info_node, relax.TupleStructInfo):
            return self.build_ast_node(
                "TupleStructInfo",
                fields=self.build_list(map(self.visit_struct_info_, struct_info_node.fields)),
            )
        elif isinstance(struct_info_node, relax.FuncStructInfo):
            fields = {}
            if struct_info_node.params is not None:
                fields["params"] = self.build_list(
                    map(self.visit_struct_info_, struct_info_node.params)
                )
            fields["ret"] = self.visit_struct_info_(struct_info_node.ret)
            fields["purity"] = bool(struct_info_node.purity)
            return self.build_ast_node("FuncStructInfo", **fields)
        else:
            raise ValueError(
                f"Invalid Relax StructInfo {struct_info_node} ({type(struct_info_node)})"
            )

    def visit_binding_block_(self, block: relax.BindingBlock) -> str:
        """
        Recurse down binding blocks
        """
        return self.build_ast_node(
            "BindingBlock",
            bindings=self.build_list(map(self.visit_binding_, block.bindings), force_newline=True),
        )

    def visit_dataflow_block_(self, block: relax.DataflowBlock) -> str:
        """
        Recurse down a dataflow block
        """
        return self.build_ast_node(
            "DataflowBlock",
            bindings=self.build_list(map(self.visit_binding_, block.bindings), force_newline=True),
        )

    def visit_binding_(self, binding: relax.Binding) -> str:
        """
        Distinguish between binding types
        """
        if isinstance(binding, relax.MatchCast):
            return self.visit_match_cast_(binding)
        if isinstance(binding, relax.VarBinding):
            return self.visit_var_binding_(binding)
        raise ValueError(f"Invalid binding type in {binding}: {type(binding)}")

    def visit_match_cast_(self, match_cast: relax.MatchCast) -> str:
        """
        Handle match shape
        """
        fields = {
            "var": self.visit_expr(match_cast.var),
            "value": self.visit_expr(match_cast.value),
            "struct_info": self.visit_struct_info_(match_cast.struct_info),
        }
        return self.build_ast_node("MatchCast", **fields)

    def visit_var_binding_(self, var_binding: relax.VarBinding) -> str:
        """
        Handle ordinary var bindings
        """
        return self.build_ast_node(
            "VarBinding",
            var=self.visit_expr(var_binding.var),
            value=self.visit_expr(var_binding.value),
        )


def dump_ast(
    exp: relax.Expr,
    indent_str="    ",
    include_struct_info_annotations=True,
    include_type_annotations=False,
    include_call_attrs=True,
) -> str:
    """
    Dump an AST in a text format.
    Can vary the indentation string and choose whether to include
    type and shape annotations or call attributes.
    """
    printer = ASTPrinter(
        indent_str=indent_str,
        include_struct_info_annotations=include_struct_info_annotations,
        include_type_annotations=include_type_annotations,
        include_call_attrs=include_call_attrs,
    )
    return printer.visit_expr(exp)
