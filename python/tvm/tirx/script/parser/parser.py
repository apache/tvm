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
"""The base parser for tirx"""

import ast
import contextlib
from copy import deepcopy
from functools import partial
from typing import Any, TypeVar

import tvm
from tvm.ir import Expr, GlobalVar, PointerType, PrimType
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder.base import IRBuilder
from tvm.script.ir_builder.base import IRBuilderFrame as Frame
from tvm.script.parser._core import Parser, collect_signature_type_vars, dispatch, doc
from tvm.script.parser.core.doc import from_doc
from tvm.tirx import Buffer, IterVar, Layout, buffer_data, is_buffer_var
from tvm.tirx.script import builder as T
from tvm.tirx.script.builder.ir import name_meta_class_value
from tvm.tirx.stmt import BufferRegion

from .entry import _OptionalAnnotation, inline
from .entry import constexpr as _constexpr_sentinel


def slice_buffer_from_region(br: BufferRegion) -> Buffer:
    """Create a matched DeclBuffer from a BufferRegion.

    Slices the layout (if present) or computes elem_offset for the sub-region,
    producing a DeclBuffer that views the same underlying data.
    """
    import functools  # pylint: disable=import-outside-toplevel

    buf = br.buffer
    region = br.region
    new_shape = [r.extent for r in region]
    sliced_layout = None
    if buf.ty.layout is not None:
        range_pairs = [(r.min, r.min + r.extent) for r in region]
        sliced_layout = buf.ty.layout.slice(list(buf.ty.shape), range_pairs)
    if sliced_layout is not None:
        return T.decl_buffer(
            new_shape,
            buf.ty.dtype,
            buffer_data(buf),
            buf.ty.strides,
            buf.ty.elem_offset,
            None,
            buf.ty.storage_scope,
            buf.ty.data_alignment,
            buf.ty.offset_factor,
            sliced_layout,
        )
    # Fallback: compute elem_offset for default/no layout
    strides = []
    for i in range(len(buf.ty.shape)):
        stride = functools.reduce(
            lambda x, y: x * y, buf.ty.shape[i + 1 :], tvm.tirx.const(1, "int32")
        )
        strides.append(stride)
    offset = tvm.tirx.const(0, "int32")
    for i, r in enumerate(region):
        offset = offset + r.min * strides[i]
    new_elem_offset = buf.ty.elem_offset + offset
    return T.decl_buffer(
        new_shape,
        buf.ty.dtype,
        buffer_data(buf),
        buf.ty.strides,
        new_elem_offset,
        None,
        buf.ty.storage_scope,
        buf.ty.data_alignment,
        buf.ty.offset_factor,
        buf.ty.layout,
    )


def bind_with_value(self: Parser, node: doc.expr, var_name: str, value: Any) -> Any:
    """Value binding methods when parsing with statement.
    e.g. binding i, j, k with T.grid(128, 128, 128), when parsing
        with T.grid(128, 128, 18) as i, j, k.

    Parameters
    ----------
    self : Parser
        The current parser.

    node : doc.expr
        The doc AST expression node for error reporting.

    var_name : str
        The variable name.

    value : Any
        The value to be bound with.

    Returns
    -------
    res : Any
        The bound value.
    """
    if isinstance(value, list | tuple):
        for i, v in enumerate(value):
            bind_with_value(self, node, f"{var_name}_{i}", v)
        return value
    elif isinstance(value, tvm.ir.Var):
        IRBuilder.name(var_name, value)
        return value
    else:
        self.report_error(node, f"Do not know how to bind type: {type(value)} in with statement")
        raise NotImplementedError


def bind_for_value(self: Parser, node: doc.expr, var_name: str, value: Any) -> Any:
    """Value binding methods when parsing for statement.
    e.g. binding i, j, k with T.grid(128, 128, 128), when parsing
        for i, j, k in T.grid(128, 128, 128).

    Parameters
    ----------
    self : Parser
        The current parser.

    node : doc.expr
        The doc AST expression node for error reporting.

    var_name : str
        The variable name.

    value : Any
        The value to be bound with.

    Returns
    -------
    res : Any
        The bound value.
    """
    if isinstance(value, list | tuple | tvm.ir.Array):
        for i, v in enumerate(value):
            bind_for_value(self, node, f"{var_name}_{i}", v)
        return value
    elif isinstance(value, tvm.ir.Var):
        IRBuilder.name(var_name, value)
        return value
    else:
        self.report_error(node, f"Do not know how to bind type: {type(value)} in for statement")
        raise NotImplementedError


def _convert_tuple_literal(self: Parser, node: doc.expr, value: Any) -> Any:
    """Convert an explicit list/tuple T.let value to a core IR Tuple.

    The generic evaluator deliberately keeps Python containers intact because
    many TIRx builder APIs use them structurally.  Conversion is restricted to
    the internal immutable-binding boundary that explicitly opts in below.
    """
    if not isinstance(node, doc.List | doc.Tuple):
        return value

    def convert_field(field: Any) -> Expr:
        if isinstance(field, list | tuple):
            return tvm.ir.Tuple([convert_field(item) for item in field])
        if isinstance(field, Expr):
            return field
        if isinstance(field, str):
            return tvm.tirx.StringImm(field)
        if isinstance(field, bool | int | float):
            return tvm.tirx.const(field)
        self.report_error(
            node,
            f"Tuple fields must be expressions or scalar literals, got {type(field).__name__}",
        )
        raise NotImplementedError

    return convert_field(value)


def bind_assign_value(
    self: Parser,
    node: doc.expr,
    var_name: str,
    value: Any,
    prim_var_declarations: set[str] | None = None,
) -> Any:
    """Value binding methods when parsing assign statement.
    e.g. binding vi, vj, vk with T.axis.remap("SSR", [i, j, k]), when parsing
        vi, vj, vk = T.axis.remap("SSR", [i, j, k]).

    Parameters
    ----------
    self : Parser
        The current parser.

    node : doc.expr
        The doc AST expression node for error reporting.

    var_name : str
        The variable name.

    value : Any
        The value to be bound with.

    Returns
    -------
    res : Any
        The bound value.
    """
    if var_name in (prim_var_declarations or set()):
        # A quoted Buffer shape may have already created this PrimVar.  In that
        # case ``n = T.int32()`` is match-like syntax: bind the Python name to
        # the signature Var instead of emitting a second definition.
        if _get_signature_match_var(self, var_name) is not None:
            return _reuse_signature_match_var(self, node, var_name, value)
    if isinstance(value, T.scalar_wrapper):  # pylint: disable=protected-access
        # special case for scalar, name the buffer, but the var is used as BufferLoad
        assert isinstance(value.scalar, T.BufferLoad)
        IRBuilder.name(var_name, value.scalar.buffer)
        return value.scalar
    if isinstance(value, I.meta_var):
        return value.value
    elif getattr(type(value), "_is_meta_class", False):
        name_meta_class_value(var_name, value)
        return value
    elif isinstance(value, list | tuple):
        # Tuple-unpacking with a starred target (e.g. ``vi, *vs = T.axis.remap(...)``)
        # collects multiple elements into a single list bound here. Recurse so each
        # element gets a per-index name; this matches apache's behavior.
        for i, v in enumerate(value):
            bind_assign_value(self, node, f"{var_name}_{i}", v)
        return value
    elif isinstance(value, BufferRegion):
        return value
    elif isinstance(value, Frame):
        value.add_callback(partial(value.__exit__, None, None, None))
        res = value.__enter__()
        IRBuilder.name(var_name, res)
        return res
    elif (
        is_buffer_var(value)
        or isinstance(value, IterVar | Layout)
        or (isinstance(value, tvm.ir.Var) and not self.var_table.exist(value))
    ):
        IRBuilder.name(var_name, value)
        return value
    else:
        is_pointer_expr = isinstance(value, tvm.ir.Expr) and isinstance(
            getattr(value, "ty", None), PointerType
        )
        if is_pointer_expr:
            if self.var_table.contains_in_current_frame(var_name):
                self.report_error(node, f"Pointer variable '{var_name}' cannot be reassigned")
            ann_var = T.Bind(value)
            IRBuilder.name(var_name, ann_var)
            return ann_var
        if not tvm.ir.is_prim_expr(value) and not isinstance(value, Expr):
            # Python scalar (int/float/bool) -> const prim expr
            value = tvm.tirx.const(value)
        if isinstance(value, tvm.tirx.StringImm) or not tvm.ir.is_prim_expr(value):
            # StringImm or non-prim-expr (e.g. pointer Call): immutable Bind var
            ann_var = tvm.tirx.Var(var_name, value.ty)
            IRBuilder.name(var_name, ann_var)
            T.Bind(value, var=ann_var)
            return ann_var
        else:
            # x = expr -> scalar (auto-typed from value)
            scalar = T.local_scalar(dtype=str(value.ty.dtype))
            IRBuilder.name(var_name, scalar.scalar.buffer)
            T.buffer_store(scalar.scalar.buffer, value, [0])
            return scalar.scalar


def find_decorator_annotation(node: doc.FunctionDef, annotation: str, default: bool = True) -> bool:
    """
    Check the value of given annotation (argument name) in the prim_func decorator.
    Returns the value of the annotation if present, otherwise giving the default value.
    """
    # look for the named argument in the prim_func / jit decorator
    for dec in node.decorator_list:
        if not isinstance(dec, doc.Call) or dec.func.attr not in ("prim_func", "jit"):
            continue
        for keyword in dec.keywords:
            if keyword.arg == annotation:
                return keyword.value.value
    return default


def _is_jit_function(node: doc.FunctionDef) -> bool:
    """Return whether the parsed source function is decorated with ``T.jit``."""

    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, doc.Call) else decorator
        if isinstance(target, doc.Attribute) and target.attr == "jit":
            return True
        if isinstance(target, doc.Name) and target.id == "jit":
            return True
    return False


def _is_prim_var_declaration(node: doc.expr) -> bool:
    """Return whether an expression is literal ``T.dtype()`` declaration syntax."""
    if not (
        isinstance(node, doc.Call)
        and not node.args
        and not node.keywords
        and isinstance(node.func, doc.Attribute)
        and isinstance(node.func.value, doc.Name)
        and node.func.value.id == "T"
    ):
        return False
    constructor = getattr(T, node.func.attr, None)
    return isinstance(constructor, T.DtypeConstructor) or constructor is T.bool


def _prim_var_annotation_dtype(node: doc.expr) -> str | None:
    """Return the dtype of a scalar ``T.dtype`` parameter annotation."""
    if not (
        isinstance(node, doc.Attribute)
        and isinstance(node.value, doc.Name)
        and node.value.id == "T"
    ):
        return None
    constructor = getattr(T, node.attr, None)
    if not isinstance(constructor, T.DtypeConstructor) and constructor is not T.bool:
        return None
    return str(constructor().ty.dtype)


def _prim_var_declaration_names(target: doc.expr, value: doc.expr) -> set[str]:
    """Return targets paired with literal ``T.dtype()`` declarations.

    This classifies only the assignment currently being visited.  In
    particular, it is not a predeclaration pass over the function body.
    """
    if isinstance(target, doc.Name):
        return {target.id} if _is_prim_var_declaration(value) else set()
    if isinstance(target, doc.Tuple | doc.List) and isinstance(value, doc.Tuple | doc.List):
        if len(target.elts) != len(value.elts):
            return set()
        declarations = set()
        for lhs, rhs in zip(target.elts, value.elts):
            declarations.update(_prim_var_declaration_names(lhs, rhs))
        return declarations
    return set()


def _signature_prim_var_dtypes(node: doc.FunctionDef) -> dict[str, str]:
    """Collect scalar parameter dtypes without inspecting the function body."""
    return {
        arg.arg: dtype
        for arg in node.args.args
        if arg.annotation is not None
        if (dtype := _prim_var_annotation_dtype(arg.annotation)) is not None
    }


@contextlib.contextmanager
def _signature_match_var_scope(self: Parser):
    """Track PrimVars introduced by Buffer-shape match expressions."""
    previous = getattr(self, "_signature_match_vars", None)
    self._signature_match_vars = {}
    try:
        yield
    finally:
        if previous is None:
            del self._signature_match_vars
        else:
            self._signature_match_vars = previous


def _get_signature_match_var(self: Parser, var_name: str) -> tvm.ir.Var | None:
    """Return the active PrimVar created by a signature match expression."""
    current = self.var_table.get().get(var_name)
    match_var = getattr(self, "_signature_match_vars", {}).get(var_name)
    if (
        isinstance(current, tvm.ir.Var)
        and isinstance(match_var, tvm.ir.Var)
        and current.same_as(match_var)
    ):
        return current
    return None


def _reuse_signature_match_var(self: Parser, node: doc.expr, var_name: str, value: Any) -> Any:
    """Match a later scalar declaration to a signature-defined PrimVar."""
    previous = _get_signature_match_var(self, var_name)
    if previous is None:
        return value
    if not (
        isinstance(previous.ty, PrimType)
        and isinstance(value, tvm.ir.Var)
        and isinstance(value.ty, PrimType)
        and not is_buffer_var(previous)
        and not is_buffer_var(value)
    ):
        self.report_error(node, f"{var_name} is already bound to a non-PrimVar value")
    if previous.ty != value.ty:
        self.report_error(
            node,
            f"Expected the same dtype for PrimVars but got {value.ty} vs {previous.ty}",
        )
    return previous


def _eval_signature_annotation(
    self: Parser,
    node: doc.expr,
    signature_dtypes: dict[str, str],
    *,
    define_missing: bool = True,
) -> Any:
    """Evaluate a function annotation using Buffer-shape match semantics.

    Function parameters are parsed from left to right.  An ordinary name in an
    annotation is therefore a lookup and must already be bound.  A quoted
    expression in ``T.Buffer``'s shape is the one exception: it is a match
    scope, so its first occurrence creates every missing PrimVar before the
    expression is evaluated.  Later dimensions and parameters reuse those
    exact Var objects.

    ``signature_dtypes`` comes only from scalar parameter annotations, never
    from the function body.  It lets a match expression that precedes a scalar
    parameter create the Var with that parameter's declared dtype.  Otherwise
    TIR match variables default to int32.  Return annotations pass
    ``define_missing=False`` and consequently cannot introduce variables.
    """

    class ShapeStringRewriter(ast.NodeTransformer):
        def visit_Name(self, name):  # pylint: disable=invalid-name
            # Direct AST names follow normal definition-before-use ordering.
            if isinstance(name.ctx, ast.Load) and name.id not in self_parser.var_table.get():
                raise NameError(f"name '{name.id}' is not defined")
            return name

        def visit_Constant(self, constant):  # pylint: disable=invalid-name
            if not isinstance(constant.value, str):
                return constant
            # Replace the quoted expression with its AST.  Register its names
            # first so evaluating the rewritten annotation resolves all uses to
            # the canonical match-scope Vars.
            expression = ast.parse(constant.value, mode="eval").body
            for child in ast.walk(expression):
                if not isinstance(child, ast.Name) or not isinstance(child.ctx, ast.Load):
                    continue
                current_value = self_parser.var_table.get().get(child.id)
                is_shadowed_type_var = child.id in signature_dtypes and isinstance(
                    current_value, TypeVar
                )
                if define_missing and (current_value is None or is_shadowed_type_var):
                    # TIR match-scope indices default to int32.  A later scalar
                    # parameter keeps its explicitly declared dtype.  That
                    # runtime parameter also shadows a module TypeVar with the
                    # same name, such as one emitted for another function.
                    var = tvm.tirx.Var(child.id, signature_dtypes.get(child.id, "int32"))
                    self_parser.var_table.add(child.id, var, allow_shadowing=False)
                    self_parser._signature_match_vars[child.id] = var
            for child in ast.walk(expression):
                ast.copy_location(child, constant)
            return expression

    class BufferShapeRewriter(ast.NodeTransformer):
        def visit_Call(self, call):  # pylint: disable=invalid-name
            is_buffer = (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "Buffer"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "T"
            )
            if not is_buffer:
                return self.generic_visit(call)
            # Only Buffer.shape has match-scope semantics.  Dtype and other
            # Buffer options remain ordinary expressions.
            call.func = self.visit(call.func)
            if call.args:
                call.args[0] = ShapeStringRewriter().visit(call.args[0])
                call.args[1:] = [self.visit(arg) for arg in call.args[1:]]
            for keyword in call.keywords:
                if keyword.arg == "shape":
                    keyword.value = ShapeStringRewriter().visit(keyword.value)
                else:
                    keyword.value = self.visit(keyword.value)
            return call

    self_parser = self
    python_node = from_doc(deepcopy(node))
    python_node = BufferShapeRewriter().visit(python_node)
    ast.fix_missing_locations(python_node)
    return self.eval_expr(doc.to_doc(python_node))


@dispatch.register(token="tirx", type_name="For")
def visit_for(self: Parser, node: doc.For) -> None:
    """The for visiting method for tirx.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.For
        The doc AST for node.
    """
    # Intercept range() at AST level so it works with both Python ints and Exprs.
    # In other contexts (e.g. list comprehensions), range remains Python's builtin.
    if (
        isinstance(node.iter, doc.Call)
        and isinstance(node.iter.func, doc.Name)
        and node.iter.func.id == "range"
    ):
        args = [self.eval_expr(a) for a in node.iter.args]
        kwargs = {kw.arg: self.eval_expr(kw.value) for kw in node.iter.keywords}
        if len(args) == 1:
            for_frame = T.serial(0, args[0], **kwargs)
        elif len(args) == 2:
            for_frame = T.serial(args[0], args[1], **kwargs)
        elif len(args) == 3:
            for_frame = T.serial(args[0], args[1], step=args[2], **kwargs)
        else:
            self.report_error(node.iter, "range() takes 1 to 3 arguments")
    else:
        for_frame = self.eval_expr(node.iter)
    if not isinstance(for_frame, T.frame.ForFrame):
        self.report_error(
            node.iter,
            "Expect the for loop to be one of the following: "
            "range, T.serial, T.grid, T.parallel, T.vectorized, T.unroll, T.thread_binding",
        )
    with self.var_table.with_frame():
        with for_frame as iters:
            self.eval_assign(target=node.target, source=iters, bind_value=bind_for_value)
            self.visit_body(node.body)


@dispatch.register(token="tirx", type_name="While")
def visit_while(self: Parser, node: doc.While) -> None:
    """The while visiting method for tirx.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.While
        The doc AST while node.
    """
    with self.var_table.with_frame():
        cond = self.eval_expr(node.test)
        with T.While(cond):
            self.visit_body(node.body)


@dispatch.register(token="tirx", type_name="Break")
def visit_break(self: Parser, node: doc.Break) -> None:
    """The break visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Break
        The doc AST break node.
    """
    T.evaluate(T.break_loop())


@dispatch.register(token="tirx", type_name="Continue")
def visit_continue(self: Parser, node: doc.Continue) -> None:
    """The continue visiting method for tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Continue
        The doc AST continue node.
    """
    T.evaluate(T.continue_loop())


@dispatch.register(token="tirx", type_name="Assign")
def visit_assign(self: Parser, node: doc.Assign) -> None:
    """The assign visiting method for tirx.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Assign
        The doc AST assign node.
    """
    if len(node.targets) != 1:
        self.report_error(node, "Consequential assignments like 'a = b = c' are not supported.")
    lhs = node.targets[0]

    if isinstance(node.value, doc.Subscript):
        check_slices = []
        if isinstance(node.value.slice, doc.Slice):
            check_slices = [node.value.slice]
        elif isinstance(node.value.slice, doc.Tuple):
            for p in node.value.slice.elts:
                if isinstance(p, doc.Slice):
                    check_slices.append(p)
        for s in check_slices:
            if not s.step and s.upper and s.lower:
                s.step = doc.Constant(
                    1,
                    None,
                    s.upper.lineno,
                    s.upper.end_col_offset + 1,
                    s.upper.lineno,
                    s.upper.end_col_offset + 2,
                )

    rhs = self.eval_expr(node.value)
    if isinstance(lhs, doc.Subscript):
        if isinstance(lhs.slice, doc.Tuple):
            indices = []
            for index in lhs.slice.elts:
                if isinstance(index, doc.Starred):
                    # x[*y]
                    indices.extend(self.eval_expr(index.value))
                else:
                    indices.append(self.eval_expr(index))
        else:
            indices = self.eval_expr(lhs.slice)
        T.buffer_store(self.eval_expr(lhs.value), rhs, indices)
    else:
        # special case for scalar buffers
        # scalar = xxx <=> scalar.buffer[()] = xxx
        # or for a normal 1-dim buffer with shape (1,)
        # buffer = xxx <=> buffer[()] = xxx
        # Try to resolve lhs as a buffer/scalar variable. eval_expr may raise
        # if the name is not yet defined (i.e. this is a new variable binding),
        # which is the expected fallthrough case.
        lhs_value = None
        try:
            lhs_copy = deepcopy(lhs)
            if hasattr(lhs_copy, "ctx"):
                lhs_copy.ctx = doc.Load()
            lhs_value = self.eval_expr(lhs_copy)
        except Exception:  # pylint: disable=broad-except
            pass
        # Buffer check and store are intentionally outside the try/except so
        # that genuine errors (e.g. wrong shape, bad store) are not swallowed.
        # Only TypeError from FFI type mismatch (e.g. rhs is a meta_var, not
        # a Expr or auto-convertible scalar) triggers fallthrough.
        if isinstance(lhs_value, T.scalar_wrapper | T.BufferLoad) or is_buffer_var(lhs_value):
            if isinstance(lhs_value, T.scalar_wrapper):
                buffer = lhs_value.scalar.buffer
            else:
                buffer = lhs_value.buffer if isinstance(lhs_value, T.BufferLoad) else lhs_value
            if len(buffer.ty.shape) == 1 and bool(buffer.ty.shape[0] == 1):
                # only 1-dim buffer with shape (1,) can be assigned directly
                # Note that shape can be a Expr, so we only judge by
                # bool(shape[0] == 1) rather than int(shape[0]) == 1.
                try:
                    T.buffer_store(buffer, rhs, [0])
                    return
                except TypeError:
                    pass  # rhs not compatible with buffer_store, fall through
        # otherwise
        declarations = _prim_var_declaration_names(lhs, node.value)
        self.eval_assign(
            target=lhs,
            source=rhs,
            bind_value=partial(bind_assign_value, prim_var_declarations=declarations),
        )


@dispatch.register(token="tirx", type_name="AugAssign")
def visit_aug_assign(self: Parser, node: doc.AugAssign) -> None:
    """The augmented assign visiting method for tirx.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.AugAssign
        The doc AST augmented assign node.
    """
    lhs_pos = (
        node.target.lineno,
        node.target.col_offset,
        node.target.end_lineno,
        node.target.end_col_offset,
    )
    rhs_pos = (
        node.value.lineno,
        node.value.col_offset,
        node.value.end_lineno,
        node.value.end_col_offset,
    )
    node.target.ctx = doc.Load()
    with self.var_table.with_frame():
        lhs_name = "__tvm_tmp_value_aug_assign_lhs"
        rhs_name = "__tvm_tmp_value_aug_assign_rhs"
        lhs_expr = self.eval_expr(node.target)
        rhs_expr = self.eval_expr(node.value)
        self.var_table.add(lhs_name, lhs_expr)
        self.var_table.add(rhs_name, rhs_expr)
        op = doc.BinOp(
            doc.Name(lhs_name, doc.Load(), *lhs_pos),
            node.op,
            doc.Name(rhs_name, doc.Load(), *rhs_pos),
            *lhs_pos,
        )
        rhs = self.eval_expr(op)
    lhs = node.target
    lhs.ctx = doc.Store()
    if isinstance(lhs, doc.Subscript):
        if isinstance(lhs.slice, doc.Tuple):
            indices = []
            for index in lhs.slice.elts:
                if isinstance(index, doc.Starred):
                    # x[*y]
                    indices.extend(self.eval_expr(index.value))
                else:
                    indices.append(self.eval_expr(index))
        else:
            indices = [self.eval_expr(lhs.slice)]
        T.buffer_store(self.eval_expr(lhs.value), rhs, indices)
    else:
        lhs_value = None
        try:
            lhs_copy = deepcopy(lhs)
            if hasattr(lhs_copy, "ctx"):
                lhs_copy.ctx = doc.Load()
            lhs_value = self.eval_expr(lhs_copy)
        except Exception:  # pylint: disable=broad-except
            pass
        if isinstance(lhs_value, T.scalar_wrapper | T.BufferLoad) or is_buffer_var(lhs_value):
            if isinstance(lhs_value, T.scalar_wrapper):
                buffer = lhs_value.scalar.buffer
            else:
                buffer = lhs_value.buffer if isinstance(lhs_value, T.BufferLoad) else lhs_value
            if len(buffer.ty.shape) == 1 and bool(buffer.ty.shape[0] == 1):
                try:
                    T.buffer_store(buffer, rhs, [0])
                    return
                except TypeError:
                    pass
        self.eval_assign(target=lhs, source=rhs, bind_value=bind_assign_value)


@dispatch.register(token="tirx", type_name="AnnAssign")
def visit_ann_assign(self: Parser, node: doc.AnnAssign) -> None:
    """The annotated assign visiting method for tirx.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.AnnAssign
        The doc AST annotated assign node.
    """
    lhs = node.target
    rhs = self.eval_expr(node.value) if node.value is not None else None
    raw_ann = self.eval_expr(node.annotation)

    if isinstance(raw_ann, T.LocalVectorAnnotation):
        # x: T.float32[N] or x: T.f32[M, N] -> local buffer allocation
        if rhs is not None:
            self.report_error(node, "Vector annotation does not support initial value")
        buf = T.alloc_local(shape=raw_ann.shape, dtype=raw_ann.dtype)
        self.eval_assign(target=lhs, source=buf, bind_value=bind_assign_value)
    elif isinstance(raw_ann, T.LetAnnotation):
        # T.let or T.let[type] -> immutable Bind var
        if rhs is None:
            self.report_error(node, "T.let annotation requires a value")
        rhs = _convert_tuple_literal(self, node.value, rhs)
        if not isinstance(rhs, Expr):
            if isinstance(rhs, str):
                rhs = tvm.tirx.StringImm(rhs)
            else:
                rhs = tvm.tirx.const(rhs)
        if raw_ann.type_spec is not None:
            ann_var = raw_ann.as_var()
        else:
            ann_var = raw_ann.as_var(rhs_dtype=rhs.ty)
        if not isinstance(ann_var, tvm.ir.Var):
            self.report_error(node.annotation, "Annotation should resolve to Var")
        self.eval_assign(target=lhs, source=ann_var, bind_value=bind_assign_value)
        T.Bind(rhs, var=ann_var)
    else:
        ann_var = raw_ann() if callable(raw_ann) and not isinstance(raw_ann, Expr) else raw_ann
        if not isinstance(ann_var, tvm.ir.Var):
            self.report_error(node.annotation, "Annotation should resolve to Var")
        if not isinstance(ann_var.ty, PrimType):
            self.report_error(
                node.annotation,
                "Use T.let[...] for non-PrimType annotations (e.g. PointerType, handle)",
            )
        if str(ann_var.ty) == "handle":
            self.report_error(
                node.annotation,
                "handle type cannot be used as scalar annotation; use T.let[T.handle] instead",
            )
        # x: T.int32 = expr -> scalar (mutable scalar buffer)
        scalar = T.local_scalar(dtype=str(ann_var.ty))
        self.eval_assign(target=lhs, source=scalar, bind_value=bind_assign_value)
        if rhs is not None:
            T.buffer_store(scalar.scalar.buffer, rhs, [0])


@dispatch.register(token="tirx", type_name="With")
def visit_with(self: Parser, node: doc.With) -> None:
    """The with visiting method for tirx.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.With
        The doc AST with node.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(self.var_table.with_frame())
        for item in node.items:
            frame = self.eval_expr(item.context_expr)
            if not isinstance(frame, Frame) and not (
                hasattr(frame, "__enter__") and hasattr(frame, "__exit__")
            ):
                self.report_error(
                    item.context_expr,
                    "Invalid context expression in the with-statement.",
                )
            rhs = stack.enter_context(frame)
            if item.optional_vars is not None:
                self.eval_assign(target=item.optional_vars, source=rhs, bind_value=bind_with_value)
        self.visit_body(node.body)


@dispatch.register(token="tirx", type_name="FunctionDef")
def visit_function_def(self: Parser, node: doc.FunctionDef) -> None:
    """The function definition visiting method for tirx.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.FunctionDef
        The doc AST function definition node.
    """
    supplied_annotation = self.function_annotations
    func_annotation = supplied_annotation.get(node.name, {})
    privacy = find_decorator_annotation(node, "private", default=False)
    s_tir = find_decorator_annotation(node, "s_tir", default=False)
    persistent = find_decorator_annotation(node, "persistent", default=False)
    self.function_annotations = None
    with self.var_table.with_frame(), _signature_match_var_scope(self):
        for name, var in collect_signature_type_vars(self, node).items():
            self.var_table.add(name, var, allow_shadowing=False)
        prim_func_ctx = T.prim_func(is_private=privacy, s_tir=s_tir, persistent=persistent)
        with prim_func_ctx:
            T.func_name(node.name)
            # This is a signature-only dtype index, not lookahead into the body.
            signature_dtypes = _signature_prim_var_dtypes(node)
            with self.with_dispatch_token("tirx"):
                # TODO: handle different types of arguments:
                # - vararg: arg | None
                # - kwonlyargs: list[arg]
                # - kw_defaults: list[expr | None]
                # - kwarg: arg | None
                # - defaults: list[expr]
                # - posonlyargs: list[arg]
                for arg in node.args.args:
                    if arg.annotation is None:
                        self.report_error(arg, "Type annotation required for function parameters.")
                    try:
                        ann = _eval_signature_annotation(self, arg.annotation, signature_dtypes)
                    except Exception:  # pylint: disable=broad-except
                        ann = func_annotation.get(arg.arg, None)
                        if ann is None:
                            raise
                    if isinstance(ann, _OptionalAnnotation):
                        if not _is_jit_function(node):
                            self.report_error(
                                arg.annotation,
                                "T.Optional parameter annotations are only supported by @T.jit",
                            )
                        if arg.arg in self.absent_params:
                            self.var_table.add(arg.arg, None)
                            continue
                        ann = ann.annotation
                    elif arg.arg in self.absent_params:
                        self.report_error(
                            arg.annotation,
                            "Only T.Optional parameters may be absent, "
                            f"but {arg.arg!r} is not optional",
                        )
                    if (
                        callable(ann)
                        and not isinstance(ann, Expr)
                        and ann is not _constexpr_sentinel
                    ):
                        ann = ann()
                    if ann is _constexpr_sentinel:
                        # T.constexpr param: value was bound in extra_vars by
                        # TIRJit.specialize() and lives in an outer var_table
                        # frame; do not register a runtime PrimFunc param.
                        continue
                    ann = _reuse_signature_match_var(self, arg.annotation, arg.arg, ann)
                    param = T.arg(arg.arg, ann)
                    self.var_table.add(arg.arg, param)

                if node.returns is not None:
                    ret_type = _eval_signature_annotation(
                        self, node.returns, signature_dtypes, define_missing=False
                    )
                    if callable(ret_type) and not isinstance(ret_type, Expr):
                        ret_type = ret_type()
                    if isinstance(ret_type, Expr):
                        ret_type = ret_type.ty
                    T.func_ret(ret_type)
                self.visit_body(node.body)
    self.function_annotations = supplied_annotation


@dispatch.register(token="tir.inline", type_name="FunctionDef")
def visit_inline_function_def(self: Parser, node: doc.FunctionDef) -> None:
    """The function definition visiting method for inline functions in tir.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.FunctionDef
        The doc AST function definition node.
    """
    # remove the inline decorator
    node.decorator_list.pop()
    # adjust the node location to the source code location
    node.lineno += self.diag.source.start_line - 1
    node.col_offset += self.diag.source.start_column + 1
    node.end_lineno += self.diag.source.start_line - 1
    node.end_col_offset += self.diag.source.start_column + 1

    # Record definition depth for LEGB late binding
    definition_depth = len(self.var_table.frames)

    def get_func():
        func_ast = from_doc(node)
        module_ast = ast.Module(body=[func_ast], type_ignores=[])
        ast.fix_missing_locations(module_ast)
        # set the filename to the source name, so that the error message can be reported correctly
        code_obj = compile(module_ast, filename=self.diag.source.source_name, mode="exec")
        namespace = self.var_table.get()
        exec(code_obj, namespace)  # pylint: disable=exec-used
        func_name = func_ast.name
        func = namespace[func_name]
        return func, func_name

    func, func_name = get_func()
    wrapper = inline(func, definition_depth=definition_depth, defining_var_table=self.var_table)

    self.var_table.add(func_name, wrapper, allow_shadowing=False)
    return None


@dispatch.register(token="tirx", type_name="tvm_annotation")
def visit_tvm_annotation(self: Parser, node: doc.expr):
    """The TVM annotation visiting method for tirx.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.expr
        The doc AST expr node.
    """
    annotation = self.eval_expr(node)
    if callable(annotation) and not isinstance(annotation, Expr):
        annotation = annotation()
    return annotation


@dispatch.register(token="tirx", type_name="Expr")
def visit_expr_stmt(self: Parser, node: doc.Expr) -> None:
    """The expr statement visiting method for tirx.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Expr
        The doc AST Expr node.
    """

    res = self.eval_expr(node.value)
    if res is None:
        pass
    elif isinstance(res, Frame):
        res.add_callback(partial(res.__exit__, None, None, None))
        res.__enter__()
    elif hasattr(res, "frames") and hasattr(res, "__enter__"):
        # _FrameScope from T.attr({...}) — enter each inner frame for concise scoping
        for f in res.frames:
            f.add_callback(partial(f.__exit__, None, None, None))
            f.__enter__()
    elif isinstance(res, tvm.ir.Var):
        # Standalone Var expression (e.g. from T.bind(value, var=v)) --
        # the Bind statement was already emitted to the parent frame by the FFI call,
        # so just discard the returned Var.
        pass
    elif tvm.ir.is_prim_expr(res):
        T.evaluate(res)
    elif isinstance(res, int | bool):
        T.evaluate(tvm.tirx.const(res))
    elif isinstance(res, tvm.ir.Call) and not tvm.ir.is_prim_expr(res):
        if isinstance(res.op, tvm.ir.GlobalVar) and res.ty.is_missing():
            # GlobalVar calls with a missing return type are ambiguous, as each IR has a
            # different function Call representation. Convert to the TIR representation.
            T.evaluate(tvm.tirx.call_tir(res.op, *res.args))
        else:
            # Pointer-valued TIR calls are general Expr rather than Expr,
            # but are still valid standalone Evaluate statements.
            T.evaluate(res)
    elif isinstance(res, str):
        # Ignore docstrings
        pass
    elif isinstance(res, tvm.tirx.stmt.BufferStore):
        T.buffer_store(res.buffer, res.value, res.indices, res.predicate)
    elif is_buffer_var(res):
        # ``T.match_buffer(...)`` used as a bare statement (no LHS) — the
        # buffer object is discarded; the underlying side effect (the
        # match_buffer node) has already been emitted into the frame.
        pass
    else:
        self.report_error(node, f"Parsing resulted in unexpected type {type(res)}")


@dispatch.register(token="tirx", type_name="If")
def visit_if(self: Parser, node: doc.If) -> None:
    """The if visiting method for tirx.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.If
        The doc AST if node.
    """
    predicate = self.eval_expr(node.test)
    if tvm.ir.is_prim_expr(predicate) or isinstance(predicate, tvm.tirx.expr.ExprOp):
        with T.If(predicate):
            with T.Then():
                with self.var_table.with_frame():
                    self.visit_body(node.body)
            if node.orelse:
                with T.Else():
                    with self.var_table.with_frame():
                        self.visit_body(node.orelse)
    elif isinstance(predicate, bool):
        # Python ``if`` does not introduce a lexical scope.  Bindings from the
        # selected compile-time branch must therefore remain visible after it.
        if predicate:
            self.visit_body(node.body)
        elif node.orelse:
            self.visit_body(node.orelse)
    else:
        self.report_error(
            node.test,
            f"If condition must be a boolean expression, but got {predicate}",
        )


@dispatch.register(token="tirx", type_name="Assert")
def visit_assert(self: Parser, node: doc.Assert) -> None:
    """The assert visiting method for tirx.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Assert
        The doc AST assert node.

    The assert message can be either:
    - A plain string: ``assert cond, "message"``
    - A tuple of (kind, [parts...]): ``assert cond, ("ValueError", ["part0", "part1"])``
    """
    cond = self.eval_expr(node.test)
    msg = self.eval_expr(node.msg)

    kind = "RuntimeError"
    message = msg

    if isinstance(msg, tuple):
        if len(msg) != 2:
            self.report_error(
                node,
                f"Assert message tuple must have exactly 2 elements (kind, [parts...]), "
                f"got {len(msg)} elements",
            )
        kind_str, parts = msg
        if isinstance(kind_str, tvm.tirx.StringImm):
            kind_str = kind_str.value
        if not isinstance(kind_str, str):
            self.report_error(
                node,
                f"Assert message tuple first element must be a string (error kind like "
                f'"ValueError"), got {type(kind_str).__name__}',
            )
        kind = kind_str
        message = parts

    if isinstance(message, list | tuple):
        message = [p.value if isinstance(p, tvm.tirx.StringImm) else str(p) for p in message]

    frame = T.Assert(cond, message, error_kind=kind)
    frame.add_callback(partial(frame.__exit__, None, None, None))
    frame.__enter__()


@dispatch.register(token="tirx", type_name="Return")
def visit_return(self: Parser, node: doc.Return) -> None:
    """The return visiting method for tirx.

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Return
        The doc AST return node.
    """
    value = self.eval_expr(node.value)
    if value is None:
        self.report_error(node, "Expression to be returned must be a Expr")
    T.Return(value)


@dispatch.register(token="tirx", type_name="tvm_declare_function")
def visit_tvm_declare_function(self: Parser, node: doc.FunctionDef) -> GlobalVar:
    """The function declaration step for tirx

    Parameters
    ----------
    self : Parser
        The visiting parser.

    node : doc.Return
        The doc AST return node.
    """

    supplied_annotation = self.function_annotations
    func_annotation = supplied_annotation.get(node.name, {})

    ret_type = None
    with self.var_table.with_frame(), _signature_match_var_scope(self):
        signature_dtypes = _signature_prim_var_dtypes(node)
        for name, var in collect_signature_type_vars(self, node).items():
            self.var_table.add(name, var, allow_shadowing=False)

        arg_annotations = []
        for arg in node.args.args:
            if arg.annotation is None:
                self.report_error(arg, "Type annotation required for function parameters.")
            try:
                ann = _eval_signature_annotation(self, arg.annotation, signature_dtypes)
                if callable(ann) and not isinstance(ann, Expr):
                    ann = ann()
            except Exception:  # pylint: disable=broad-except
                ann = func_annotation.get(arg.arg, None)
                if ann is None:
                    raise

            ann = _reuse_signature_match_var(self, arg.annotation, arg.arg, ann)
            IRBuilder.name(arg.arg, ann)
            self.var_table.add(arg.arg, ann)
            arg_annotations.append(ann)

        if node.returns is not None:
            ret_type = _eval_signature_annotation(
                self, node.returns, signature_dtypes, define_missing=False
            )
            if callable(ret_type) and not isinstance(ret_type, Expr):
                ret_type = ret_type()
            if isinstance(ret_type, Expr):
                ret_type = ret_type.ty

    func_signature = tvm.tirx.PrimFunc(arg_annotations, None, ret_type=ret_type)
    return I.decl_function(node.name, func_signature)
