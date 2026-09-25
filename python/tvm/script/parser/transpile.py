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
"""Rewrite an entry-owned source AST into an executable native-builder AST.

Entry acquires a fresh source AST and collects its syntax facts in a
``PrescanContext``. The recursive rewriter consumes those facts without mutating
their collections. A ``ModuleContext`` references them and shares translation
inputs, generated-name allocation and injected bindings across the whole parse,
including when the root is a standalone function. A ``FunctionContext`` holds
the active function's lexical rewrite state. Each function, including a nested
function, gets a fresh context; the enclosing context is restored on both normal
and exceptional exit.

These contexts are temporary Python translation state. They refer to the one
entry-owned definition scope and hand generated helpers to private recomposition
in one direction, without back-references to the rewriter. Entry executes the
recomposed builder and releases temporary captures. Short-lived frame names,
statement lists and assembly results stay local to the methods that need them.
Native frames own symbols, declarations, parameters, region results and final
IR; the Python contexts do not mirror that construction state.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from types import FunctionType
from typing import Any, NamedTuple, NoReturn, TypeVar

from . import protocol_registry as protocol
from .annotation import parse_annotation
from .inspect_source import _AnnotationScope
from .prescan import (
    Binding,
    PrescanContext,
    collect_annotation_free_names,
    collect_annotation_free_reads,
)

_Node = TypeVar("_Node", bound=ast.AST)
_Value = TypeVar("_Value")


def _require_constexpr_arg(value: _Value, name: str) -> _Value:
    """Return the constexpr argument value, or raise if it is missing.

    Parameters
    ----------
    value : Any
        Selected or captured compile-time value. Only the builder's MISSING
        sentinel denotes an absent binding; None is an explicit valid value.
    name : str
        Source parameter name included in a missing-binding diagnostic.

    Returns
    -------
    Any
        The identical input value, preserving its Python type and identity.

    Raises
    ------
    TypeError
        If value is the MISSING sentinel and no constexpr binding was selected.
    """
    from tvm.script.ir_builder.base import MISSING

    if value is MISSING:
        raise TypeError(f"constexpr parameter {name!r} requires a specialization binding")
    return value


def _require_annotation_value(value: _Value, name: str) -> _Value:
    """Keep captured values unchanged, reporting missing names only when read."""
    from tvm.script.ir_builder.base import MISSING

    if value is MISSING:
        raise NameError(f"name {name!r} is not defined")
    return value


def _unwrap_optional_annotation(
    annotation: Any, const_args: Mapping[str, Any] | None = None
) -> Any:
    """Read an optional runtime annotation only within a JIT specialization.

    Parameters
    ----------
    annotation : Any
        Evaluated runtime annotation. An object implementing the callable
        ``__tvm_optional_annotation__`` adapter supplies its contained annotation;
        all other objects pass through unchanged.
    const_args : Mapping[str, Any] or None, optional
        Parameter names mapped to fixed values for the active root. None, the
        default, means ordinary parsing and forbids optional annotation adapters. Any
        mapping, including an empty mapping, enables the adapter. Its contents
        are not inspected here.

    Returns
    -------
    Any
        The adapter's result, or the identical input object when no adapter exists.

    Raises
    ------
    TypeError
        If an optional annotation adapter is used without specialization.

    Notes
    -----
    Generated specialization calls this after checking selected values and
    absences, so omitted parameters never evaluate their annotation. Ordinary
    argument construction delegates annotation validation to the language variant.
    Adapter lookup and execution exceptions propagate unchanged.
    """
    unwrap = getattr(annotation, "__tvm_optional_annotation__", None)
    if unwrap is not None:
        if const_args is None:
            raise TypeError("T.Optional is only supported by @T.jit")
        return unwrap()
    return annotation


class GeneratedBuilder(NamedTuple):
    """Generated helper AST that builds one IR function body, with source metadata.

    This per-function helper runs inside the outer execution wrapper emitted
    by entry; recomposition restores its source globals and closure bindings.
    """

    # Generated helper AST adjusted by _recompose_builder before compilation.
    body: ast.FunctionDef
    # Original Python function lookup key used to inspect its globals/closure.
    original_func_name: str
    # Generated bindings preserved while restoring source globals/closures.
    protected_names: set[str]


class ModuleContext:
    """Share inputs and generated bindings for one temporary translation.

    Entry's preparation creates this context for a module, standalone function
    or macro. The recursive rewriter shares it across lexical function contexts;
    private recomposition then consumes its generated-helper handoff. References
    last only through that parse or macro invocation. There is one shared root
    definition scope, no per-function copy and no reference back to the rewriter.
    """

    def __init__(
        self,
        filename: str,
        environment: Mapping[str, object],
        ir_prefix: str,
        make_span_expr: Callable[[ast.AST], ast.expr],
        make_fresh_name: Callable[[str], str],
        *,
        prescan_ctx: PrescanContext,
        track_span: bool,
        enable_jit_map: bool = False,
        definition_scope: Mapping[str, Any],
        original_func_map: Mapping[str, FunctionType],
        exec_globals: dict[str, Any],
        root_function_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        # Source filename used by _raise_error and parse_annotation diagnostics.
        self.filename = filename
        # Definition/lexical lookup layers used to identify construction namespaces.
        # These preserve source meanings independently of injected execution globals.
        self.environment = environment
        # Generated name of the shared IR namespace used for module frames and MISSING.
        self.ir_prefix = ir_prefix
        # Entry callback creates/reuses a SpanEntry and emits its AST table reference
        # for _call_dialect, binding locations and expression instrumentation.
        self.make_span_expr = make_span_expr
        # _call and _attach_span omit native source instrumentation when false.
        self.track_span = track_span
        # Emit reading/use of the selected root JIT map, including an empty map;
        # create_function_builder_fragments leaves nested signatures unspecialized.
        self.enable_jit_map = enable_jit_map
        # Read-only syntax facts used by binding rewrites, annotation declarations,
        # namespace protection and declaration-before-body selection.
        self.prescan_ctx = prescan_ctx
        # Actual root definition bindings supplied to _recompose_builder defaults;
        # annotation rewriting distinguishes available captures from lazy lookups.
        self.definition_scope = definition_scope
        # Original Python functions by source name; rewrite_module preserves pyfunc
        # identity instead of recreating those ordinary callables from their AST.
        self.original_func_map = original_func_map
        # Entry-owned allocator used by rewriting/recomposition for generated names;
        # it reserves source identifiers without renaming them.
        self.make_fresh_name = make_fresh_name
        # Source values plus injected objects, copied into generated execution globals
        # by _recompose_builder; _inject adds collision-free internal bindings here.
        self.exec_globals = exec_globals
        # Evaluated kwargs passed to function_() for the root function:
        # the standalone function being parsed. Nested functions and IRModule
        # members obtain their options from their own decorators.
        self.root_function_kwargs = root_function_kwargs
        # rewrite_module sets the root class name, or None for a standalone function.
        # Reference rewrites use it to recognize lexical module aliases.
        self.module_name: str | None = None
        # IR global-function names in this parse, including a standalone root.
        # visit_Call uses these to lower global calls; recomposition preserves
        # their generated bindings when restoring source globals and closures.
        self.global_func_names: frozenset[str] = frozenset()
        # create_function_builder_fragments appends per-function helper metadata;
        # _recompose_builder restores each helper's original globals and closures.
        self.generated_builders: list[GeneratedBuilder] = []


class FunctionContext:
    """Hold only the active lexical function's temporary rewrite state.

    Entry creates the initial context. Function lowering creates a fresh context
    for each source function, including nested functions, and the same recursive
    visitor consumes it. The enclosing context stays in a local variable and is
    restored in ``finally``, so neither successful nor failed nesting leaks state.
    """

    def __init__(self, current_scope: ast.AST | None, dialect_prefix: str) -> None:
        # Active source function (or None before root lowering); declaration rewriting
        # uses it to select lexical prescan facts.
        self.current_scope = current_scope
        # Namespace identifier used by _call_dialect for this function's builder operations.
        self.dialect_prefix = dialect_prefix
        # One snapshot supplies original-name helper defaults at the definition site.
        self.definition_captures: str | None = None


class IRBuilderTranspiler(ast.NodeTransformer):
    """A single statement/expression visitor over the entry-owned AST.

    The module context shares source inputs, name allocation and injected values.
    The active function context selects lexical facts and definition captures.
    Function entry and temporary expression modes save and restore their state at
    lexical boundaries, including failures. Generated AST calls construct native
    frames at execution time; this visitor does not own their IR state.
    """

    def __init__(
        self, module: ModuleContext, function: FunctionContext, *, preserve_return: bool = False
    ) -> None:
        # Contexts own syntax inputs only and never refer back to this rewriter.
        self.module = module
        self.function = function
        # Macro return policy is fixed for one invocation.
        self.preserve_return = preserve_return
        # Bypass syntax-to-builder lowering.
        # Still traverse children and instrument source calls with spans.
        self.bypass_ast_rewrite = False
        # Only free reads of the active annotation need lazy missing-name checks.
        self.annotation_reads: set[ast.Name] = set()
        # Only this expression's result is already located by its enclosing emit_.
        # Child operations retain their locations; restore the borrowed node on exit.
        self.emitted_expression: ast.expr | None = None
        # An ordinary binding owns this RHS's attachment through value_span;
        # nested operations still receive their own locations and call contexts.
        self.binding_expression: ast.expr | None = None

    def _inject(self, value: object, prefix: str = "_host") -> ast.Name:
        name = self.module.make_fresh_name(prefix)
        self.module.exec_globals[name] = value
        return ast.Name(name, ast.Load())

    def _raise_error(self, node: ast.AST, message: str) -> NoReturn:
        raise SyntaxError(
            message,
            (
                self.module.filename,
                node.lineno,
                node.col_offset + 1,
                None,
                node.end_lineno,
                node.end_col_offset + 1,
            ),
        )

    def _call(
        self,
        namespace: str,
        member: str,
        args: list[ast.expr],
        node: ast.AST,
        *,
        keywords: Mapping[str, ast.expr] | None = None,
        span: ast.expr | None = None,
        name_span: ast.expr | None = None,
        value_span: ast.expr | None = None,
    ) -> ast.Call:
        """Build a generated operation with its source range and named arguments."""
        arguments = [ast.keyword(key, value) for key, value in keywords.items()] if keywords else []
        if self.module.track_span:
            if span is not None:
                arguments.append(ast.keyword("span", span))
            if name_span is not None:
                arguments.append(ast.keyword("name_span", name_span))
            if value_span is not None:
                arguments.append(ast.keyword("value_span", value_span))
        return ast.copy_location(
            ast.Call(
                ast.Attribute(ast.Name(namespace, ast.Load()), member, ast.Load()),
                args,
                arguments,
            ),
            node,
        )

    def _call_dialect(
        self,
        member: str,
        args: list[ast.expr],
        node: ast.AST,
        *,
        keywords: Mapping[str, ast.expr] | None = None,
        span: ast.expr | None = None,
        name_span: ast.expr | None = None,
        value_span: ast.expr | None = None,
    ) -> ast.Call:
        # Dialect calls evaluate their span, name span, builder arguments, then value span.
        call = self._call(
            self.function.dialect_prefix,
            member,
            args,
            node,
            span=self.module.make_span_expr(node) if span is None else span,
            name_span=name_span,
        )
        if keywords:
            call.keywords.extend(ast.keyword(key, value) for key, value in keywords.items())
        if self.module.track_span and value_span is not None:
            call.keywords.append(ast.keyword("value_span", value_span))
        return call

    def _attach_span(self, value: ast.expr, node: ast.AST) -> ast.expr:
        if (
            not self.module.track_span
            or node is self.emitted_expression
            or node is self.binding_expression
        ):
            return value
        # _S[i](value)
        return ast.copy_location(ast.Call(self.module.make_span_expr(node), [value], []), node)

    @staticmethod
    def _assign(name: str, value: ast.expr, node: ast.AST) -> ast.Assign:
        """Assign an injected or source name while retaining its source range."""
        return ast.copy_location(ast.Assign([ast.Name(name, ast.Store())], value), node)

    @staticmethod
    def _create_lambda(names: list[str], value: ast.expr) -> ast.Lambda:
        # lambda first, second: value
        return ast.Lambda(
            ast.arguments(
                posonlyargs=[],
                args=[ast.arg(name) for name in names],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            value,
        )

    @staticmethod
    def _create_definition(name: str, body: list[ast.stmt], node: ast.AST) -> ast.FunctionDef:
        # def build(): <translated statements>
        definition = ast.copy_location(
            ast.FunctionDef(
                name,
                ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body or [ast.Pass()],
                [],
                None,
            ),
            node,
        )
        if "type_params" in ast.FunctionDef._fields:
            definition.type_params = []
        return definition

    def transform_statements(self, body: list[ast.stmt]) -> list[ast.stmt]:
        """Visit source statements once, flattening statement-list rewrites."""
        result = []
        for statement in body:
            rewritten = self.visit(statement)
            if rewritten is not None:
                result.extend(rewritten if isinstance(rewritten, list) else [rewritten])
        return result

    @contextmanager
    def _bypass_rewrite(self) -> Iterator[None]:
        # Only constexpr operands and module host syntax keep Python operators.
        # The same visitor still instruments their source calls and restores mode.
        old = self.bypass_ast_rewrite
        self.bypass_ast_rewrite = True
        try:
            yield
        finally:
            self.bypass_ast_rewrite = old

    @contextmanager
    def _rewrite_annotation(self, node: ast.expr) -> Iterator[None]:
        """Check missing values only at free reads, without renaming Python bindings."""
        previous_reads = self.annotation_reads
        self.annotation_reads = set(collect_annotation_free_reads(node))
        try:
            yield
        finally:
            self.annotation_reads = previous_reads

    def _read_constexpr_operand(self, node: ast.expr) -> ast.expr | None:
        # -------------------- Pattern --------------------
        # Python source:
        #     I.constexpr(expr)
        #
        # Builder:
        #     expr
        # -------------------------------------------------
        # The same visitor keeps Python operators and still instruments nested source calls.
        if not isinstance(node, ast.Call) or not self._is_constexpr_annotation(node.func):
            return None
        if len(node.args) != 1 or node.keywords or isinstance(node.args[0], ast.Starred):
            self._raise_error(node, "constexpr expects exactly one controlling value")
        return node.args[0]

    def visit_Name(self, node: ast.Name) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     value: X.Tensor((n,))
        #
        # Builder:
        #     value_type = X.Tensor((_require_annotation_value(n, "n"),))
        # -------------------------------------------------
        # Helpers bind captured values under their original names. Python handles
        # lambda/comprehension locals and preceding signature parameters directly.
        if node in self.annotation_reads:
            if node.id in self.module.prescan_ctx.namespaces:
                return node
            return ast.copy_location(
                ast.Call(
                    self._inject(_require_annotation_value), [node, ast.Constant(node.id)], []
                ),
                node,
            )
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     Module.f
        #
        # Builder:
        #     _S[i](Module.f)
        # -------------------------------------------------
        # Keeping the module owner prevents a local f from shadowing its GlobalVar.
        root = node
        while isinstance(root, ast.Attribute):
            root = root.value
        fixed_namespace = (
            isinstance(root, ast.Name) and root.id in self.module.prescan_ctx.namespaces
        )
        result = self.generic_visit(node)
        return (
            self._attach_span(result, node)
            if isinstance(node.ctx, ast.Load)
            and not self.bypass_ast_rewrite
            and not fixed_namespace
            else result
        )

    def visit_Constant(self, node: ast.Constant) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     42
        #
        # Builder:
        #     42
        # -------------------------------------------------
        return node

    def visit_List(self, node: ast.List | ast.Tuple | ast.Set | ast.Dict) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     [a, b]
        #
        # Builder:
        #     _S[i]([a, b])
        # -------------------------------------------------
        result = self.generic_visit(node)
        return result if self.bypass_ast_rewrite else self._attach_span(result, node)

    visit_Tuple = visit_List
    visit_Set = visit_List
    visit_Dict = visit_List

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.JoinedStr:
        # -------------------- Pattern --------------------
        # Python source:
        #     f"value={expr}"
        #
        # Builder:
        #     f"value={expr}"
        # -------------------------------------------------
        # Literal fragments remain Constant/FormattedValue nodes required by Python.
        for child in node.values:
            if isinstance(child, ast.FormattedValue):
                child.value = self.visit(child.value)
                if child.format_spec is not None:
                    child.format_spec = self.visit_JoinedStr(child.format_spec)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     buffer[index]
        #
        # Builder:
        #     _S[i](buffer[index])
        # -------------------------------------------------
        node.value = self.visit(node.value)
        node.slice = self._rewrite_index(node.slice)
        if not self.bypass_ast_rewrite and isinstance(node.ctx, ast.Load):
            return self._attach_span(node, node)
        return node

    def _is_module_owner(self, node: ast.expr) -> bool:
        """Recognize fixed source module aliases from existing binding records."""
        if not isinstance(node, ast.Name):
            return False
        if node.id == self.module.module_name:
            return True
        records = [
            item
            for item in self.module.prescan_ctx.bindings.get(self.function.current_scope, ())
            if item.name == node.id
        ]
        return bool(records) and all(item.kind == "module_alias" for item in records)

    def _visit_direct_operand(self, node: ast.expr) -> ast.expr:
        """Preserve an existing payload's span without bypassing child operations."""
        if isinstance(node, ast.Name):
            return self.visit(node)
        if isinstance(node, ast.Attribute):
            node.value = self._visit_direct_operand(node.value)
            return node
        if isinstance(node, ast.Tuple | ast.List):
            node.elts = [self._visit_direct_operand(value) for value in node.elts]
            return node
        if isinstance(node, ast.Starred):
            node.value = self._visit_direct_operand(node.value)
            return node
        return self.visit(node)

    def visit_Call(self, node: ast.Call, *, callee: ast.expr | None = None) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     X.Tensor((n,), vdevice="cuda:0")
        #
        # Builder:
        #     X.Tensor((n,), vdevice="cuda:0")
        # -------------------------------------------------
        binding_value = node is self.binding_expression
        marker = self._read_constexpr_operand(node)
        if marker is not None:
            with self._bypass_rewrite():
                return self.visit(marker)
        constructor = self.module.prescan_ctx._match_special_func(node.func)
        global_call = (
            isinstance(node.func, ast.Name) and node.func.id in self.module.global_func_names
        ) or (
            isinstance(node.func, ast.Attribute)
            and self._is_module_owner(node.func.value)
            and node.func.attr in self.module.global_func_names
        )
        # Visit source callee/arguments first. A normalized range callee is
        # assembled afterward, but its original arguments keep normal rewriting.
        if callee is None:
            node.func = self._visit_direct_operand(node.func)
        node.args = [self.visit(value) for value in node.args]
        for keyword in node.keywords:
            keyword.value = self.visit(keyword.value)
        if callee is not None:
            node.func = callee
        # -------------------- Pattern --------------------
        # Python source:
        #     declared_global(x, y)
        #
        # Builder:
        #     X.call_global_var_(declared_global, [x, y])
        # -------------------------------------------------
        if global_call and not self.bypass_ast_rewrite:
            if node.keywords:
                self._raise_error(node, "Global function calls require positional arguments")
            node = self._call(
                self.function.dialect_prefix,
                "call_global_var_",
                [node.func, ast.List(node.args, ast.Load())],
                node,
            )
        # -------------------- Pattern --------------------
        # Python source:
        #     f(a)
        #
        # Builder:
        #     _S[i].ctx(lambda: f(a))
        # -------------------------------------------------
        # Calls retain their construction context; bind_ owns ordinary RHS attribution.
        if self.module.track_span and callee is None:
            if protocol.RESULT_SPAN.get(constructor, False):
                return node if binding_value else self._attach_span(node, node)
            # _S[i].ctx(lambda: callee(*args, **keywords))
            return ast.copy_location(
                ast.Call(
                    ast.Attribute(self.module.make_span_expr(node), "ctx", ast.Load()),
                    [self._create_lambda([], node)],
                    [ast.keyword("attach_result", ast.Constant(False))] if binding_value else [],
                ),
                node,
            )
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     not x
        #
        # Builder:
        #     X.not_(x)
        # -------------------------------------------------
        # An explicit constexpr operand retains ordinary Python not.
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Not) and not self.bypass_ast_rewrite:
            return self._attach_span(
                self._call(self.function.dialect_prefix, "not_", [node.operand], node), node
            )
        return self._attach_span(node, node) if not self.bypass_ast_rewrite else node

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     a + b
        #
        # Builder:
        #     _S[i](a + b)
        # -------------------------------------------------
        return (
            self._attach_span(self.generic_visit(node), node)
            if not self.bypass_ast_rewrite
            else self.generic_visit(node)
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     I.constexpr(enabled) and expr
        #
        # Builder:
        #     enabled and expr
        # -------------------------------------------------
        # Unmarked IR operands use X.and_ or X.or_.
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)

        def lower(position: int) -> ast.expr:
            value = node.values[position]
            marker = self._read_constexpr_operand(value)
            if marker is not None:
                with self._bypass_rewrite():
                    left = self.visit(marker)
                if position + 1 == len(node.values):
                    return left
                return ast.copy_location(ast.BoolOp(node.op, [left, lower(position + 1)]), node)
            left = self.visit(value)
            method = "and_" if isinstance(node.op, ast.And) else "or_"
            for index in range(position + 1, len(node.values)):
                if self._read_constexpr_operand(node.values[index]) is not None:
                    return self._call(
                        self.function.dialect_prefix, method, [left, lower(index)], node
                    )
                left = self._call(
                    self.function.dialect_prefix,
                    method,
                    [left, self.visit(node.values[index])],
                    node,
                )
            return left

        return self._attach_span(lower(0), node)

    def visit_IfExp(self, node: ast.IfExp) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     yes if I.constexpr(test) else no
        #
        # Builder:
        #     yes if test else no
        # -------------------------------------------------
        # Unmarked tests use X.if_then_else_.
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        marker = self._read_constexpr_operand(node.test)
        if marker is not None:
            with self._bypass_rewrite():
                test = self.visit(marker)
            return ast.copy_location(
                ast.IfExp(test, self.visit(node.body), self.visit(node.orelse)), node
            )
        return self._attach_span(
            self._call(
                self.function.dialect_prefix,
                "if_then_else_",
                [self.visit(node.test), self.visit(node.body), self.visit(node.orelse)],
                node,
            ),
            node,
        )

    def _create_comparison(
        self, left: ast.expr, operation: ast.cmpop, right: ast.expr, node: ast.AST
    ) -> ast.expr:
        operations = {
            ast.Lt: "lt_",
            ast.LtE: "le_",
            ast.Gt: "gt_",
            ast.GtE: "ge_",
            ast.Eq: "eq_",
            ast.NotEq: "ne_",
        }
        if type(operation) in operations:
            return self._call(
                self.function.dialect_prefix, operations[type(operation)], [left, right], node
            )
        return ast.copy_location(ast.Compare(left, [operation], [right]), node)

    def visit_Compare(self, node: ast.Compare) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     0 < i < 10
        #
        # Builder:
        #     X.and_(X.lt_(0, i), X.lt_(i, 10))
        # -------------------------------------------------
        # Only names and numeric literals may be repeated in a simple chain.
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        operands = [node.left, *node.comparators]
        if len(node.ops) > 1:
            for operand in operands:
                numeric = operand
                if isinstance(numeric, ast.UnaryOp) and isinstance(numeric.op, ast.UAdd | ast.USub):
                    numeric = numeric.operand
                if not isinstance(operand, ast.Name) and not (
                    isinstance(numeric, ast.Constant)
                    and type(numeric.value) in (int, float, complex)
                ):
                    self._raise_error(
                        operand, "Comparison chains support only names and numeric literals"
                    )
        values = [self.visit(operand) for operand in operands]
        comparisons = [
            self._create_comparison(left, operation, right, node)
            for left, operation, right in zip(values, node.ops, values[1:])
        ]
        result = (
            comparisons[0]
            if len(comparisons) == 1
            else self._call(self.function.dialect_prefix, "and_", comparisons, node)
        )
        return self._attach_span(result, node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> NoReturn:
        # -------------------- Pattern --------------------
        # Python source:
        #     (x := value)
        #
        # Builder:
        #     raise SyntaxError("Assignment expressions are unsupported")
        # -------------------------------------------------
        # Source assignment expressions have no builder declaration contract.
        self._raise_error(node, "Unsupported expression: NamedExpr")

    def visit_Await(self, node: ast.Await | ast.Yield | ast.YieldFrom) -> NoReturn:
        # -------------------- Pattern --------------------
        # Python source:
        #     await expression
        #
        # Builder:
        #     raise SyntaxError("Async and generator expressions are unsupported")
        # -------------------------------------------------
        self._raise_error(node, f"Unsupported expression: {type(node).__name__}")

    visit_Yield = visit_Await
    visit_YieldFrom = visit_Await

    def _rewrite_index(self, node: ast.expr) -> ast.expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     a[start:stop:step] = value
        #
        # Builder:
        #     X.setitem_(a, slice(start, stop, step), value)
        # -------------------------------------------------
        if isinstance(node, ast.Slice):
            return ast.copy_location(
                ast.Call(
                    self._inject(slice),
                    [
                        self.visit(value) if value else ast.Constant(None)
                        for value in (node.lower, node.upper, node.step)
                    ],
                    [],
                ),
                node,
            )
        if isinstance(node, ast.Tuple):
            return ast.copy_location(
                ast.Tuple([self._rewrite_index(value) for value in node.elts], ast.Load()), node
            )
        return self.visit(node)

    def _binding_kind(self, target: ast.Name, *, frame_value: bool = False) -> str:
        """Select the declaration/store/binding operation from existing syntax facts."""
        if frame_value:
            return "ordinary"
        site = self.module.prescan_ctx.sites.get(target)
        kind = site.kind if site is not None else "ordinary"
        if kind in ("mutable", "module_alias"):
            return kind
        mutable = self.module.prescan_ctx.mutable_names.get(self.function.current_scope, ())
        return "mutable_update" if target.id in mutable else "ordinary"

    def _uses_ordinary_binding(self, target: ast.expr) -> bool:
        """Test assignment targets without evaluating or inspecting their RHS values."""
        if isinstance(target, ast.Name):
            return self._binding_kind(target) == "ordinary"
        if isinstance(target, ast.Starred):
            return self._uses_ordinary_binding(target.value)
        if isinstance(target, ast.Tuple | ast.List):
            return any(self._uses_ordinary_binding(item) for item in target.elts)
        return False

    def _rewrite_assignment_value(self, node: ast.expr, *, ordinary: bool) -> ast.expr:
        """Leave ordinary RHS attachment to bind_ while preserving nested instrumentation."""
        previous, self.binding_expression = self.binding_expression, node if ordinary else None
        try:
            return self.visit(node)
        finally:
            self.binding_expression = previous

    def _bind(
        self,
        target: ast.expr,
        value: ast.expr,
        statement: ast.stmt,
        *,
        ty: ast.expr | None = None,
        frame_value: bool = False,
        value_span: ast.expr | None = None,
    ) -> list[ast.stmt]:
        # Declaration syntax has precedence; no previous/existence tracking.
        if isinstance(target, ast.Name):
            kind = self._binding_kind(target, frame_value=frame_value)
            keywords = {"name": ast.Constant(target.id)}
            if ty is not None:
                keywords["ty"] = ty
            if frame_value:
                keywords["frame_value"] = ast.Constant(True)
            if kind == "mutable" and not frame_value:
                # -------------------- Pattern --------------------
                # Python source:
                #     x = X.local_scalar(initial)
                #
                # Builder:
                #     x = X.decl_mutable_cell_(X.local_scalar(initial), name="x")
                # -------------------------------------------------
                value = self._call_dialect(
                    "decl_mutable_cell_",
                    [value],
                    statement,
                    name_span=self.module.make_span_expr(target),
                    keywords=keywords,
                )
            elif kind == "module_alias" and not frame_value:
                # -------------------- Pattern --------------------
                # Python source:
                #     alias = Module
                #
                # Builder:
                #     alias = Module
                # -------------------------------------------------
                pass
            elif kind == "mutable_update":
                # -------------------- Pattern --------------------
                # Python source:
                #     x = value
                #
                # Builder:
                #     X.set_mutable_cell_(x, value)
                # -------------------------------------------------
                # The prescan identifies x as mutable.
                return [
                    ast.copy_location(
                        ast.Expr(
                            self._call_dialect(
                                "set_mutable_cell_",
                                [ast.Name(target.id, ast.Load()), value],
                                statement,
                            )
                        ),
                        statement,
                    )
                ]
            else:
                # -------------------- Pattern --------------------
                # Python source:
                #     y = value
                #
                # Builder:
                #     y = X.bind_(value, name="y", span=target_span, value_span=rhs_span)
                # -------------------------------------------------
                value = (
                    self._call_dialect(
                        "bind_",
                        [value],
                        statement,
                        name_span=self.module.make_span_expr(target),
                        keywords=keywords,
                        value_span=value_span,
                    )
                    if frame_value
                    else self._call_dialect(
                        "bind_", [value], target, keywords=keywords, value_span=value_span
                    )
                )
            if frame_value:
                # Binding an entered frame originates at the source as-target;
                # retain the existing native span argument while locating this call.
                ast.copy_location(value, target)
            return [ast.copy_location(ast.Assign([target], value), statement)]
        if isinstance(target, ast.Attribute):
            # -------------------- Pattern --------------------
            # Python source:
            #     a.field = value
            #
            # Builder:
            #     X.setattr_(a, "field", value)
            # -------------------------------------------------
            value = self._call_dialect(
                "setattr_", [self.visit(target.value), ast.Constant(target.attr), value], statement
            )
            return [ast.copy_location(ast.Expr(value), statement)]
        if isinstance(target, ast.Subscript):
            # -------------------- Pattern --------------------
            # Python source:
            #     a[index] = value
            #
            # Builder:
            #     X.setitem_(a, index, value)
            # -------------------------------------------------
            value = self._call_dialect(
                "setitem_",
                [self.visit(target.value), self._rewrite_index(target.slice), value],
                statement,
            )
            return [ast.copy_location(ast.Expr(value), statement)]
        if isinstance(target, ast.Tuple | ast.List):
            # -------------------- Pattern --------------------
            # Python source:
            #     a, (b, c) = rhs
            #
            # Builder:
            #     first, second = X.unpack(rhs)
            #     a = X.bind_(first, name="a")
            #     left, right = X.unpack(second)
            #     b = X.bind_(left, name="b")
            #     c = X.bind_(right, name="c")
            # -------------------------------------------------
            names = [self.module.make_fresh_name("_unpack") for _ in target.elts]
            pattern: list[ast.expr] = [
                ast.Starred(ast.Name(name, ast.Store()), ast.Store())
                if isinstance(item, ast.Starred)
                else ast.Name(name, ast.Store())
                for name, item in zip(names, target.elts)
            ]
            result: list[ast.stmt] = [
                ast.copy_location(
                    ast.Assign(
                        [ast.Tuple(pattern, ast.Store())],
                        self._call(self.function.dialect_prefix, "unpack", [value], target),
                    ),
                    target,
                )
            ]
            for item, name in zip(target.elts, names):
                result.extend(
                    self._bind(
                        item.value if isinstance(item, ast.Starred) else item,
                        ast.Name(name, ast.Load()),
                        statement,
                        frame_value=frame_value,
                        value_span=value_span,
                    )
                )
            return result
        self._raise_error(target, f"Unsupported assignment target: {type(target).__name__}")

    def visit_Assign(self, node: ast.Assign) -> ast.stmt | list[ast.stmt]:
        target: ast.expr
        # -------------------- Pattern --------------------
        # Python source:
        #     a = b = rhs
        #
        # Builder:
        #     temporary = rhs
        #     a = X.bind_(temporary, name="a")
        #     b = X.bind_(temporary, name="b")
        # -------------------------------------------------
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        ordinary = any(self._uses_ordinary_binding(target) for target in node.targets)
        value_span = self.module.make_span_expr(node.value) if ordinary else None
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0]
            value = self._rewrite_assignment_value(node.value, ordinary=ordinary)
            return self._bind(target, value, node, value_span=value_span)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
            # -------------------- Pattern --------------------
            # Python source:
            #     target()[index()] = value()
            #
            # Builder:
            #     X.setitem_(value=value(), target=target(), key=index())
            # -------------------------------------------------
            # Keyword order preserves RHS-before-target/index evaluation without
            # a temporary. Chained and augmented assignments keep their own order.
            target = node.targets[0]
            return ast.copy_location(
                ast.Expr(
                    self._call(
                        self.function.dialect_prefix,
                        "setitem_",
                        [],
                        node,
                        keywords={
                            "value": self._rewrite_assignment_value(node.value, ordinary=ordinary),
                            "target": self.visit(target.value),
                            "key": self._rewrite_index(target.slice),
                        },
                        span=self.module.make_span_expr(node),
                    )
                ),
                node,
            )
        temporary = self.module.make_fresh_name("_value")
        result: list[ast.stmt] = [
            self._assign(
                temporary, self._rewrite_assignment_value(node.value, ordinary=ordinary), node
            )
        ]
        for target in node.targets:
            result.extend(
                self._bind(
                    target,
                    ast.Name(temporary, ast.Load()),
                    node,
                    value_span=value_span,
                )
            )
        return result

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign | list[ast.stmt]:
        # -------------------- Pattern --------------------
        # Python source:
        #     x: X.int32 = value
        #
        # Builder:
        #     x = X.decl_mutable_cell_(value, ty=X.int32, name="x")
        # -------------------------------------------------
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        if not isinstance(node.target, ast.Name):
            self._raise_error(node.target, "An annotated binding requires a name")
        ordinary = self._uses_ordinary_binding(node.target)
        value_span = self.module.make_span_expr(node.value) if ordinary and node.value else None
        value = (
            self._rewrite_assignment_value(node.value, ordinary=ordinary)
            if node.value
            else ast.Attribute(ast.Name(self.module.ir_prefix, ast.Load()), "MISSING", ast.Load())
        )
        annotation = parse_annotation(node.annotation, self.module.filename)
        names = set(collect_annotation_free_names(annotation))
        with self._rewrite_annotation(annotation):
            helper = self._create_lambda([], self.visit(annotation))
        self._capture_defaults(helper, names, node.annotation, body_locals=True)
        annotation = ast.copy_location(ast.Call(helper, [], []), node.annotation)
        return self._bind(node.target, value, node, ty=annotation, value_span=value_span)

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AugAssign | list[ast.stmt]:
        key: ast.expr
        load: ast.expr
        value: ast.expr
        # -------------------- Pattern --------------------
        # Python source:
        #     a[index] += value
        #
        # Builder:
        #     base = a
        #     key = index
        #     old = base[key]
        #     X.setitem_(base, key, old + value)
        # -------------------------------------------------
        # Base, index, load and RHS each evaluate once; name targets use declaration dispatch.
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            value = ast.copy_location(
                ast.BinOp(ast.Name(node.target.id, ast.Load()), node.op, self.visit(node.value)),
                node,
            )
            if self._uses_ordinary_binding(node.target):
                return self._bind(
                    node.target, value, node, value_span=self.module.make_span_expr(node)
                )
            return self._bind(node.target, self._attach_span(value, node), node)
        if not isinstance(node.target, ast.Subscript | ast.Attribute):
            self._raise_error(
                node.target, "An augmented assignment requires a name, attribute, or index"
            )
        base = self.module.make_fresh_name("_base")
        statements: list[ast.stmt] = [self._assign(base, self.visit(node.target.value), node)]
        if isinstance(node.target, ast.Attribute):
            key = ast.Constant(node.target.attr)
            load = ast.Attribute(ast.Name(base, ast.Load()), node.target.attr, ast.Load())
            operation = "setattr_"
        else:
            index = self.module.make_fresh_name("_index")
            statements.append(self._assign(index, self._rewrite_index(node.target.slice), node))
            key = ast.Name(index, ast.Load())
            load = ast.Subscript(ast.Name(base, ast.Load()), key, ast.Load())
            operation = "setitem_"
        old = self.module.make_fresh_name("_old")
        statements.append(self._assign(old, self._attach_span(load, node.target), node))
        value = self._attach_span(
            ast.BinOp(ast.Name(old, ast.Load()), node.op, self.visit(node.value)), node
        )
        statements.append(
            ast.copy_location(
                ast.Expr(
                    self._call_dialect(operation, [ast.Name(base, ast.Load()), key, value], node)
                ),
                node,
            )
        )
        return statements

    def visit_Expr(self, node: ast.Expr) -> ast.Expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     f()
        #
        # Builder:
        #     X.emit_(_S[i].ctx(lambda: f()), span=_S[i])
        # -------------------------------------------------
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        previous, self.emitted_expression = self.emitted_expression, node.value
        try:
            value = self.visit(node.value)
        finally:
            self.emitted_expression = previous
        return ast.copy_location(ast.Expr(self._call_dialect("emit_", [value], node)), node)

    def visit_Return(self, node: ast.Return) -> ast.Return | ast.Expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     return x
        #
        # Builder:
        #     X.return_(x)
        # -------------------------------------------------
        # Macro and bypass modes retain ordinary Python return.
        value = self.visit(node.value) if node.value else None
        if self.preserve_return or self.bypass_ast_rewrite:
            return ast.copy_location(ast.Return(value), node)
        return ast.copy_location(
            ast.Expr(self._call_dialect("return_", [] if value is None else [value], node)), node
        )

    def visit_Break(self, node: ast.Break) -> ast.Break | ast.Expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     break
        #
        # Builder:
        #     X.break_()
        # -------------------------------------------------
        return (
            node
            if self.bypass_ast_rewrite
            else ast.copy_location(ast.Expr(self._call_dialect("break_", [], node)), node)
        )

    def visit_Continue(self, node: ast.Continue) -> ast.Continue | ast.Expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     continue
        #
        # Builder:
        #     X.continue_()
        # -------------------------------------------------
        return (
            node
            if self.bypass_ast_rewrite
            else ast.copy_location(ast.Expr(self._call_dialect("continue_", [], node)), node)
        )

    def visit_Assert(self, node: ast.Assert) -> ast.Assert | ast.Expr:
        # -------------------- Pattern --------------------
        # Python source:
        #     assert condition, message
        #
        # Builder:
        #     X.assert_(condition, message)
        # -------------------------------------------------
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        message = self.visit(node.msg) if node.msg else ast.Constant("")
        return ast.copy_location(
            ast.Expr(self._call_dialect("assert_", [self.visit(node.test), message], node)), node
        )

    def _rewrite_branch(self, body: list[ast.stmt], node: ast.AST, prefix: str) -> list[ast.stmt]:
        # Branch helper scoping keeps source names; mutable stores bind no locals.
        name = self.module.make_fresh_name(prefix)
        return [
            self._create_definition(name, self.transform_statements(body), node),
            ast.copy_location(ast.Expr(ast.Call(ast.Name(name, ast.Load()), [], [])), node),
        ]

    def visit_If(self, node: ast.If) -> ast.If | list[ast.stmt]:
        # -------------------- Pattern --------------------
        # Python source:
        #     if I.constexpr(flag):
        #         body()
        #
        # Builder:
        #     if flag:
        #         X.emit_(body())
        # -------------------------------------------------
        marker = self._read_constexpr_operand(node.test)
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        if marker is not None:
            with self._bypass_rewrite():
                condition = self.visit(marker)
            return ast.copy_location(
                ast.If(
                    condition,
                    self.transform_statements(node.body) or [ast.Pass()],
                    self.transform_statements(node.orelse),
                ),
                node,
            )
        # -------------------- Pattern --------------------
        # Python source:
        #     if condition:
        #         y = a
        #     else:
        #         y = b
        #
        # Builder:
        #     with X.if_(condition) as frame:
        #         with X.then_():
        #             def then():
        #                 y = X.bind_(a, name="y")
        #             then()
        #         with X.else_():
        #             def otherwise():
        #                 y = X.bind_(b, name="y")
        #             otherwise()
        #     y = frame.var
        # -------------------------------------------------
        frame = self.module.make_fresh_name("_conditional")
        then = (
            self.transform_statements(node.body)
            if self.preserve_return
            else self._rewrite_branch(node.body, node, "_then")
        )
        branches: list[ast.stmt] = [
            ast.copy_location(
                ast.With(
                    [ast.withitem(self._call_dialect("then_", [], node))], then or [ast.Pass()]
                ),
                node,
            )
        ]
        if node.orelse:
            otherwise = (
                self.transform_statements(node.orelse)
                if self.preserve_return
                else self._rewrite_branch(node.orelse, node, "_else")
            )
            branches.append(
                ast.copy_location(
                    ast.With(
                        [ast.withitem(self._call_dialect("else_", [], node))],
                        otherwise or [ast.Pass()],
                    ),
                    node,
                )
            )
        result: list[ast.stmt] = [
            ast.copy_location(
                ast.With(
                    [
                        ast.withitem(
                            self._call_dialect("if_", [self.visit(node.test)], node),
                            ast.Name(frame, ast.Store()),
                        )
                    ],
                    branches,
                ),
                node,
            )
        ]
        output = self.module.prescan_ctx.conditional_outputs.get(node)
        if output is not None:
            result.append(
                self._assign(
                    output, ast.Attribute(ast.Name(frame, ast.Load()), "var", ast.Load()), node
                )
            )
        return result

    def visit_For(self, node: ast.For) -> ast.For | ast.With | list[ast.stmt]:
        # -------------------- Pattern --------------------
        # Python source:
        #     for i, *tail in X.grid(m, n, k):
        #         body(i, tail)
        #
        # Builder:
        #     loop = X.for_(X.grid(m, n, k), names=("i", "*tail"))
        #     with loop:
        #         i, *tail = loop.vars
        #         X.emit_(body(i, tail))
        #
        # A scalar source target uses ordinary entry directly:
        #     with X.for_(X.grid(n), names="i") as i:
        #         X.emit_(body(i))
        # -------------------------------------------------
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        if node.orelse:
            self._raise_error(node, "A construction loop does not support an else clause")
        if (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            # Prescan reserves range, so symbolic bounds need no per-use scope lookup.
            # Arguments keep the original call's ordinary evaluation order.
            iterable = self.visit_Call(
                node.iter,
                callee=ast.Attribute(
                    ast.Name(self.function.dialect_prefix, ast.Load()), "range_", ast.Load()
                ),
            )
        else:
            iterable = self.visit(node.iter)
        if isinstance(node.target, ast.Name):
            names: ast.expr = ast.Constant(node.target.id)
        elif isinstance(node.target, ast.Tuple | ast.List):
            names = ast.Tuple(
                [
                    ast.Constant("*" + item.value.id if isinstance(item, ast.Starred) else item.id)
                    for item in node.target.elts
                ],
                ast.Load(),
            )
        else:
            self._raise_error(node.target, "Loop targets must be names or a flat tuple of names")
        context = self._call_dialect("for_", [iterable], node, keywords={"names": names})
        # The generated iteration check originates at the source iterable, not
        # the final body line. Its native frame span still covers the whole loop.
        ast.copy_location(context, node.iter)
        # A sequence target unpacks stable frame.vars after entry, including a
        # one-dimensional loop whose public entry returns a scalar variable.
        frame = (
            self.module.make_fresh_name("_loop") if not isinstance(node.target, ast.Name) else None
        )
        body = self.transform_statements(node.body)
        if frame is not None:
            unpack = ast.copy_location(
                ast.Assign(
                    [node.target],
                    ast.Attribute(ast.Name(frame, ast.Load()), "vars", ast.Load()),
                ),
                node.target,
            )
            return [
                self._assign(frame, context, node.iter),
                ast.copy_location(
                    ast.With(
                        [ast.withitem(ast.Name(frame, ast.Load()))],
                        [unpack, *body],
                    ),
                    node,
                ),
            ]
        return ast.copy_location(
            ast.With([ast.withitem(context, node.target)], body or [ast.Pass()]), node
        )

    def visit_While(self, node: ast.While) -> ast.While | ast.With:
        # -------------------- Pattern --------------------
        # Python source:
        #     while condition:
        #         body()
        #
        # Builder:
        #     with X.while_(condition):
        #         X.emit_(body())
        # -------------------------------------------------
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        if node.orelse:
            self._raise_error(node, "A construction loop does not support an else clause")
        context = self._call_dialect("while_", [self.visit(node.test)], node)
        return ast.copy_location(
            ast.With([ast.withitem(context)], self.transform_statements(node.body) or [ast.Pass()]),
            node,
        )

    def visit_With(self, node: ast.With) -> ast.With | list[ast.stmt]:
        # -------------------- Pattern --------------------
        # Python source:
        #     with X.block() as value:
        #         body(value)
        #
        # Builder:
        #     with X.block() as entered:
        #         value = X.bind_(entered, name="value", frame_value=True)
        #         X.emit_(body(value))
        # -------------------------------------------------
        # Ordinary regions use Python locals; native dataflow outputs retain identity.
        if self.bypass_ast_rewrite:
            return self.generic_visit(node)
        item, rest = node.items[0], node.items[1:]
        body: list[ast.stmt] = (
            [ast.copy_location(ast.With(rest, node.body), node)] if rest else node.body
        )
        if item.optional_vars is None:
            target, initial = None, []
        else:
            entered = self.module.make_fresh_name("_entered")
            target = ast.Name(entered, ast.Store())
            initial = self._bind(
                item.optional_vars, ast.Name(entered, ast.Load()), node, frame_value=True
            )
        outputs = self.module.prescan_ctx.with_outputs.get(node, ())
        context = self.visit(item.context_expr)
        translated_body = initial + self.transform_statements(body) or [ast.Pass()]
        if not outputs:
            return ast.copy_location(
                ast.With([ast.withitem(context, target)], translated_body), node
            )
        # -------------------- Pattern --------------------
        # Python source:
        #     with X.dataflow():
        #         y = expression
        #         X.output(y)
        #
        # Builder:
        #     with X.dataflow() as frame:
        #         y = X.bind_(expression, name="y")
        #         X.output(y)
        #     y = frame.output_vars[0]
        # -------------------------------------------------
        # The native frame converts exports to ordinary output variables.
        frame = self.module.make_fresh_name("_dataflow")
        statements: list[ast.stmt] = [
            self._assign(frame, context, node),
            ast.copy_location(
                ast.With([ast.withitem(ast.Name(frame, ast.Load()), target)], translated_body), node
            ),
        ]
        for index, name in enumerate(outputs):
            value = ast.Subscript(
                ast.Attribute(ast.Name(frame, ast.Load()), "output_vars", ast.Load()),
                ast.Constant(index),
                ast.Load(),
            )
            statements.append(self._assign(name, value, node))
        return statements

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef | list[ast.stmt]:
        # -------------------- Pattern --------------------
        # Python source:
        #     @X.function
        #     def nested():
        #         body()
        #
        # Builder:
        #     with X.function_(decl=True, local=True) as frame:
        #         X.func_name("nested")
        #     nested = frame.local_var
        #     with frame:
        #         def build():
        #             X.emit_(body())
        #         build()
        # -------------------------------------------------
        if self.bypass_ast_rewrite:
            return node
        kind, _ = self.read_function_metadata(node, allow_python=True)
        if kind is None:
            return node
        declaration, _, body = self.create_function_builder_fragments(node, local_function=True)
        return [*declaration, body]

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.Pass:
        # -------------------- Pattern --------------------
        # Python source:
        #     nonlocal value
        #
        # Builder:
        #     pass
        # -------------------------------------------------
        # Source closure declarations do not mutate the host closure during build.
        return ast.copy_location(ast.Pass(), node)

    def visit_Pass(self, node: ast.Pass) -> ast.Pass:
        # -------------------- Pattern --------------------
        # Python source:
        #     pass
        #
        # Builder:
        #     pass
        # -------------------------------------------------
        return node

    def generic_visit(self, node: _Node) -> _Node:
        if isinstance(node, ast.stmt) and not self.bypass_ast_rewrite:
            self._raise_error(node, f"Unsupported statement: {type(node).__name__}")
        return super().generic_visit(node)

    def read_function_metadata(
        self, node: ast.FunctionDef, *, allow_python: bool = False
    ) -> tuple[object | None, ast.Dict]:
        """Read the construction namespace from a qualified source decorator."""
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            key = self.module.prescan_ctx._match_special_func(target)
            if key is None:
                continue
            kind = protocol.DECLARATION_KIND.get(key)
            if kind == "helper":
                return None, ast.Dict([], [])
            if kind != "function":
                continue
            # A matched decorator is a direct member of a fixed root. Construction
            # needs that root's actual builder, not another syntax/value resolution.
            namespace = self.module.environment[target.value.id]
            if (
                self.module.root_function_kwargs is not None
                and self.module.module_name is None
                and self.function.current_scope is None
            ):
                # The public root decorator already evaluated its arguments once.
                # Even an empty mapping overrides the source options expression.
                kwargs = self.module.root_function_kwargs
                return namespace, ast.copy_location(
                    ast.Dict(
                        [ast.Constant(key) for key in kwargs],
                        [self._inject(value) for value in kwargs.values()],
                    ),
                    node,
                )
            keys, values = [], []
            if isinstance(decorator, ast.Call):
                if decorator.args:
                    self._raise_error(decorator, "Function decorators accept keyword options only")
                for keyword in decorator.keywords:
                    if keyword.arg != "check_well_formed":
                        keys.append(ast.Constant(keyword.arg) if keyword.arg is not None else None)
                        values.append(keyword.value)
            return namespace, ast.copy_location(ast.Dict(keys, values), node)
        if allow_python:
            return None, ast.Dict([], [])
        self._raise_error(
            node,
            f"Function {node.name!r} requires a qualified construction decorator "
            "such as @T.prim_func; bare and preconfigured decorator aliases are unsupported",
        )

    def _function_namespace(self, node: ast.FunctionDef, namespace: object) -> str:
        """Keep a fixed source alias when definition and execution scopes agree."""
        references = (
            reference for source in [*node.decorator_list, node] for reference in ast.walk(source)
        )
        for reference in references:
            if (
                isinstance(reference, ast.Name)
                and reference.id in self.module.prescan_ctx.namespaces
                and self.module.environment.get(reference.id) is namespace
                and self.module.exec_globals.get(reference.id) is namespace
            ):
                return reference.id
        # Conflicting definition and execution namespaces need an injected name.
        return self._inject(namespace, "_X").id

    def _read_function_annotations(
        self, node: ast.FunctionDef, parameters: list[ast.arg], facts: list[Binding]
    ) -> tuple[list[ast.expr | None], ast.expr | None, set[str]]:
        """Find definition-scope names needed by signatures and body annotations."""
        declared_names = {item.name for item in getattr(node, "type_params", ())}
        annotations = [
            parse_annotation(parameter.annotation, self.module.filename)
            if parameter.annotation
            else None
            for parameter in parameters
        ]
        returns = parse_annotation(node.returns, self.module.filename) if node.returns else None
        annotation_names = {
            name
            for annotation in [*annotations, returns]
            if annotation is not None
            for name in collect_annotation_free_names(annotation)
        }
        body_annotations = [
            parse_annotation(item.annotation, self.module.filename)
            for item in facts
            if item.annotation is not None
            and item.kind not in ("parameter", "mutable_parameter", "symbol")
        ]
        annotation_names.update(
            name
            for annotation in body_annotations
            for name in collect_annotation_free_names(annotation)
        )
        return annotations, returns, annotation_names - declared_names

    def _create_definition_bindings(
        self,
        node: ast.FunctionDef,
        names: set[str],
        *,
        captures: str,
    ) -> list[ast.stmt]:
        """Snapshot only lexical annotation/constexpr names at their definition site."""
        names = names | {
            parameter.arg
            for parameter in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            if self._is_constexpr_annotation(parameter.annotation)
        }
        # Inject builtin operations so same-named source bindings cannot replace them.
        # Acquire each scope once; the snapshot retains selected values, never frames.
        scopes: list[ast.expr] = [ast.Call(self._inject(locals), [], [])]
        scopes.append(ast.Call(self._inject(globals), [], []))
        defaults = {name: getattr(builtins, name) for name in names if hasattr(builtins, name)}
        if defaults:
            scopes.append(self._inject(defaults))
        captured = ast.Call(
            self._inject(_AnnotationScope),
            [ast.Tuple([ast.Constant(name) for name in sorted(names)], ast.Load()), *scopes],
            [],
        )
        return [self._assign(captures, captured, node)]

    def _capture_defaults(
        self,
        helper: ast.FunctionDef | ast.Lambda,
        names: set[str],
        node: ast.AST,
        *,
        body_locals: bool = False,
    ) -> None:
        """Give a helper original-name captures; Python owns all lexical shadowing."""
        for name in sorted(names):
            value = self._call(
                self.function.definition_captures,
                "get",
                [
                    ast.Constant(name),
                    ast.Attribute(
                        ast.Name(self.module.ir_prefix, ast.Load()), "MISSING", ast.Load()
                    ),
                ],
                node,
            )
            if body_locals and any(
                item.name == name
                and (item.node.lineno, item.node.col_offset) < (node.lineno, node.col_offset)
                for item in self.module.prescan_ctx.bindings.get(self.function.current_scope, ())
            ):
                # Keep a real name read so Python captures enclosing helper cells.
                # A prior conditional binding may not have executed: snapshot its
                # absence without reading it until the annotation selects that name.
                value = ast.IfExp(
                    ast.Compare(
                        ast.Constant(name), [ast.In()], [ast.Call(self._inject(locals), [], [])]
                    ),
                    ast.Name(name, ast.Load()),
                    ast.Attribute(
                        ast.Name(self.module.ir_prefix, ast.Load()), "MISSING", ast.Load()
                    ),
                )
            helper.args.kwonlyargs.append(ast.arg(name))
            helper.args.kw_defaults.append(value)

    def _create_const_args(self, node: ast.FunctionDef, *, const_args: str) -> list[ast.stmt]:
        """Read root JIT inputs; nested functions keep ordinary runtime parameters."""
        from . import jit_support

        # _const_args = read_specialization_bindings("f")
        const_args_expr = ast.Call(
            self._inject(jit_support.read_specialization_bindings),
            [ast.Constant(node.name)],
            [],
        )
        return [self._assign(const_args, const_args_expr, node)]

    def _create_symbol_declarations(self, node: ast.FunctionDef) -> list[ast.stmt]:
        """Bind explicit signature type parameters with their declared dtypes."""
        declaration: list[ast.stmt] = []
        # -------------------- Pattern --------------------
        # Python source:
        #     def f[n]():
        #         body(n)
        #
        # Builder:
        #     n = X.resolve_type_var_("n")
        # -------------------------------------------------
        for parameter in getattr(node, "type_params", ()):
            if not isinstance(parameter, getattr(ast, "TypeVar", ())):
                self._raise_error(parameter, "Only scalar type parameters are supported")
            bound = getattr(parameter, "bound", None)
            dtype = self.module.prescan_ctx.sites[parameter].dtype
            if bound is not None and not (isinstance(bound, ast.Name) and bound.id == "int"):
                if (
                    protocol.SCALAR_ANNOTATION_DTYPE.get(
                        self.module.prescan_ctx._match_special_func(bound)
                    )
                    is None
                ):
                    self._raise_error(
                        parameter,
                        "A symbolic type parameter bound must be int or a registered scalar dtype",
                    )
            if getattr(parameter, "default_value", None) is not None:
                self._raise_error(parameter, "A symbolic type parameter cannot have a default")
            declaration.append(
                self._assign(
                    parameter.name,
                    self._call_dialect(
                        "resolve_type_var_",
                        [ast.Constant(parameter.name)],
                        parameter,
                        keywords={"dtype": ast.Constant(dtype)},
                    ),
                    parameter,
                )
            )
        return declaration

    @staticmethod
    def _select_specialized_value(
        name: str, fallback: ast.expr, node: ast.AST, *, const_args: str
    ) -> ast.IfExp:
        """Select a JIT value or explicit absence without evaluating the fallback."""
        selected = ast.BoolOp(
            ast.And(),
            [
                ast.Compare(ast.Name(const_args, ast.Load()), [ast.IsNot()], [ast.Constant(None)]),
                ast.Compare(ast.Constant(name), [ast.In()], [ast.Name(const_args, ast.Load())]),
            ],
        )
        # _const_args["x"] if _const_args is not None and "x" in it else fallback
        return ast.copy_location(
            ast.IfExp(
                selected,
                ast.Subscript(ast.Name(const_args, ast.Load()), ast.Constant(name), ast.Load()),
                fallback,
            ),
            node,
        )

    def _is_constexpr_annotation(self, annotation: ast.expr | None) -> bool:
        """Recognize constexpr syntax through a canonical registered root/member key."""
        key = self.module.prescan_ctx._match_special_func(annotation)
        return key is not None and key.endswith(".constexpr")

    def _rewrite_parameters(
        self,
        parameters: list[ast.arg],
        annotations: list[ast.expr | None],
        *,
        captures: str,
        const_args: str | None,
    ) -> tuple[list[ast.stmt], dict[str, str]]:
        """Declare signature parameters, making constexpr values available first."""
        declaration: list[ast.stmt] = []
        constexpr_aliases: dict[str, str] = {}
        constexpr_params: list[tuple[ast.arg, ast.expr | None, bool]] = []
        other_params: list[tuple[ast.arg, ast.expr | None, bool]] = []
        # Runtime annotations may depend on a later constexpr parameter. Preserve
        # source order within each group and reuse this one syntactic classification.
        for parameter, annotation in zip(parameters, annotations):
            is_constexpr = self._is_constexpr_annotation(annotation)
            group = constexpr_params if is_constexpr else other_params
            group.append((parameter, annotation, is_constexpr))
        for parameter, annotation, is_constexpr in [*constexpr_params, *other_params]:
            if annotation is None:
                self._raise_error(parameter, f"Parameter {parameter.arg!r} requires an annotation")
            name = parameter.arg
            alias = name
            if is_constexpr:
                # The body binds this value in its own scope; retain an outer
                # alias so that binding does not become a local `name = name`.
                alias = self.module.make_fresh_name("_parameter")
                constexpr_aliases[name] = alias
                fallback = ast.Call(
                    self._inject(_require_constexpr_arg),
                    [
                        self._call(
                            captures,
                            "get",
                            [
                                ast.Constant(name),
                                ast.Attribute(
                                    ast.Name(self.module.ir_prefix, ast.Load()),
                                    "MISSING",
                                    ast.Load(),
                                ),
                            ],
                            parameter,
                        ),
                        ast.Constant(name),
                    ],
                    [],
                )
            else:
                with self._rewrite_annotation(annotation):
                    translated = self.visit(annotation)
                if const_args is not None:
                    # Optional annotations are unwrapped only for specialization;
                    # ordinary annotation validation belongs to the language variant arg.
                    translated = ast.Call(
                        self._inject(_unwrap_optional_annotation),
                        [translated, ast.Name(const_args, ast.Load())],
                        [],
                    )
                # x = X.arg("x", annotation, span=_S[i])
                fallback = self._call_dialect("arg", [ast.Constant(name), translated], parameter)
            # Specialized parameters have no runtime ABI slot. The annotation
            # thunk is absent from the selected generated Python branch.
            value = (
                fallback
                if const_args is None
                else self._select_specialized_value(
                    name, fallback, parameter, const_args=const_args
                )
            )
            declaration.append(self._assign(alias, value, parameter))
            if alias != name:
                declaration.append(self._assign(name, ast.Name(alias, ast.Load()), parameter))
        return declaration, constexpr_aliases

    def _create_function_frame(
        self,
        node: ast.FunctionDef,
        options: ast.Dict,
        declaration: list[ast.stmt],
        *,
        frame: str,
        local_function: bool,
        split_declare: bool,
    ) -> tuple[list[ast.stmt], ast.withitem]:
        """Create a declaration pass, if needed, and the body frame entry."""
        # -------------------- Pattern --------------------
        # Python source:
        #     @X.function
        #     def f(x: X.int32):
        #         body(x)
        #
        # Builder:
        #     with X.function_(decl=True) as frame:
        #         X.func_name("f")
        #         parameter = X.arg("x", X.int32)
        #     f = frame.global_var
        # -------------------------------------------------
        keywords = [ast.keyword(None, options)] if options.keys else []
        if split_declare:
            keywords.append(ast.keyword("decl", ast.Constant(True)))
        if local_function:
            keywords.append(ast.keyword("local", ast.Constant(True)))
        if self.module.track_span:
            keywords.append(ast.keyword("span", self.module.make_span_expr(node)))
        constructor = ast.copy_location(
            ast.Call(
                ast.Attribute(
                    ast.Name(self.function.dialect_prefix, ast.Load()), "function_", ast.Load()
                ),
                [],
                keywords,
            ),
            node,
        )
        if split_declare:
            declaration_scope = ast.With(
                [ast.withitem(constructor, ast.Name(frame, ast.Store()))], declaration
            )
            reference = ast.Attribute(
                ast.Name(frame, ast.Load()),
                "local_var" if local_function else "global_var",
                ast.Load(),
            )
            return (
                [
                    ast.copy_location(declaration_scope, node),
                    self._assign(node.name, reference, node),
                ],
                ast.withitem(ast.Name(frame, ast.Load())),
            )
        # Ordinary standalone functions enter once for their signature and body.
        return [], ast.withitem(constructor, ast.Name(frame, ast.Store()))

    def _create_body_parameters(
        self,
        node: ast.FunctionDef,
        parameters: list[ast.arg],
        constexpr_aliases: dict[str, str],
        *,
        frame: str,
        const_args: str | None,
    ) -> list[ast.stmt]:
        """Bind known ordinary parameters directly; selected JIT parameters have no ABI slot."""
        body: list[ast.stmt] = []
        if const_args is None:
            # -------------------- Pattern --------------------
            # Python source:
            #     def f(x: X.int32, y: X.int32):
            #         body(x, y)
            #
            # Builder:
            #     x, y = frame.params
            # -------------------------------------------------
            runtime = [p for p in parameters if p.arg not in constexpr_aliases]
            if runtime:
                body.append(
                    ast.copy_location(
                        ast.Assign(
                            [
                                ast.Tuple(
                                    [ast.Name(p.arg, ast.Store()) for p in runtime], ast.Store()
                                )
                            ],
                            ast.Attribute(ast.Name(frame, ast.Load()), "params", ast.Load()),
                        ),
                        node,
                    )
                )
            for name, alias in constexpr_aliases.items():
                body.append(self._assign(name, ast.Name(alias, ast.Load()), node))
        else:
            iterator = self.module.make_fresh_name("_arguments")
            body.append(
                self._assign(
                    iterator,
                    ast.Call(
                        self._inject(iter),
                        [ast.Attribute(ast.Name(frame, ast.Load()), "params", ast.Load())],
                        [],
                    ),
                    node,
                )
            )
            for parameter in parameters:
                name = parameter.arg
                value = (
                    ast.Name(constexpr_aliases[name], ast.Load())
                    if name in constexpr_aliases
                    else self._select_specialized_value(
                        name,
                        ast.Call(self._inject(next), [ast.Name(iterator, ast.Load())], []),
                        parameter,
                        const_args=const_args,
                    )
                )
                body.append(self._assign(name, value, parameter))
        for parameter in getattr(node, "type_params", ()):
            body.append(
                self._assign(
                    parameter.name,
                    self._call_dialect(
                        "resolve_type_var_", [ast.Constant(parameter.name)], parameter
                    ),
                    parameter,
                )
            )
        return body

    def _create_split_declaration(
        self,
        node: ast.FunctionDef,
        frame: str,
        definition: ast.FunctionDef,
        frame_declaration: list[ast.stmt],
        capture_names: set[str],
        constexpr_aliases: dict[str, str],
    ) -> list[ast.stmt]:
        """Keep signature snapshots separate from the body's enclosing Python scope."""
        # The declaration helper returns only the frame and selected host values.
        # Runtime parameters and explicit symbols are recovered from the frame by
        # the sibling body helper, whose closures still observe source class setup.
        declare_name = self.module.make_fresh_name("_declare")
        signature = frame_declaration[0]
        # Decorator options evaluate outside the signature's same-named parameters.
        constructor = signature.items[0].context_expr
        signature.items[0].context_expr = ast.copy_location(ast.Name(frame, ast.Load()), node)
        outputs = [frame, *constexpr_aliases.values()]
        returned = ast.copy_location(
            ast.Return(ast.Tuple([ast.Name(name, ast.Load()) for name in outputs], ast.Load())),
            node,
        )
        declare = self._create_definition(declare_name, [signature, returned], node)
        declare.args.args.append(ast.arg(frame))
        self._capture_defaults(declare, capture_names, node)
        return [
            declare,
            ast.copy_location(
                ast.Assign(
                    [ast.Tuple([ast.Name(name, ast.Store()) for name in outputs], ast.Store())],
                    ast.Call(ast.Name(declare_name, ast.Load()), [constructor], []),
                ),
                node,
            ),
            definition,
            frame_declaration[1],
        ]

    def create_function_builder_fragments(
        self, node: ast.FunctionDef, *, local_function: bool = False, split_declare: bool = True
    ) -> tuple[list[ast.stmt], str, ast.With]:
        """Declare a native frame and emit a lexical body helper inside its scope.

        Original-name helper defaults retain definition values. Subsequent Python
        assignments expose each signature parameter to later annotations. Body
        helpers preserve source globals/closures, independently of annotation captures.
        """
        kind, options = self.read_function_metadata(node)
        builder = self._function_namespace(node, kind)
        frame, body_name = self.module.make_fresh_name("_fn"), self.module.make_fresh_name("_build")
        # Keep the shared module context, but activate fresh lexical state for this
        # function. Saving its caller locally avoids a context ownership back-reference.
        old = self.function
        self.function = FunctionContext(node, builder)
        try:
            parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            if node.args.vararg or node.args.kwarg:
                self._raise_error(node, "IR signatures require ordinary named parameters")
            facts = self.module.prescan_ctx.bindings.get(node, [])
            annotations, returns, capture_names = self._read_function_annotations(
                node, parameters, facts
            )
            captures = self.module.make_fresh_name("_definition")
            self.function.definition_captures = captures
            statements = self._create_definition_bindings(node, capture_names, captures=captures)
            const_args = None
            if self.module.enable_jit_map and not local_function:
                const_args = self.module.make_fresh_name("_const_args")
                statements.extend(self._create_const_args(node, const_args=const_args))
            declaration: list[ast.stmt] = [
                ast.copy_location(
                    ast.Expr(self._call(builder, "func_name", [ast.Constant(node.name)], node)),
                    node,
                )
            ]
            declaration.extend(self._create_symbol_declarations(node))
            arguments, constexpr_aliases = self._rewrite_parameters(
                parameters, annotations, captures=captures, const_args=const_args
            )
            declaration.extend(arguments)
            if returns is not None:
                with self._rewrite_annotation(returns):
                    translated_return = self.visit(returns)
                declaration.append(
                    ast.copy_location(
                        ast.Expr(
                            self._call(
                                builder,
                                "func_ret_type",
                                [self._create_lambda([], translated_return)],
                                returns,
                            )
                        ),
                        returns,
                    )
                )
            with self._bypass_rewrite():
                options = self.visit(options)
            frame_declaration, body_entry = self._create_function_frame(
                node,
                options,
                declaration,
                frame=frame,
                local_function=local_function,
                split_declare=split_declare,
            )
            body = self._create_body_parameters(
                node, parameters, constexpr_aliases, frame=frame, const_args=const_args
            )
            body.extend(self.transform_statements(node.body))
            definition = self._create_definition(body_name, body, node)
            if not local_function:
                protected = (
                    {item.name for item in getattr(node, "type_params", ())}
                    | {node.name}
                    | set(self.module.global_func_names)
                    | self.module.prescan_ctx.namespaces.keys()
                )
                if self.module.module_name:
                    protected.add(self.module.module_name)
                self.module.generated_builders.append(
                    GeneratedBuilder(definition, node.name, protected)
                )
            # -------------------- Pattern --------------------
            # Python source:
            #     def f():
            #         body()
            #
            # Builder:
            #     with frame:
            #         def build():
            #             X.emit_(body())
            #         build()
            # -------------------------------------------------
            # The helper owns lexical scope; the surrounding with owns frame
            # entry/exit and error unwinding.
            invocation = ast.copy_location(
                ast.Expr(ast.Call(ast.Name(body_name, ast.Load()), [], [])), node
            )
            if split_declare:
                statements.extend(
                    self._create_split_declaration(
                        node, frame, definition, frame_declaration, capture_names, constexpr_aliases
                    )
                )
            else:
                outputs = list(constexpr_aliases.values())
                if outputs:
                    declaration.append(
                        ast.copy_location(
                            ast.Return(
                                ast.Tuple(
                                    [ast.Name(name, ast.Load()) for name in outputs], ast.Load()
                                )
                            ),
                            node,
                        )
                    )
                signature = self._create_definition(
                    self.module.make_fresh_name("_declare"), declaration, node
                )
                self._capture_defaults(signature, capture_names, node)
                call = ast.Call(ast.Name(signature.name, ast.Load()), [], [])
                signature_call = ast.copy_location(
                    ast.Assign(
                        [ast.Tuple([ast.Name(name, ast.Store()) for name in outputs], ast.Store())],
                        call,
                    )
                    if outputs
                    else ast.Expr(call),
                    node,
                )
            resumed = ast.copy_location(
                ast.With(
                    [body_entry],
                    [invocation]
                    if split_declare
                    else [signature, signature_call, definition, invocation],
                ),
                node,
            )
            return statements, frame, resumed
        finally:
            # Restore the enclosing function before returning fragments or propagating
            # an error; partially rewritten nested functions cannot leave active state.
            self.function = old

    def _create_result_metadata(
        self,
        root: ast.ClassDef | ast.FunctionDef,
        result: str,
        python_functions: list[tuple[str, str]],
        *,
        check_well_formed: bool,
    ) -> list[ast.stmt]:
        """Attach source identity and check the completed construction result."""
        is_module = isinstance(root, ast.ClassDef)
        statements: list[ast.stmt] = [
            ast.copy_location(
                ast.Assign(
                    [ast.Attribute(ast.Name(result, ast.Load()), "__name__", ast.Store())],
                    ast.Constant(root.name),
                ),
                root,
            )
        ]
        if is_module:
            statements.append(
                ast.copy_location(
                    ast.Assign(
                        [ast.Attribute(ast.Name(result, ast.Load()), "__pyfuncs__", ast.Store())],
                        ast.Dict(
                            [ast.Constant(name) for name, _ in python_functions],
                            [ast.Name(alias, ast.Load()) for _, alias in python_functions],
                        ),
                    ),
                    root,
                )
            )
        if check_well_formed:
            # -------------------- Pattern --------------------
            # Python source:
            #     @X.function
            #     def f():
            #         body()
            #
            # Builder:
            #     result = builder.get()
            #     X.check_well_formed_(result)
            # -------------------------------------------------
            # Module syntax selects I.check_well_formed_ instead, after all bodies complete.
            namespace = (
                self.module.ir_prefix
                if is_module
                else self._function_namespace(root, self.read_function_metadata(root)[0])
            )
            statements.append(
                ast.copy_location(
                    ast.Expr(
                        self._call(
                            namespace, "check_well_formed_", [ast.Name(result, ast.Load())], root
                        )
                    ),
                    root,
                )
            )
        return statements

    def rewrite_module(
        self, tree: ast.Module, *, check_well_formed: bool = True
    ) -> tuple[ast.Module, str]:
        """Emit direct native module construction, declarations, then bodies."""
        root, prefix = tree.body[-1], tree.body[:-1]
        if not isinstance(root, ast.ClassDef | ast.FunctionDef):
            self._raise_error(root, "Source must contain one function or module class")
        is_module = isinstance(root, ast.ClassDef)
        members = root.body if is_module else [root]
        functions = [item for item in members if isinstance(item, ast.FunctionDef)]
        if len({item.name for item in functions}) != len(functions):
            self._raise_error(root, "Duplicate function declaration")
        # One module context serves either root shape; only root syntax facts persist
        # here. Frame names and output lists below remain local assembly values.
        self.module.module_name = root.name if is_module else None
        self.module.global_func_names = frozenset(
            item.name for item in functions if self.read_function_metadata(item)[0] is not None
        )
        split_declare = is_module or root in self.module.prescan_ctx.recursive_functions
        builder, result = (
            self.module.make_fresh_name("_builder"),
            self.module.make_fresh_name("_result"),
        )
        body: list[ast.stmt] = []
        definitions, frames, python_functions = [], [], []
        for member in members:
            if not isinstance(member, ast.FunctionDef):
                # Source class statements are host setup inside the native module;
                # global-info registration therefore precedes dependent annotations.
                with self._bypass_rewrite():
                    body.append(self.visit(member))
                targets = (
                    member.targets
                    if isinstance(member, ast.Assign)
                    else [member.target]
                    if isinstance(member, ast.AnnAssign)
                    else []
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        value = self._call(
                            self.module.ir_prefix,
                            "module_member_",
                            [ast.Constant(target.id), ast.Name(target.id, ast.Load())],
                            target,
                        )
                        body.append(self._assign(target.id, value, target))
                continue
            function = member
            kind, _ = self.read_function_metadata(function)
            if kind is None:
                # -------------------- Pattern --------------------
                # Python source:
                #     @I.pyfunc
                #     def f(value):
                #         return value
                #
                # Builder:
                #     f = original_callable
                #     result.__pyfuncs__["f"] = f
                # -------------------------------------------------
                original = self.module.original_func_map.get(function.name)
                if original is not None:
                    body.append(self._assign(function.name, self._inject(original), function))
                else:
                    function.decorator_list = []
                    body.append(function)
                python_functions.append((function.name, function.name))
                continue
            declaration, frame, definition = self.create_function_builder_fragments(
                function, split_declare=split_declare
            )
            body.extend(declaration)
            definitions.append(definition)
            frames.append(frame)
        body.extend(definitions)
        # -------------------- Pattern --------------------
        # Python source:
        #     class Module:
        #         @X.function
        #         def f():
        #             first()
        #         @X.function
        #         def g():
        #             second()
        #
        # Builder:
        #     with IRBuilder() as builder:
        #         with I.ir_module():
        #             with X.function_(decl=True) as f_frame:
        #                 X.func_name("f")
        #             with X.function_(decl=True) as g_frame:
        #                 X.func_name("g")
        #             with f_frame:
        #                 def build_f():
        #                     X.emit_(first())
        #                 build_f()
        #             with g_frame:
        #                 def build_g():
        #                     X.emit_(second())
        #                 build_g()
        #     result = builder.get()
        # -------------------------------------------------
        module = ast.copy_location(
            ast.With(
                [
                    ast.withitem(
                        self._call(self.module.ir_prefix, "ir_module", [], root),
                        ast.Name(root.name, ast.Store()) if is_module else None,
                    )
                ],
                body,
            ),
            root,
        )
        construction = ast.copy_location(
            ast.With(
                [
                    ast.withitem(
                        self._call(self.module.ir_prefix, "IRBuilder", [], root),
                        ast.Name(builder, ast.Store()),
                    )
                ],
                [module] if split_declare else body,
            ),
            root,
        )
        output = (
            self._call(builder, "get", [], root)
            if is_module
            else ast.Attribute(ast.Name(frames[0], ast.Load()), "function", ast.Load())
        )
        translated = [item for item in prefix if not isinstance(item, ast.Import | ast.ImportFrom)]
        translated.extend(
            [
                construction,
                self._assign(result, output, root),
            ]
        )
        translated.extend(
            self._create_result_metadata(
                root, result, python_functions, check_well_formed=check_well_formed
            )
        )
        return ast.fix_missing_locations(ast.Module(translated, [])), result
