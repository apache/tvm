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
"""One syntax prescan for generated names, declarations and region outputs."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import NamedTuple, NoReturn

from . import protocol_registry as protocol
from .expr_str_handling import parse_annotation


def collect_annotation_free_names(
    node: ast.expr, bound: set[str] | None = None
) -> dict[str, ast.Name]:
    """Find annotation reads, respecting Python lambda/comprehension binders.

    The returned nodes retain the introduction locations for diagnostics and
    supply the names needed by temporary/deferred definition capture.
    """
    bound = set() if bound is None else bound
    if isinstance(node, ast.Name):
        return {node.id: node} if isinstance(node.ctx, ast.Load) and node.id not in bound else {}
    if isinstance(node, ast.Lambda):
        # -------------------- Pattern --------------------
        # Python source:
        #     lambda n=outer: shape(n)
        #
        # Builder:
        #     lambda n=captured_outer: shape(n)
        # -------------------------------------------------
        # Only outer is free; n belongs to the lambda.
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if node.args.vararg:
            arguments.append(node.args.vararg)
        if node.args.kwarg:
            arguments.append(node.args.kwarg)
        result: dict[str, ast.Name] = {}
        for value in [*node.args.defaults, *node.args.kw_defaults]:
            if value is not None:
                result.update(collect_annotation_free_names(value, bound))
        result.update(
            collect_annotation_free_names(node.body, bound | {arg.arg for arg in arguments})
        )
        return result
    if isinstance(node, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
        # -------------------- Pattern --------------------
        # Python source:
        #     tuple(n for n in shape)
        #
        # Builder:
        #     tuple(n for n in captured_shape)
        # -------------------------------------------------
        # Only shape is free; n belongs to the comprehension.
        result = {}
        local = set(bound)
        for generator in node.generators:
            result.update(collect_annotation_free_names(generator.iter, local))
            local.update(
                item.id for item in ast.walk(generator.target) if isinstance(item, ast.Name)
            )
            for condition in generator.ifs:
                result.update(collect_annotation_free_names(condition, local))
        values = [node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]
        for value in values:
            result.update(collect_annotation_free_names(value, local))
        return result
    result = {}
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.expr):
            result.update(collect_annotation_free_names(child, bound))
        elif isinstance(child, ast.keyword):
            result.update(collect_annotation_free_names(child.value, bound))
    return result


def resolve_namespace_key(
    node: ast.AST | None,
    environment: Mapping[str, object],
    bindings: Sequence[Binding] = (),
) -> str | None:
    """Normalize a fixed namespace alias, without resolving its members or receivers."""
    from . import _NAMESPACES

    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or any(item.name == node.id for item in bindings):
        return None
    owner = environment.get(node.id)
    parts.reverse()
    key = None
    for index in range(len(parts) + 1):
        for alias, namespace in _NAMESPACES.items():
            if owner is namespace:
                key = ".".join([alias, *parts[index:]])
                break
        if index == len(parts) or not issubclass(type(owner), ModuleType):
            break
        # A qualified import may lead to a registered namespace. Only module
        # dictionaries are traversed; receivers and lazy descriptors stay opaque.
        owner = vars(owner).get(parts[index])
    return key


def resolve_namespace_value(node: ast.AST | None, environment: Mapping[str, object]) -> object:
    """Read configuration from an explicitly registered namespace, never a receiver."""
    from . import _NAMESPACES

    key = resolve_namespace_key(node, environment)
    if key is None:
        return None
    root, *parts = key.split(".")
    value = _NAMESPACES[root]
    for part in parts:
        try:
            value = vars(value).get(part)
        except TypeError:
            return None
    return value


def resolve_constructor(
    node: ast.AST | None, environment: Mapping[str, object], bindings: Sequence[Binding] = ()
) -> str | None:
    """Read only a fixed-namespace source call's canonical policy key."""
    return (
        resolve_namespace_key(node.func, environment, bindings)
        if isinstance(node, ast.Call)
        else None
    )


class Binding(NamedTuple):
    """A source binding site, retained through rewriting without value state."""

    name: str
    node: ast.AST
    kind: str
    annotation: ast.expr | None = None
    dtype: object = None
    direct: bool = False


class PrescanContext:
    """Hand collected syntax facts to one temporary translation.

    ``PrescanCollector.collect`` creates this context from its accumulators and
    then discards the collector. Entry's name allocator and the recursive
    rewriter consume the fact collections without mutating them; ``ModuleContext``
    holds their shared reference for the translation. Referenced nodes belong to
    entry's freshly acquired AST and may be rewritten; the source input is not mutated.
    The context does not own native construction state and is released with the
    parse or macro invocation that uses it.
    """

    def __init__(
        self,
        reserved_names: set[str],
        bindings: dict[ast.AST | None, list[Binding]],
        sites: dict[ast.AST, Binding],
        conditional_outputs: dict[ast.stmt, str],
        namespaces: set[str],
        with_outputs: dict[ast.With, list[str]],
        recursive_functions: set[ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> None:
        # Collected source/entry names seed the allocator; this set stays read-only.
        self.reserved_names = reserved_names
        # Collected, source-ordered declarations per scope drive signature/body lowering.
        # Neither this map nor its declaration lists are extended during rewriting.
        self.bindings = bindings
        # Collected target-node lookup selects declaration/assignment rewrites unchanged.
        self.sites = sites
        # Derive mutable-name sets once for assignment dispatch, then read them only;
        # these are syntax categories, not independently updated runtime value state.
        self.mutable_names = {
            scope: {item.name for item in items if item.kind in ("mutable", "mutable_parameter")}
            for scope, items in bindings.items()
        }
        # Collected branch-ending names select conditional results without later updates.
        self.conditional_outputs = conditional_outputs
        # Collected entry namespaces forbid source rebinding; the set stays read-only.
        self.namespaces = namespaces
        # Collected explicit output names select exports after each with-region exits;
        # rewriting reads the original lists without adding inferred outputs.
        self.with_outputs = with_outputs
        # Collected self-reference facts select standalone declaration-before-body
        # lowering; rewriting does not add or remove functions from this set.
        self.recursive_functions = recursive_functions


class PrescanCollector(ast.NodeVisitor):
    """Collect binding syntax once, then discard all traversal accumulators."""

    def __init__(self, environment: Mapping[str, object], *, filename: str = "<str>") -> None:
        # Fixed lookup inputs last for this scan; never updated by assignments.
        self.environment: Mapping[str, object] = environment
        self.filename: str = filename
        # Accumulators pass to the temporary context; rewriting does not mutate them.
        self.names: set[str] = set(environment)
        self.bindings: dict[ast.AST | None, list[Binding]] = {}
        self.sites: dict[ast.AST, Binding] = {}
        self.outputs: dict[ast.stmt, str] = {}
        self.namespaces: set[str] = set()
        self.exports: dict[ast.With, list[str]] = {}
        self.recursive: set[ast.FunctionDef | ast.AsyncFunctionDef] = set()
        # Active source declarations, used only to recognize self references.
        self.functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.module_name: str | None = None
        # Temporary lexical with-stack routes explicit output calls, then resets.
        self.regions: list[ast.With] = []
        # Lexical scope and language variant restore on function/class exit. direct marks
        # unconditional function-body declarations eligible before signatures.
        self.scope: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | None = None
        self.builder: object = None
        self.direct: bool = False

    def collect(self, tree: ast.Module, *, root_builder: object | None = None) -> PrescanContext:
        """Collect reserved names, scoped declarations and region-result syntax."""
        # Only registered namespace objects establish fixed source aliases.
        self.namespaces.update(
            name
            for name in self.environment
            if resolve_namespace_key(ast.Name(name, ast.Load()), self.environment) is not None
        )
        self.scope = tree
        self.bindings[tree] = []
        # A directly applied decorator has no corresponding decorator AST.
        # Supply its builder as a phase input, not an attachment on the tree.
        self.builder = root_builder
        self.visit(tree)
        # Transfer the completed collections directly. Consumers keep the facts, not
        # this collector, and do not mutate the collections during AST rewriting.
        return PrescanContext(
            self.names,
            self.bindings,
            self.sites,
            self.outputs,
            self.namespaces,
            self.exports,
            self.recursive,
        )

    def _raise_error(self, node: ast.AST, message: str) -> NoReturn:
        raise SyntaxError(
            message,
            (
                self.filename,
                node.lineno,
                node.col_offset + 1,
                None,
                node.end_lineno,
                node.end_col_offset + 1,
            ),
        )

    def _record_binding(
        self,
        name: str,
        node: ast.AST,
        kind: str = "ordinary",
        annotation: ast.expr | None = None,
        dtype: object = None,
    ) -> None:
        if name in self.namespaces:
            self._raise_error(node, f"Script namespace {name!r} cannot be rebound or shadowed")
        self.names.add(name)
        item = Binding(name, node, kind, annotation, dtype, self.direct)
        self.bindings[self.scope].append(item)
        self.sites[node] = item

    def visit_Name(self, node: ast.Name) -> None:
        # -------------------- Pattern --------------------
        # Python source:
        #     value = f()
        #
        # Builder:
        #     value = X.bind_(f(), name="value")
        # -------------------------------------------------
        # Reserve source names and recognize self references before lowering.
        if isinstance(node.ctx, ast.Store) and node.id in self.namespaces:
            self._raise_error(node, f"Script namespace {node.id!r} cannot be rebound or shadowed")
        self.names.add(node.id)
        if isinstance(node.ctx, ast.Load):
            for function in reversed(self.functions):
                if node.id == function.name:
                    self.recursive.add(function)
                    break

    def visit_arg(self, node: ast.arg) -> None:
        # -------------------- Pattern --------------------
        # Python source:
        #     def f(x: ty):
        #         body(x)
        #
        # Builder:
        #     x = X.arg("x", ty)
        # -------------------------------------------------
        # Parameter names cannot shadow registered namespaces.
        if node.arg in self.namespaces:
            self._raise_error(node, f"Script namespace {node.arg!r} cannot be rebound or shadowed")
        self.names.add(node.arg)
        self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> None:
        # -------------------- Pattern --------------------
        # Python source:
        #     import module as alias
        #
        # Builder:
        #     import module as alias
        # -------------------------------------------------
        # Reserve the imported Python name.
        name = node.asname or node.name.split(".")[0]
        if isinstance(self.scope, ast.FunctionDef) and name in self.namespaces:
            self._raise_error(node, f"Script namespace {name!r} cannot be rebound or shadowed")
        self.names.add(name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # -------------------- Pattern --------------------
        # Python source:
        #     class Module:
        #         @X.function
        #         def f():
        #             body()
        #
        # Builder:
        #     with I.ir_module():
        #         with X.function_(decl=True):
        #             X.func_name("f")
        # -------------------------------------------------
        # Class host bindings and members share one lexical scope.
        self.names.add(node.name)
        old, old_module = self.scope, self.module_name
        self.scope, self.module_name = node, node.name
        self.bindings[node] = []
        self.generic_visit(node)
        self.scope, self.module_name = old, old_module

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # -------------------- Pattern --------------------
        # Python source:
        #     def f(x: ty):
        #         body(x)
        #
        # Builder:
        #     with X.function_(decl=True):
        #         X.func_name("f")
        #         x = X.arg("x", ty)
        # -------------------------------------------------
        # Collect this function separately from its enclosing scope.
        self._record_binding(node.name, node, "function")
        old_scope, old_builder, old_direct = self.scope, self.builder, self.direct
        self.scope, self.direct = node, True
        self.functions.append(node)
        self.bindings[node] = []
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if (
                isinstance(target, ast.Attribute)
                and protocol.DECLARATION_KIND.get(resolve_namespace_key(target, self.environment))
                == "function"
            ):
                namespace = resolve_namespace_value(target.value, self.environment)
                if namespace is not None:
                    self.builder = namespace
                    break
        for parameter in getattr(node, "type_params", ()):
            # -------------------- Pattern --------------------
            # Python source:
            #     def f[n]():
            #         body(n)
            #
            # Builder:
            #     n = X.resolve_type_var_("n")
            # -------------------------------------------------
            self._record_binding(parameter.name, parameter, "symbol", dtype="int64")
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            # -------------------- Pattern --------------------
            # Python source:
            #     def f(x: X.int32, tensor: X.Tensor(shape)):
            #         body(x, tensor)
            #
            # Builder:
            #     x = X.arg("x", X.int32)
            #     tensor = X.arg("tensor", X.Tensor(shape))
            # -------------------------------------------------
            # Registered annotation metadata distinguishes mutable parameters
            # without evaluating types.
            annotation = parse_annotation(arg.annotation, self.filename) if arg.annotation else None
            arg.annotation = annotation
            constructor = resolve_namespace_key(
                annotation.func if isinstance(annotation, ast.Call) else annotation,
                self.environment,
            )
            dtype = protocol.TYPE_VAR_DECL.get(constructor)
            self._record_binding(
                arg.arg,
                arg,
                "mutable_parameter"
                if inspect.getattr_static(self.builder, "supports_mutable_declarations", True)
                is True
                and "parameter" in protocol.MUTABLE_CELL_DECL.get(constructor, ())
                else "parameter",
                arg.annotation,
                dtype if not isinstance(annotation, ast.Call) else None,
            )
        for statement in node.body:
            self.visit(statement)
            if isinstance(statement, ast.Return | ast.Raise):
                self.direct = False
        for decorator in node.decorator_list:
            self.visit(decorator)
        if node.returns:
            node.returns = parse_annotation(node.returns, self.filename)
            self.visit(node.returns)
        self._validate_symbols(node)
        self.functions.pop()
        self.scope, self.builder, self.direct = old_scope, old_builder, old_direct

    visit_AsyncFunctionDef = visit_FunctionDef

    def _validate_symbols(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Reject ordinary writes after symbolic introduction in this function."""
        facts = self.bindings[node]
        # Mutable storage and loop/parameter names retain their existing assignment
        # rules. Annotation lambda/comprehension binders are handled by free-name lookup.
        assignable = {
            item.name
            for item in facts
            if item.kind in ("parameter", "mutable_parameter", "mutable", "loop")
        }
        annotations = [item.annotation for item in facts if item.annotation is not None]
        if node.returns is not None:
            annotations.append(node.returns)
        # This temporary diagnostic index is derived from existing source facts;
        # it is not retained in the prescan result or used as value state.
        origins: dict[str, ast.AST] = {}
        for annotation in annotations:
            for name, annotation_origin in collect_annotation_free_names(annotation).items():
                # A prior body target makes this a local annotation operand,
                # not a free symbolic introduction. Explicit symbols below
                # still establish their own origin and reject ordinary writes.
                bound_in_body = any(
                    item.name == name
                    and isinstance(item.node, ast.Name)
                    and (item.node.lineno, item.node.col_offset)
                    < (annotation_origin.lineno, annotation_origin.col_offset)
                    for item in facts
                )
                if name not in assignable and name not in self.namespaces and not bound_in_body:
                    previous = origins.get(name)
                    if previous is None or annotation_origin.lineno < previous.lineno:
                        origins[name] = annotation_origin
        for item in facts:
            if item.kind == "symbol":
                origin = origins.get(item.name)
                if origin is None or item.node.lineno < origin.lineno:
                    origins[item.name] = item.node
        for item in facts:
            # Explicit declarations and mutable updates are not symbol rebindings.
            if item.kind == "symbol" or item.name in assignable:
                continue
            origin = origins.get(item.name)
            if origin is not None and (item.node.lineno, item.node.col_offset) > (
                origin.lineno,
                origin.col_offset,
            ):
                self._raise_error(
                    item.node,
                    f"Symbolic variable {item.name!r} cannot be reassigned; "
                    f"introduced at line {origin.lineno}",
                )

    def _collect_target(
        self,
        target: ast.expr,
        value: ast.expr | None = None,
        annotation: ast.expr | None = None,
        *,
        kind: str = "ordinary",
    ) -> None:
        constructor = resolve_constructor(value, self.environment, self.bindings[self.scope])
        if isinstance(target, ast.Name):
            dtype = protocol.TYPE_VAR_DECL.get(constructor)
            if constructor in protocol.TYPE_VAR_DECL and not value.args and not value.keywords:
                self._record_binding(target.id, target, "symbol", value, dtype)
            elif getattr(self.builder, "supports_mutable_declarations", True) and (
                "call" in protocol.MUTABLE_CELL_DECL.get(constructor, ())
                or (
                    annotation is not None
                    and "annotation"
                    in protocol.MUTABLE_CELL_DECL.get(
                        resolve_namespace_key(
                            annotation.value
                            if isinstance(annotation, ast.Subscript)
                            else annotation,
                            self.environment,
                        ),
                        (),
                    )
                )
            ):
                self._record_binding(target.id, target, "mutable", annotation)
            elif isinstance(value, ast.Name) and value.id == self.module_name:
                self._record_binding(target.id, target, "module_alias")
            elif isinstance(value, ast.Name) and any(
                item.name == value.id and item.kind == "module_alias"
                for item in self.bindings[self.scope]
            ):
                # -------------------- Pattern --------------------
                # Python source:
                #     second_alias = first_alias
                #
                # Builder:
                #     second_alias = first_alias
                # -------------------------------------------------
                # Preserve the active module frame identity.
                self._record_binding(target.id, target, "module_alias")
            else:
                self._record_binding(target.id, target, kind, annotation=annotation)
        elif isinstance(target, ast.Tuple | ast.List):
            values = (
                value.elts
                if isinstance(value, ast.Tuple | ast.List) and len(value.elts) == len(target.elts)
                else [None] * len(target.elts)
            )
            for child, rhs in zip(target.elts, values):
                # One declaration call may return several already-owned values.
                self._collect_target(child, rhs, kind=kind)
        elif isinstance(target, ast.Starred):
            self._collect_target(target.value, kind=kind)
        self.visit(target)

    def visit_Assign(self, node: ast.Assign) -> None:
        # -------------------- Pattern --------------------
        # Python source:
        #     x = rhs
        #
        # Builder:
        #     x = X.bind_(rhs, name="x")
        # -------------------------------------------------
        # Classify lexical targets in source order, including nested destructuring.
        for target in node.targets:
            self._collect_target(target, node.value)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # -------------------- Pattern --------------------
        # Python source:
        #     x: annotation = rhs
        #
        # Builder:
        #     x = X.bind_(rhs, ty=annotation, name="x")
        # -------------------------------------------------
        # Retain decoded annotation syntax and its declaration location.
        node.annotation = parse_annotation(node.annotation, self.filename)
        self._collect_target(node.target, node.value, node.annotation)
        self.visit(node.annotation)
        if node.value:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # -------------------- Pattern --------------------
        # Python source:
        #     n += value
        #
        # Builder:
        #     X.set_mutable_cell_(n, n + value)
        # -------------------------------------------------
        # A name update is a write for symbolic-reassignment diagnostics too.
        self._collect_target(node.target)
        self.visit(node.value)

    def visit_If(self, node: ast.If) -> None:
        # -------------------- Pattern --------------------
        # Python source:
        #     if condition:
        #         y = a
        #     else:
        #         y = b
        #
        # Builder:
        #     with X.if_(condition) as frame:
        #         with X.Then():
        #             y = X.bind_(a, name="y")
        #         with X.Else():
        #             y = X.bind_(b, name="y")
        #     y = frame.var
        # -------------------------------------------------
        # Record matching branch outputs; lexical helpers are assembled during rewriting.
        old_direct, self.direct = self.direct, False
        self.generic_visit(node)
        marker = (
            isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Attribute)
            and node.test.func.attr == "constexpr"
        )
        if not marker and inspect.getattr_static(self.builder, "__tvm_value_if__", False) is True:

            def ending(body: list[ast.stmt]) -> str | None:
                last = body[-1] if body else None
                if (
                    isinstance(last, ast.Assign)
                    and len(last.targets) == 1
                    and isinstance(last.targets[0], ast.Name)
                ):
                    return last.targets[0].id
                if isinstance(last, ast.AnnAssign) and isinstance(last.target, ast.Name):
                    return last.target.id
                return self.outputs.get(last)

            then, otherwise = ending(node.body), ending(node.orelse)
            # Effect-only branches have no Python output. Native branch frames
            # still reject a non-void expression used as an effect-only ending.
            effects = bool(
                node.body
                and node.orelse
                and isinstance(node.body[-1], ast.Expr)
                and isinstance(node.orelse[-1], ast.Expr)
            )
            if not effects:
                if then is None or then != otherwise:
                    location = node.orelse[-1] if node.orelse else node
                    self._raise_error(
                        location, "IR conditional branches must end with the same named output"
                    )
                self.outputs[node] = then
        self.direct = old_direct

    def visit_For(self, node: ast.For) -> None:
        # -------------------- Pattern --------------------
        # Python source:
        #     for i in values:
        #         body(i)
        #
        # Builder:
        #     with X.for_(values, names=("i",)) as i:
        #         X.emit_(body(i))
        # -------------------------------------------------
        # Loop binders remain assignable and are never signature declarations.
        old_direct, self.direct = self.direct, False
        self._collect_target(node.target, kind="loop")
        self.visit(node.iter)
        for statement in node.body + node.orelse:
            self.visit(statement)
        self.direct = old_direct

    def visit_While(self, node: ast.While) -> None:
        # -------------------- Pattern --------------------
        # Python source:
        #     while condition:
        #         body()
        #
        # Builder:
        #     with X.While(condition):
        #         X.emit_(body())
        # -------------------------------------------------
        # Declarations inside this region are conditional.
        old_direct, self.direct = self.direct, False
        self.generic_visit(node)
        self.direct = old_direct

    def visit_With(self, node: ast.With) -> None:
        # -------------------- Pattern --------------------
        # Python source:
        #     with frame as value:
        #         body(value)
        #
        # Builder:
        #     with frame as entered:
        #         value = X.bind_(entered, name="value", frame_value=True)
        #         X.emit_(body(value))
        # -------------------------------------------------
        # Collect targets and explicit region outputs.
        old_direct, self.direct = self.direct, False
        self.regions.append(node)
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self._collect_target(item.optional_vars)
        for statement in node.body:
            self.visit(statement)
        self.regions.pop()
        self.direct = old_direct

    def visit_Call(self, node: ast.Call) -> None:
        # -------------------- Pattern --------------------
        # Python source:
        #     with X.dataflow():
        #         X.output(x, y)
        #
        # Builder:
        #     with X.dataflow() as frame:
        #         X.output(x, y)
        #     x, y = frame.output_vars
        # -------------------------------------------------
        # The enclosing region owns these exported names.
        if self.regions and isinstance(node.func, ast.Attribute) and node.func.attr == "output":
            namespace = resolve_namespace_value(node.func.value, self.environment)
            if (
                namespace is not None
                and self.builder is not None
                and vars(namespace).get("output") is vars(self.builder).get("output")
                and vars(namespace).get("output") is not None
            ):
                names = [arg.id for arg in node.args if isinstance(arg, ast.Name)]
                self.exports.setdefault(self.regions[-1], []).extend(names)
        self.generic_visit(node)
