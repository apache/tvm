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
"""Translate Python syntax into calls on a context-selected construction namespace.

The translator owns ordering, lexical scopes, and original source locations.
Construction operations and callable syntax metadata come from its environment;
ordinary expressions remain nested Python expressions producing concrete values.
"""

import ast
import builtins
import copy
import inspect
from types import GetSetDescriptorType, MemberDescriptorType

from . import protocol
from .expression import rewrite_expression


class NameCollector(ast.NodeVisitor):
    """Reserve identifiers and collect declaration syntax for one source unit.

    Parameters
    ----------
    names : dict of str to int
        Shared name-to-counter mapping, populated in place. The enclosing
        compilation owns its lifetime.

    Notes
    -----
    Visits reserve function, class, import, parameter, and expression-string
    names before generated bindings are allocated. Source names are unchanged.

    Each visited FunctionDef receives ``_tvm_type_var_declarations``: ordered
    ``(name, expression AST, location AST, body-assignment flag)`` tuples.
    Registered constructor metadata filters these syntax candidates during
    emission. Signature candidates precede direct unconditional body candidates;
    nested functions receive independent lists. Collection executes no source
    expression and retains no IR construction state.
    """

    def __init__(self, names):
        self.names = names

    def visit_Name(self, node):
        self.names.setdefault(node.id, 0)

    def visit_arg(self, node):
        self.names.setdefault(node.arg, 0)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Reserve definition names even if no ast.Name refers to them.
        self.names.setdefault(node.name, 0)
        node._tvm_type_var_declarations = [
            (parameter.arg, parameter.annotation, parameter, False)
            for parameter in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            if parameter.annotation is not None
        ]
        # Only direct statements belong to this function's declaration prescan.
        # Do not descend through conditions, loops, with scopes or nested defs.
        for statement in node.body:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    node._tvm_type_var_declarations.extend(
                        self._body_declarations(target, statement.value)
                    )
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                node._tvm_type_var_declarations.extend(
                    self._body_declarations(statement.target, statement.value)
                )
            elif isinstance(statement, ast.Return | ast.Raise):
                break
        self.generic_visit(node)

    @staticmethod
    def _body_declarations(target, value):
        # Pattern: n = X.int64() or m, n = X.int64(), X.int64(). No argument
        # expressions, starred unpacking or ordinary value-producing calls qualify.
        if isinstance(target, ast.Name) and isinstance(value, ast.Call):
            if not value.args and not value.keywords:
                yield target.id, value, target, True
        elif isinstance(target, ast.Tuple | ast.List) and isinstance(value, ast.Tuple | ast.List):
            if len(target.elts) == len(value.elts):
                for lhs, rhs in zip(target.elts, value.elts):
                    yield from NameCollector._body_declarations(lhs, rhs)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.names.setdefault(node.name, 0)
        self.generic_visit(node)

    def visit_alias(self, node):
        self.names.setdefault(node.asname or node.name.split(".")[0], 0)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            try:
                expression = ast.parse(node.value, mode="eval")
            except SyntaxError:
                return
            for inner in ast.walk(expression):
                if isinstance(inner, ast.Name):
                    self.names.setdefault(inner.id, 0)


class IRBuilderTranspiler(ast.NodeTransformer):
    """Match original Python patterns and emit a builder-program AST.

    Parameters
    ----------
    filename : str
        Original source filename, used in Python compilation and diagnostics.
    environment : dict of str to object
        Host namespace and callable bindings for static metadata lookup.
        Other visible lexical spellings map to None, without retaining values.
        No expression evaluation or concrete symbol construction occurs here.
    builder_name : str
        Collision-free injected alias for the current construction namespace.
    infrastructure_name : str
        Collision-free injected alias for shared execution helpers.
    span : callable
        Map an original AST node to an expression containing location data.
    fresh : callable or None
        Whole-unit name allocator accepting an optional prefix. None selects
        `fresh_unique_name`. Source identifiers are never renamed.
    signature_names : iterable of str, optional
        Initially bound parameter names. Default is None, interpreted as empty.
    nested_function : callable, optional
        Syntax-only nested-function handler. Default is None. Whole-program
        lowering temporarily installs its declaration/definition emitter.
    preserve_return : bool, optional
        Retain Python return semantics in construction helpers. Default is False.
    name_map : dict of str to int, optional
        Shared name reservations and counters. Default is None, which initializes
        the mapping from environment names. `NameCollector` reserves source names
        before whole-program lowering.
    track_span : bool, optional
        Emit IR source instrumentation. Default is True. False retains AST ranges
        without expression ``at`` calls or builder ``span`` keywords.

    Notes
    -----
    ``filename``, the span and name callbacks, and ``infrastructure_name`` belong
    to one source unit. ``namespace_bindings`` maps visible source spellings to
    host namespaces and callables, or None for opaque lexical values;
    assignments and parameters shadow entries.
    ``dialect_prefix`` selects the builder alias and is restored after functions.
    ``bound`` holds identifier strings only. ``optional`` maps conditionally
    exported names to AST reads of cached named outputs, so a name defined by
    only one branch is checked when read.

    Each lexical helper copies and restores ``bound``, ``namespace_bindings``,
    and ``optional``. No concrete value or IR identity is stored. Return policy
    and syntax hooks configure one body pass; nested scopes share the whole-unit
    allocator without resetting it. Unsupported syntax raises SyntaxError during
    translation rather than construction.

    Examples
    --------
    Representative source patterns lower to builder calls::

        v = e                     # v = X.bind_(e, name="v", span=location)
        v: ty = e                 # X.bind_ also receives ty
        return e                  # X.return_(e)
        for i in range(n): ...    # with X.for_(X.range_(n)) as i: ...

    An annotation-only binding uses the builder MISSING initializer; a bare
    return supplies no argument. Grid iteration retains its grid expression and
    tuple target. Function factories declare signatures before invoking body
    definitions; builders retain parameter and symbol identities. Marked strings
    become ``resolve_type_var`` calls, while unmarked dtype strings remain
    literal. No rendered Python is reparsed to recover locations.
    """

    def __init__(
        self,
        filename,
        environment,
        builder_name,
        infrastructure_name,
        span,
        fresh,
        signature_names=None,
        nested_function=None,
        preserve_return=False,
        name_map=None,
        track_span=True,
        parser_support_name="_PS",
        definition_scopes_name=None,
    ):
        self.filename = filename
        self.track_span = track_span
        self.namespace_bindings = dict(environment)
        self.dialect_prefix = builder_name
        self.infrastructure_name = infrastructure_name
        self.parser_support_name = parser_support_name
        self.definition_scopes_name = definition_scopes_name
        self.span = span
        self.fresh = fresh or self.fresh_unique_name
        self.name_map = name_map if name_map is not None else dict.fromkeys(environment, 0)
        self.bound = set(signature_names or ())
        self.optional = {}
        self.nested_function = nested_function
        self.preserve_return = preserve_return
        self.annotation_scope = None

    def fresh_unique_name(self, prefix="_t"):
        """Allocate an unused identifier from the shared whole-unit name map.

        Parameters
        ----------
        prefix : str, optional
            Generated identifier prefix. Default is ``"_t"``.

        Returns
        -------
        str
            An identifier absent from the current reservations.

        Notes
        -----
        Updates ``name_map`` with the reservation and next prefix counter. All
        nested functions share this dictionary; no source name is renamed and
        no lexical scope resets the allocator. No builder frame is entered.
        """
        counter = self.name_map.get(prefix, 0)
        while f"{prefix}{counter}" in self.name_map:
            counter += 1
        name = f"{prefix}{counter}"
        self.name_map[prefix] = counter + 1
        self.name_map[name] = 0
        return name

    def transform_statements(self, body):
        """Translate a sequence of original statements into builder syntax.

        Parameters
        ----------
        body : sequence of ast.stmt
            Original statements, deep-copied before visiting.

        Returns
        -------
        list of ast.stmt
            Lowered statements in source order. A source statement can expand
            into several generated statements.

        Raises
        ------
        SyntaxError
            If a statement or expression uses unsupported construction syntax.

        Notes
        -----
        Source nodes are unchanged. Syntactic name state is updated for later
        statements in this body. Builder operations receive statement locations
        through ``span=``; expression locations use value-wrapping ``at`` calls.
        Only syntax callbacks run here; no construction frame is entered.
        """
        result = []
        for statement in copy.deepcopy(body):
            translated = self.visit(statement)
            if translated is not None:
                block = translated if isinstance(translated, list) else [translated]
                result.extend(block)
        return result

    def _error(self, node, message):
        raise SyntaxError(message, (self.filename, node.lineno, node.col_offset + 1, None))

    @staticmethod
    def _located(value, original):
        return ast.copy_location(value, original)

    def _name(self, name, original, store=False):
        return self._located(ast.Name(name, ast.Store() if store else ast.Load()), original)

    def _attribute(self, namespace, member, original):
        return self._located(
            ast.Attribute(self._name(namespace, original), member, ast.Load()), original
        )

    def _call(self, namespace, member, args, original, **keywords):
        # Disabled tracking removes instrumentation, while copy_location below
        # preserves Python diagnostics for the same generated computation.
        if not self.track_span:
            if namespace == self.parser_support_name and member == "at":
                return args[1]
            keywords.pop("span", None)
            keywords.pop("name_span", None)
        return self._located(
            ast.Call(
                self._attribute(namespace, member, original),
                args,
                [ast.keyword(arg=key, value=value) for key, value in keywords.items()],
            ),
            original,
        )

    def _operation(self, member, args, original, **keywords):
        return self._call(
            self.dialect_prefix, member, args, original, span=self.span(original), **keywords
        )

    def _statement(self, expression, original):
        return self._located(ast.Expr(expression), original)

    def _assign(self, name, value, original):
        return self._located(ast.Assign([self._name(name, original, True)], value), original)

    def _cache(self, value, original, prefix="value"):
        name = self.fresh(prefix)
        return self._assign(name, value, original), self._name(name, original)

    def _resolve(self, node):
        if isinstance(node, ast.Name):
            return self.namespace_bindings.get(node.id, getattr(builtins, node.id, None))
        if isinstance(node, ast.Attribute):
            owner = self._resolve(node.value)
            if owner is None:
                return None
            value = inspect.getattr_static(owner, node.attr, None)
            if isinstance(value, staticmethod):
                return value.__func__
            if inspect.isfunction(value):
                if inspect.ismodule(owner) or inspect.isclass(owner):
                    return value
                # Functions stored on namespaces/instances are ordinary values,
                # not bound methods. Read only a builtin instance-dict slot;
                # arbitrary source descriptors must not execute during lookup.
                dictionary = inspect.getattr_static(owner, "__dict__", None)
                if isinstance(dictionary, GetSetDescriptorType | MemberDescriptorType):
                    if node.attr in dictionary.__get__(owner):
                        return value
                return value.__get__(owner)
            # Inspect syntax metadata without executing a source-level descriptor.
            if hasattr(type(value), "__get__"):
                return None
            return value
        return None

    def _constexpr_operand(self, node):
        if not isinstance(node, ast.Call) or not protocol.is_constexpr_marker(
            self._resolve(node.func)
        ):
            return None
        if len(node.args) != 1 or node.keywords or isinstance(node.args[0], ast.Starred):
            self._error(node, "constexpr expects exactly one controlling value")
        for child in ast.walk(node.args[0]):
            if isinstance(child, ast.NamedExpr | ast.Await | ast.Yield | ast.YieldFrom):
                self._error(child, f"Unsupported expression: {type(child).__name__}")
        return node.args[0]

    def _logical_expression(self, node, index=0):
        value = node.values[index]
        marker = self._constexpr_operand(value)
        if marker is not None:
            result = self._scoped_host(marker, resolve_optional=True)
            if index + 1 < len(node.values):
                result = self._located(
                    ast.BoolOp(
                        copy.deepcopy(node.op), [result, self._logical_expression(node, index + 1)]
                    ),
                    node,
                )
            return result
        result = self._expression(value)
        method = "and_" if isinstance(node.op, ast.And) else "or_"
        for position in range(index + 1, len(node.values)):
            if self._constexpr_operand(node.values[position]) is not None:
                return self._call(
                    self.dialect_prefix,
                    method,
                    [result, self._logical_expression(node, position)],
                    node,
                )
            result = self._call(
                self.dialect_prefix, method, [result, self._expression(node.values[position])], node
            )
        return result

    def _expression(self, original, *, attach_span=True):
        node = copy.deepcopy(original)
        if isinstance(getattr(node, "ctx", None), ast.Store):
            return node
        if isinstance(node, ast.Name) and node.id in self.optional:
            value = self._call(
                self.infrastructure_name,
                "require_defined",
                [copy.deepcopy(self.optional[node.id]), ast.Constant(node.id)],
                node,
            )
            return self._call(
                self.parser_support_name, "at", [self.span(original), value], original
            )
        marker = self._constexpr_operand(node)
        if marker is not None:
            value = self._scoped_host(marker, resolve_optional=True)
            if not attach_span:
                return value
            return self._call(
                self.parser_support_name, "at", [self.span(original), value], original
            )
        node = rewrite_expression(
            node,
            self._resolve,
            self.dialect_prefix,
            self.filename,
            parser_support=self.parser_support_name,
        )
        source_call = isinstance(node, ast.Call) and not getattr(node, "_tvm_parser_support", False)
        if isinstance(node, ast.Await | ast.Yield | ast.YieldFrom) or (
            isinstance(node, ast.NamedExpr) and not getattr(node, "_tvm_signature_binding", False)
        ):
            self._error(original, f"Unsupported expression: {type(node).__name__}")
        # Pattern: not e -> X.not_(e). These operations construct IR eagerly;
        # the builders, not Python truth testing, own compiled short-circuiting.
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            node = self._call(self.dialect_prefix, "not_", [self._expression(node.operand)], node)
        # Explicit markers retain host selection; unmarked expressions build IR.
        elif isinstance(node, ast.BoolOp):
            node = self._logical_expression(node)
        elif isinstance(node, ast.IfExp):
            marker = self._constexpr_operand(node.test)
            if marker is not None:
                node = self._located(
                    ast.IfExp(
                        self._scoped_host(marker, resolve_optional=True),
                        self._expression(node.body),
                        self._expression(node.orelse),
                    ),
                    node,
                )
            else:
                node = self._call(
                    self.dialect_prefix,
                    "if_then_else_",
                    [
                        self._expression(node.test),
                        self._expression(node.body),
                        self._expression(node.orelse),
                    ],
                    node,
                )
        # Pattern: a < b < c -> X.and_(a < b, b < c, chain=(a, b, c)).
        elif isinstance(node, ast.Compare) and len(node.ops) > 1:
            node = self._compare_chain(node)
        elif isinstance(node, ast.Compare):
            self._expression_children(node)
            node = self._comparison(node.left, node.ops[0], node.comparators[0], node)
        # Pattern: f"text{e}" -> preserve fragments and translate only expressions.
        elif isinstance(node, ast.JoinedStr):
            # JoinedStr's children must remain literal fragments/FormattedValue nodes.
            for child in node.values:
                if isinstance(child, ast.FormattedValue):
                    child.value = self._expression(child.value)
                    if child.format_spec is not None:
                        child.format_spec = self._format_spec(child.format_spec)
        else:
            self._expression_children(node)
        # Pattern: f(args, keyword=value) remains an ordinary Python call.
        # Recursive argument translation above preserves order and locations;
        # callable overloads own concrete IR argument/result semantics.
        if isinstance(node, ast.Starred) or (
            isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        ):
            return node
        if source_call and self.track_span:
            return self._call(
                self.parser_support_name,
                "with_at_scope",
                [self.span(original), self._lambda([], node, original)],
                original,
            )
        if not attach_span:
            return node
        return self._call(self.parser_support_name, "at", [self.span(original), node], original)

    def _scoped_host(self, original, *, resolve_optional=False):
        """Wrap host-source calls without changing module-level Python syntax."""
        owner = self

        class Calls(ast.NodeTransformer):
            def __init__(self):
                self.locals = set()

            def visit_Name(self, node):
                if (
                    resolve_optional
                    and isinstance(node.ctx, ast.Load)
                    and node.id in owner.optional
                    and node.id not in self.locals
                ):
                    return owner._call(
                        owner.infrastructure_name,
                        "require_defined",
                        [copy.deepcopy(owner.optional[node.id]), ast.Constant(node.id)],
                        node,
                    )
                return node

            def visit_Lambda(self, node):
                node.args.defaults = [self.visit(value) for value in node.args.defaults]
                node.args.kw_defaults = [
                    self.visit(value) if value else None for value in node.args.kw_defaults
                ]
                previous = self.locals
                arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                arguments += [arg for arg in (node.args.vararg, node.args.kwarg) if arg]
                self.locals = previous | {argument.arg for argument in arguments}
                node.body = self.visit(node.body)
                self.locals = previous
                return node

            def visit_ListComp(self, node):
                previous = self.locals
                self.locals = set(previous)
                for generator in node.generators:
                    generator.iter = self.visit(generator.iter)
                    self.locals.update(
                        item.id for item in ast.walk(generator.target) if isinstance(item, ast.Name)
                    )
                    generator.ifs = [self.visit(value) for value in generator.ifs]
                if isinstance(node, ast.DictComp):
                    node.key, node.value = self.visit(node.key), self.visit(node.value)
                else:
                    node.elt = self.visit(node.elt)
                self.locals = previous
                return node

            visit_SetComp = visit_ListComp
            visit_DictComp = visit_ListComp
            visit_GeneratorExp = visit_ListComp

            def visit_Call(self, node):
                node = self.generic_visit(node)
                if not owner.track_span or getattr(node, "_tvm_parser_support", False):
                    return node
                return owner._call(
                    owner.parser_support_name,
                    "with_at_scope",
                    [owner.span(node), owner._lambda([], node, node)],
                    node,
                )

        return Calls().visit(copy.deepcopy(original))

    def _lambda(self, names, value, original):
        arguments = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in names],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        return self._located(ast.Lambda(arguments, value), original)

    def _comparison(self, left, operation, right, original):
        operations = {
            ast.Lt: "lt",
            ast.LtE: "le",
            ast.Gt: "gt",
            ast.GtE: "ge",
            ast.Eq: "eq",
            ast.NotEq: "ne",
        }
        if type(operation) in operations:
            return self._call(
                self.dialect_prefix,
                operations[type(operation)],
                [left, right],
                original,
            )
        return self._located(ast.Compare(left, [operation], [right]), original)

    def _compare_chain(self, node):
        # An immediately invoked Python lambda binds eagerly evaluated operands
        # once in source order. The builder constructs each written comparison
        # from concrete operands. chain=(a, b, c) records syntax provenance so
        # builders also bind shared middle operands once at compiled runtime,
        # where duplicating an effectful expression would change semantics.
        operands = [node.left, *node.comparators]
        names = [self.fresh("operand") for _ in operands]
        comparisons = []
        for left, operation, right in zip(names, node.ops, names[1:]):
            comparison = self._comparison(
                self._name(left, node), operation, self._name(right, node), node
            )
            comparisons.append(comparison)
        value = self._call(
            self.dialect_prefix,
            "and_",
            comparisons,
            node,
            chain=ast.Tuple([self._name(name, node) for name in names], ast.Load()),
        )
        return self._located(
            ast.Call(
                self._lambda(names, value, node),
                [self._expression(value) for value in operands],
                [],
            ),
            node,
        )

    def _format_spec(self, node):
        for child in node.values:
            if isinstance(child, ast.FormattedValue):
                child.value = self._expression(child.value)
                if child.format_spec is not None:
                    child.format_spec = self._format_spec(child.format_spec)
        return node

    def _expression_children(self, node):
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.expr):
                setattr(node, field, self._expression(value))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, ast.expr):
                        value[index] = self._expression(item)
                    elif isinstance(item, ast.AST):
                        self._expression_children(item)
            elif isinstance(value, ast.AST):
                self._expression_children(value)

    def _index(self, node):
        if isinstance(node, ast.Slice):
            fields = [
                self._expression(value) if value is not None else ast.Constant(None)
                for value in (node.lower, node.upper, node.step)
            ]
            return self._call(self.infrastructure_name, "slice", fields, node)
        if isinstance(node, ast.Tuple):
            return self._located(
                ast.Tuple([self._index(value) for value in node.elts], ast.Load()), node
            )
        return self._expression(node)

    def _bind(self, target, value, statement, ty=None, frame_value=False):
        # Pattern: name = value -> X.bind_; Python name analysis tracks previous values.
        if isinstance(target, ast.Name):
            keywords = {"name": ast.Constant(target.id), "name_span": self.span(target)}
            if frame_value:
                keywords["frame_value"] = ast.Constant(True)
            if ty is not None:
                keywords["ty"] = ty
            if target.id in self.bound:
                keywords["previous"] = self._name(target.id, target)
            elif target.id in self.optional:
                keywords["previous"] = copy.deepcopy(self.optional[target.id])
            self.bound.add(target.id)
            self.optional.pop(target.id, None)
            return [
                self._assign(
                    target.id, self._operation("bind_", [value], statement, **keywords), target
                )
            ]
        # Pattern: object.field = value -> X.setattr(object, "field", value).
        if isinstance(target, ast.Attribute):
            return [
                self._statement(
                    self._operation(
                        "setattr",
                        [self._expression(target.value), ast.Constant(target.attr), value],
                        statement,
                    ),
                    statement,
                )
            ]
        # Pattern: object[index] = value -> X.setitem(object, index, value).
        if isinstance(target, ast.Subscript):
            return [
                self._statement(
                    self._operation(
                        "setitem",
                        [self._expression(target.value), self._index(target.slice), value],
                        statement,
                    ),
                    statement,
                )
            ]
        if isinstance(target, ast.Tuple | ast.List):
            # Each Python unpack finishes before visiting that level's targets. A nested
            # unpack occurs only when reached, preserving assignment and failure order.
            names = [self.fresh("unpack") for _ in target.elts]
            pattern = []
            for item, name in zip(target.elts, names):
                temporary = self._name(name, item, True)
                pattern.append(
                    self._located(ast.Starred(temporary, ast.Store()), item)
                    if isinstance(item, ast.Starred)
                    else temporary
                )
            unpack = self._call(self.dialect_prefix, "unpack", [value], target)
            assignment = self._located(
                ast.Assign([ast.Tuple(pattern, ast.Store())], unpack), target
            )
            result = [assignment]
            for item, name in zip(target.elts, names):
                item = item.value if isinstance(item, ast.Starred) else item
                result.extend(
                    self._bind(item, self._name(name, item), statement, frame_value=frame_value)
                )
            return result
        self._error(target, f"Unsupported assignment target: {type(target).__name__}")

    def visit_Assign(self, node):
        # Pattern: var = value -> var = X.bind_(value, name="var", span=...).
        # Cache the RHS once before chained stores, index loads and nested unpacking.
        cache, value = self._cache(self._expression(node.value), node.value)
        result = [cache]
        resolved = self._resolve(node.value)
        for target in node.targets:
            result.extend(self._bind(target, copy.deepcopy(value), node))
            if isinstance(target, ast.Name):
                if resolved is not None:
                    self.namespace_bindings[target.id] = resolved
                else:
                    self.namespace_bindings.pop(target.id, None)
        return result

    def visit_AnnAssign(self, node):
        # Pattern: var: ty -> var = X.bind_(I.MISSING, ty=ty, name="var", ...).
        # Pattern: var: ty = value -> var = X.bind_(value, ty=ty, name="var", ...).
        if not isinstance(node.target, ast.Name):
            self._error(node.target, "An annotated binding requires a name")
        result = []
        if node.value is None:
            value = self._attribute(self.infrastructure_name, "MISSING", node)
        else:
            cache, value = self._cache(self._expression(node.value), node.value)
            result.append(cache)
        annotation = rewrite_expression(
            node.annotation,
            self._resolve,
            self.dialect_prefix,
            self.filename,
            annotation=True,
            parser_support=self.parser_support_name,
        )
        if self.annotation_scope is not None:
            annotation = self.annotation_scope(annotation)
        result.extend(self._bind(node.target, value, node, self._expression(annotation)))
        return result

    def visit_AugAssign(self, node):
        # Pattern: target += value -> load target once, then X.bind_/setitem/setattr.
        result = []
        if isinstance(node.target, ast.Name):
            old, previous = self._cache(
                self._expression(self._located(ast.Name(node.target.id, ast.Load()), node.target)),
                node.target,
                "old",
            )
            result.append(old)
            value = self._located(ast.BinOp(previous, node.op, self._expression(node.value)), node)
            value = self._call(self.parser_support_name, "at", [self.span(node), value], node)
            return result + self._bind(node.target, value, node)
        if not isinstance(node.target, ast.Subscript | ast.Attribute):
            self._error(node.target, "An augmented assignment requires a name, attribute, or index")
        base_stmt, base = self._cache(
            self._expression(node.target.value), node.target.value, "base"
        )
        result.append(base_stmt)
        if isinstance(node.target, ast.Attribute):
            key, operation = ast.Constant(node.target.attr), "setattr"
            load = self._located(
                ast.Attribute(copy.deepcopy(base), node.target.attr, ast.Load()), node.target
            )
        else:
            key_stmt, key = self._cache(self._index(node.target.slice), node.target.slice, "key")
            result.append(key_stmt)
            operation = "setitem"
            load = self._located(
                ast.Subscript(copy.deepcopy(base), copy.deepcopy(key), ast.Load()), node.target
            )
        old_stmt, old = self._cache(
            self._call(self.parser_support_name, "at", [self.span(node.target), load], node.target),
            node.target,
            "old",
        )
        result.append(old_stmt)
        value = self._located(ast.BinOp(old, node.op, self._expression(node.value)), node)
        value = self._call(self.parser_support_name, "at", [self.span(node), value], node)
        result.append(self._statement(self._operation(operation, [base, key, value], node), node))
        return result

    def visit_Expr(self, node):
        # Calls attach their own source context; emission only consumes the result.
        return self._statement(
            self._call(self.dialect_prefix, "emit_", [self._expression(node.value)], node), node
        )

    def visit_Return(self, node):
        # Pattern: return value -> X.return_(value); return -> X.return_().
        value = self._expression(node.value) if node.value is not None else None
        if self.preserve_return:
            return self._located(ast.Return(value), node)
        return self._statement(
            self._operation("return_", [] if value is None else [value], node), node
        )

    def visit_Break(self, node):
        # Pattern: break -> X.break_().
        return self._statement(self._operation("break_", [], node), node)

    def visit_Continue(self, node):
        # Pattern: continue -> X.continue_().
        return self._statement(self._operation("continue_", [], node), node)

    def visit_Assert(self, node):
        # Pattern: assert condition, message -> X.assert_(condition, message).
        message = self._expression(node.msg) if node.msg is not None else ast.Constant("")
        return self._statement(
            self._operation("assert_", [self._expression(node.test), message], node), node
        )

    @staticmethod
    def _assigned_names(body):
        names = set()

        class Names(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    names.add(node.id)

            def visit_FunctionDef(self, node):
                # Nested definitions introduce a name, not their body bindings.
                names.add(node.name)

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Lambda(self, node):
                pass

        visitor = Names()
        for statement in body:
            visitor.visit(statement)
        return names

    def _scope(self, body, original, prefix, initial=None, return_bindings=False):
        outer_bound, outer_environment, outer_optional = (
            self.bound,
            self.namespace_bindings,
            self.optional,
        )
        referenced = {
            node.id
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
        }
        captures = sorted((outer_bound | outer_optional.keys()).intersection(referenced))
        self.bound, self.namespace_bindings = set(outer_bound), dict(outer_environment)
        self.optional = {
            name: (
                self._name(name, original)
                if name in captures and not self.preserve_return
                else value
            )
            for name, value in outer_optional.items()
        }
        prefix_statements = [] if initial is None else initial()
        translated = prefix_statements + self.transform_statements(body)
        if return_bindings and not self.preserve_return:
            names = sorted(self._assigned_names(body))
            values = [
                (
                    self._name(name, original)
                    if name in self.bound
                    else copy.deepcopy(
                        self.optional.get(
                            name, self._attribute(self.infrastructure_name, "MISSING", original)
                        )
                    )
                )
                for name in names
            ]
            translated.append(
                self._located(
                    ast.Return(ast.Dict([ast.Constant(name) for name in names], values)), original
                )
            )
        self.bound, self.namespace_bindings, self.optional = (
            outer_bound,
            outer_environment,
            outer_optional,
        )
        if self.preserve_return:
            return translated or [self._located(ast.Pass(), original)]
        # A helper isolates construction locals. Optional exports are captured as
        # values or MISSING, and checked only when the original body reads them.
        defaults = [
            (
                self._name(name, original)
                if name in outer_bound
                else copy.deepcopy(outer_optional[name])
            )
            for name in captures
        ]
        helper = self.fresh(prefix)
        arguments = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in captures],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=defaults,
        )
        definition = self._located(
            ast.FunctionDef(
                helper, arguments, translated or [self._located(ast.Pass(), original)], [], None
            ),
            original,
        )
        if "type_params" in ast.FunctionDef._fields:
            definition.type_params = []
        invocation = self._located(ast.Call(self._name(helper, original), [], []), original)
        return [definition, self._statement(invocation, original)]

    def _exports(self, frame, candidates, original):
        # Pattern: names assigned in a completed region -> one named builder
        # request per candidate, e.g. y = X.frame_result(frame, "y"). Builders
        # decide which names are designated outputs; absent results remain
        # MISSING so reads cannot expose branch-local bindings accidentally.
        result = []
        for name in sorted(candidates):
            assignment, value = self._cache(
                self._call(
                    self.dialect_prefix,
                    "frame_result",
                    [self._name(frame, original), ast.Constant(name)],
                    original,
                ),
                original,
                "output",
            )
            result.append(assignment)
            condition = self._located(
                ast.Compare(
                    copy.deepcopy(value),
                    [ast.IsNot()],
                    [self._attribute(self.infrastructure_name, "MISSING", original)],
                ),
                original,
            )
            result.append(
                self._located(
                    ast.If(condition, [self._assign(name, copy.deepcopy(value), original)], []),
                    original,
                )
            )
            if name not in self.bound:
                self.optional[name] = copy.deepcopy(value)
        return result

    def _with(self, context, body, original, target=None):
        return self._located(
            ast.With([ast.withitem(context, target)], body or [ast.Pass()]), original
        )

    def visit_If(self, node):
        marker = self._constexpr_operand(node.test)
        if marker is not None:
            # A Python branch shares the enclosing lexical and builder scope.
            # Translate both syntactic arms, then let execution select one arm.
            outer_bound = set(self.bound)
            outer_environment = dict(self.namespace_bindings)
            outer_optional = dict(self.optional)
            condition_stmt, condition = self._cache(
                self._scoped_host(marker, resolve_optional=True), node.test, "condition"
            )
            then_body = self.transform_statements(node.body)
            then_bound = set(self.bound)
            self.bound = set(outer_bound)
            self.namespace_bindings = dict(outer_environment)
            self.optional = dict(outer_optional)
            else_body = self.transform_statements(node.orelse)
            self.bound.intersection_update(then_bound)
            candidates = self._assigned_names(node.body + node.orelse)
            initial = []
            for name in sorted(candidates):
                if name not in outer_bound:
                    initial.append(
                        self._assign(
                            name,
                            (
                                copy.deepcopy(outer_optional[name])
                                if name in outer_optional
                                else self._attribute(self.infrastructure_name, "MISSING", node)
                            ),
                            node,
                        )
                    )
                if name not in self.bound:
                    self.optional[name] = self._name(name, node)
                else:
                    self.optional.pop(name, None)
                self.namespace_bindings.pop(name, None)
            branch = self._located(ast.If(condition, then_body or [ast.Pass()], else_body), node)
            return [condition_stmt, *initial, branch]
        frame = self.fresh("conditional")
        condition_stmt, condition = self._cache(self._expression(node.test), node.test, "condition")
        if self.preserve_return:
            then_call = self._scope(node.body, node, "then")
            else_call = self._scope(node.orelse, node, "else")
            definitions = []
        else:
            then_def, then_invoke = self._scope(node.body, node, "then", return_bindings=True)
            else_def, else_invoke = self._scope(node.orelse, node, "else", return_bindings=True)
            definitions = [then_def, else_def]
            then_call, else_call = [then_invoke], [else_invoke]
        branches = [self._with(self._operation("Then", [], node), then_call, node)]
        if node.orelse:
            branches.append(self._with(self._operation("Else", [], node), else_call, node))
        region = self._with(
            self._operation("If", [condition], node), branches, node, self._name(frame, node, True)
        )
        exports = self._exports(frame, self._assigned_names(node.body + node.orelse), node)
        return [condition_stmt, *definitions, region, *exports]

    def visit_For(self, node):
        # Pattern: for i, j in T.grid(m, n) -> with X.for_(T.grid(m, n)) as (i, j).
        # Pattern: for i in range(...) -> with X.for_(X.range_(...)) as i.
        # Only the actual builtin binding is normalized; a shadowing callable stays intact.
        if node.orelse:
            self._error(node, "A construction loop does not support an else clause")
        iterable = copy.deepcopy(node.iter)
        if isinstance(iterable, ast.Call) and self._resolve(iterable.func) is builtins.range:
            iterable.func = self._attribute(self.dialect_prefix, "range_", iterable.func)
        old_bound, old_optional, old_environment = (
            self.bound,
            self.optional,
            self.namespace_bindings,
        )
        self.bound, self.optional, self.namespace_bindings = (
            set(old_bound),
            dict(old_optional),
            dict(old_environment),
        )
        for target in ast.walk(node.target):
            if isinstance(target, ast.Name):
                self.bound.add(target.id)
                self.optional.pop(target.id, None)
                self.namespace_bindings.pop(target.id, None)
        body = self.transform_statements(node.body)
        self.bound, self.optional, self.namespace_bindings = (
            old_bound,
            old_optional,
            old_environment,
        )
        return self._with(
            self._operation(
                "for_",
                [self._expression(iterable)],
                node,
                names=(
                    ast.Constant(node.target.id)
                    if isinstance(node.target, ast.Name)
                    else ast.Tuple(
                        [
                            ast.Constant(
                                "*" + item.value.id if isinstance(item, ast.Starred) else item.id
                            )
                            for item in node.target.elts
                        ],
                        ast.Load(),
                    )
                ),
            ),
            body,
            node,
            copy.deepcopy(node.target),
        )

    def visit_While(self, node):
        # Pattern: while condition: ... -> with X.While(condition): translated body.
        if node.orelse:
            self._error(node, "A construction loop does not support an else clause")
        frame = self.fresh("loop")
        region = self._with(
            self._operation("While", [self._expression(node.test)], node),
            self._scope(node.body, node, "body"),
            node,
            self._name(frame, node, True),
        )
        return [region, *self._exports(frame, self._assigned_names(node.body), node)]

    def _bind_entered(self, target, value, original):
        # Context/iteration targets introduce lexical names; they never reassign
        # an outer mutable variable merely because its source spelling matches.
        for item in ast.walk(target):
            if isinstance(item, ast.Name):
                self.bound.discard(item.id)
                self.optional.pop(item.id, None)
        return self._bind(target, value, original, frame_value=True)

    def visit_With(self, node):
        # Pattern: with context as target: ... -> builder context plus lexical exports.
        item = node.items[0]
        body = node.body
        if len(node.items) > 1:
            nested = self._located(ast.With(node.items[1:], node.body), node)
            body = [nested]
        manager, value = self.fresh("context"), self.fresh("entered")
        cache = self._assign(manager, self._expression(item.context_expr), item.context_expr)
        initial = (
            None
            if item.optional_vars is None
            else lambda: self._bind_entered(
                item.optional_vars, self._name(value, item.context_expr), node
            )
        )
        region = self._with(
            self._name(manager, node),
            self._scope(body, node, "scope", initial),
            node,
            self._name(value, node, True),
        )
        return [cache, region, *self._exports(manager, self._assigned_names(body), node)]

    def visit_FunctionDef(self, node):
        # Pattern: nested def -> ordinary Python definition or complete registered builder program.
        self.bound.add(node.name)
        if self.nested_function is not None:
            return self.nested_function(node)
        # Ordinary construction helpers retain normal Python execution. Registered
        # function-kind entry points are resolved by the enclosing compiler callback.
        if any(
            getattr(self._resolve(decorator), "__tvm_function_info__", None)
            for decorator in node.decorator_list
        ):
            self._error(node, "A registered nested function requires a function compiler")
        return node

    def visit_Nonlocal(self, node):
        # Source acquisition already supplies enclosing Python bindings. Like the
        # original parser, a lexical declaration emits no IR and does not mutate
        # the host closure while constructing the function.
        return self._located(ast.Pass(), node)

    def visit_Pass(self, node):
        # Pattern: pass -> pass; it has no builder effect.
        return node

    def generic_visit(self, node):
        if isinstance(node, ast.stmt):
            self._error(node, f"Unsupported statement: {type(node).__name__}")
        return super().generic_visit(node)

    def function_metadata(self, node, *, allow_python=False):
        """Read decorator metadata without evaluating source annotations.

        Parameters
        ----------
        node : ast.FunctionDef
            Original function definition with decorator syntax.
        allow_python : bool, optional
            Permit unregistered ordinary Python helpers. Default is False.

        Returns
        -------
        info : FunctionDecoratorInfo
            Registered metadata or ordinary-Python metadata when permitted.
        options : ast.Dict
            Unevaluated builder options after keyword-name mapping and defaults.

        Raises
        ------
        SyntaxError
            If a construction decorator uses positional options, or registration
            is missing while ordinary Python functions are disallowed.

        Notes
        -----
        Metadata lookup does not execute annotations or option expressions and
        enters no builder frame. Returned option syntax belongs to this lowering
        pass; registered metadata is shared and treated as read-only.
        """
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            kind = protocol.function_info(self._resolve(target))
            if kind is None:
                continue
            mapping = kind.option_map or {}
            values = {key: ast.Constant(value) for key, value in (kind.defaults or {}).items()}
            expansions = []
            if isinstance(decorator, ast.Call):
                if decorator.args:
                    self._error(decorator, "Function decorators accept keyword options only")
                for keyword in decorator.keywords:
                    if keyword.arg is None:
                        expansions.append(keyword.value)
                    elif keyword.arg != "check_well_formed":
                        values[mapping.get(keyword.arg, keyword.arg)] = copy.deepcopy(keyword.value)
            result = ast.Dict(
                [ast.Constant(key) for key in values] + [None] * len(expansions),
                list(values.values()) + expansions,
            )
            return kind, ast.copy_location(result, node)
        if allow_python:
            return protocol.FunctionDecoratorInfo(None, python=True), ast.Dict([], [])
        self._error(node, f"Function {node.name!r} has no registered construction kind")

    def function_program(self, node, runtime, bindings, *, local=False):
        """Emit a declaration factory and deferred function-definition callback.

        Parameters
        ----------
        node : ast.FunctionDef
            Original function with a registered decorator.
        runtime : str
            Generated alias for the builder runtime.
        bindings : dict of str to object
            Caller-owned mapping receiving injected builder namespace aliases.
        local : bool, optional
            Request a nested-function reference from the builder. Default is False.

        Returns
        -------
        tuple of (list of ast.stmt, str, str) or None
            Factory statements, function-record binding, and body binding. None
            indicates a registered ordinary Python function.

        Raises
        ------
        SyntaxError
            If signatures, decorators, or body syntax cannot be lowered.

        Notes
        -----
        The method allocates generated names and restores body-local syntax state
        following recursive translation. It neither executes annotations nor
        creates IR. The generated factory enters a builder declaration frame;
        its deferred callback defines the body only when invoked.
        """
        # Pattern: @X.function def f(a: ty) -> ret: body -> declaration factory + definition body.
        # A factory gives each function an ordinary Python closure: parameters and
        # symbols from sibling signatures never overwrite one another's captures.
        node = copy.deepcopy(node)
        node.args.args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        node.args.posonlyargs, node.args.kwonlyargs, node.args.kw_defaults = [], [], []
        kind, options = self.function_metadata(node)
        if kind.python:
            return None
        builder = self.fresh("_t")
        bindings[builder] = kind.builder
        record, factory, body_name, captures = (self.fresh("_t") for _ in range(4))
        original_builder = self.dialect_prefix
        self.dialect_prefix = builder
        record_expr = self._call(
            runtime,
            "FunctionRecord",
            [
                self._name(builder, node),
                ast.Constant(node.name),
                self._scoped_host(options),
                self.span(node),
            ],
            node,
            local=ast.Constant(local),
            captures=self._name(captures, node),
            parameters=ast.Tuple(
                [ast.Constant(argument.arg) for argument in node.args.args], ast.Load()
            ),
        )
        factory_body = [self._assign(record, record_expr, node)]
        if node.args.posonlyargs or node.args.kwonlyargs or node.args.vararg or node.args.kwarg:
            self._error(node, "IR signatures require ordinary named parameters")
        annotation_names = {
            item.id
            for argument in node.args.args
            if argument.annotation is not None
            for item in ast.walk(argument.annotation)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
        }
        if node.returns is not None:
            annotation_names.update(
                item.id
                for item in ast.walk(node.returns)
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
            )
        declaration = []
        introduced_names = set()
        signature_setters = {}

        def bind_signature(annotation):
            # Scoped source calls run in thunks. A walrus inside such a thunk
            # would bind only its lambda, hiding signature symbols from sibling
            # constructors, later annotations and the deferred body. Setters
            # instead update the factory's lexical cell at the original string
            # occurrence, without resolving or binding any symbol early.
            transformer = self

            class BindSignature(ast.NodeTransformer):
                def visit_NamedExpr(self, current):
                    if not getattr(current, "_tvm_signature_binding", False):
                        return self.generic_visit(current)
                    name = current.target.id
                    if name not in signature_setters:
                        setter = transformer.fresh("signature_bind")
                        value = transformer.fresh("signature_value")
                        signature_setters[name] = setter
                        definition = transformer._located(
                            ast.FunctionDef(
                                setter,
                                transformer._lambda([], ast.Constant(None), current).args,
                                [
                                    transformer._located(ast.Nonlocal([name]), current),
                                    transformer._assign(
                                        name, transformer._name(value, current), current
                                    ),
                                    transformer._located(
                                        ast.Return(transformer._name(value, current)), current
                                    ),
                                ],
                                [],
                                None,
                            ),
                            current,
                        )
                        definition.args.args = [ast.arg(value)]
                        if "type_params" in ast.FunctionDef._fields:
                            definition.type_params = []
                        factory_body.append(definition)
                    result = transformer._located(
                        ast.Call(
                            transformer._name(signature_setters[name], current),
                            [self.visit(current.value)],
                            [],
                        ),
                        current,
                    )
                    result._tvm_parser_support = True
                    return result

            return BindSignature().visit(annotation)

        for parameter in getattr(node, "type_params", []):
            # Pattern: def f[n: int](...) -> n = X.resolve_type_var("n")
            # in this function's fresh symbol frame. Only scalar parameters fit
            # the registered builder protocol; Python owns the syntax itself.
            if not isinstance(parameter, getattr(ast, "TypeVar", ())):
                self._error(parameter, "Only scalar type parameters are supported")
            bound = getattr(parameter, "bound", None)
            if bound is not None and self._resolve(bound) is not int:
                self._error(parameter, "A symbolic type parameter bound must be int")
            if getattr(parameter, "default_value", None) is not None:
                self._error(parameter, "A symbolic type parameter cannot have a default")
            declaration.append(
                self._assign(
                    parameter.name,
                    self._call(
                        builder, "resolve_type_var", [ast.Constant(parameter.name)], parameter
                    ),
                    parameter,
                )
            )
        # Pattern: f(A: X.Buffer(("n",)), n: X.int64), or a direct body
        # n = X.int64(): reserve the declared type before signature shapes.
        # The shared name prescan collects candidates; registered metadata alone
        # selects declarations here. Body constructors are NOT executed early:
        # emit their registered dtype spelling and let TypeVarFrame construct it.
        # The original body call and uniform bind_ remain at their source position.
        local_names = self._assigned_names(node.body) | {arg.arg for arg in node.args.args}
        for name, annotation, location, in_body in getattr(node, "_tvm_type_var_declarations", ()):
            target = annotation.func if isinstance(annotation, ast.Call) else annotation
            if in_body and any(
                isinstance(item, ast.Name) and item.id in local_names for item in ast.walk(target)
            ):
                continue  # A local namespace/callable binding shadows outer metadata.
            constructor = self._resolve(target)
            metadata = getattr(constructor, "__tvm_type_var_decl__", None)
            parameter_dtype = getattr(constructor, "__tvm_parameter_dtype__", None)
            if in_body:
                if metadata is None or not isinstance(metadata.dtype, str):
                    continue
                value = ast.copy_location(ast.Constant(metadata.dtype), annotation)
            elif metadata is not None or parameter_dtype is not None:
                value = copy.deepcopy(annotation)
            else:
                continue
            declaration.append(
                self._statement(
                    self._call(
                        record,
                        "predeclare",
                        [ast.Constant(name), value, self.span(location)],
                        location,
                    ),
                    location,
                )
            )
        capture_position = len(declaration)
        ordered_parameters = sorted(
            node.args.args,
            key=lambda parameter: not protocol.is_constexpr_marker(
                self._resolve(parameter.annotation)
            ),
        )
        for parameter in ordered_parameters:
            if parameter.annotation is None:
                self._error(parameter, f"Parameter {parameter.arg!r} requires an annotation")
            annotation = rewrite_expression(
                parameter.annotation,
                self._resolve,
                builder,
                self.filename,
                annotation=True,
                parser_support=self.parser_support_name,
            )
            annotation_names.update(
                item.id
                for item in ast.walk(annotation)
                if isinstance(item, ast.Name)
                and isinstance(item.ctx, ast.Load)
                and item.id not in (builder, self.parser_support_name)
            )
            introduced_names.update(
                call.args[0].value
                for call in ast.walk(annotation)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "resolve_type_var"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)
            )
            declaration.append(
                self._assign(
                    parameter.arg,
                    self._call(
                        record,
                        "parameter_lazy",
                        [
                            ast.Constant(parameter.arg),
                            ast.Lambda(
                                ast.arguments(
                                    posonlyargs=[],
                                    args=[],
                                    kwonlyargs=[],
                                    kw_defaults=[],
                                    defaults=[],
                                ),
                                self._expression(bind_signature(annotation)),
                            ),
                            self.span(parameter),
                        ],
                        parameter,
                    ),
                    parameter,
                )
            )
        if node.returns is not None:
            annotation = rewrite_expression(
                node.returns,
                self._resolve,
                builder,
                self.filename,
                annotation=True,
                parser_support=self.parser_support_name,
            )
            # Return expression strings also assign lexical names. Seed any
            # enclosing binding before earlier parameter expressions read it.
            annotation_names.update(
                item.id
                for item in ast.walk(annotation)
                if isinstance(item, ast.Name)
                and isinstance(item.ctx, ast.Load)
                and item.id not in (builder, self.parser_support_name)
            )
            introduced_names.update(
                call.args[0].value
                for call in ast.walk(annotation)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "resolve_type_var"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)
            )
            declaration.append(
                self._statement(
                    self._call(
                        record,
                        "returns",
                        [self._expression(bind_signature(annotation))],
                        node.returns,
                    ),
                    node.returns,
                )
            )

        def captured_value(name):
            fallback = self._attribute(self.infrastructure_name, "MISSING", node)
            if hasattr(builtins, name):
                alias = self.fresh("_t")
                bindings[alias] = getattr(builtins, name)
                fallback = self._name(alias, node)
            return self._call(captures, "get", [ast.Constant(name), fallback], node)

        # Signature annotations have their own lexical cells. Only names
        # introduced by DSL strings share a cell with the deferred body; an
        # ordinary annotation name never turns a body global into a closure.
        signature_names = (
            (annotation_names | introduced_names)
            - {argument.arg for argument in node.args.args}
            - {parameter.name for parameter in getattr(node, "type_params", [])}
        )
        annotation_aliases = {
            name: name if name in introduced_names else self.fresh("_annotation")
            for name in sorted(signature_names)
        }
        factory_body.extend(
            self._assign(alias, captured_value(name), node)
            for name, alias in annotation_aliases.items()
        )
        capture_statements = [
            self._assign(
                alias,
                self._call(record, "capture", [ast.Constant(name), captured_value(name)], node),
                node,
            )
            for name, alias in annotation_aliases.items()
        ]
        transformer = self

        class AnnotationScope(ast.NodeTransformer):
            def visit_Name(self, current):
                if isinstance(current.ctx, ast.Load) and current.id in annotation_aliases:
                    return transformer._call(
                        transformer.infrastructure_name,
                        "require_defined",
                        [
                            transformer._name(annotation_aliases[current.id], current),
                            ast.Constant(current.id),
                        ],
                        current,
                    )
                return current

        declaration = [AnnotationScope().visit(statement) for statement in declaration]
        # Resolve captures inside the retained TypeVarFrame, after explicit
        # declarations, but before any annotation constructor executes.
        declaration[capture_position:capture_position] = capture_statements
        factory_body.append(
            self._with(self._call(record, "declaration", [], node), declaration, node)
        )
        factory_body.append(
            self._assign(node.name, self._attribute(record, "reference", node), node)
        )

        old_bound, old_optional, old_environment, old_nested, old_annotation_scope = (
            self.bound,
            self.optional,
            self.namespace_bindings,
            self.nested_function,
            self.annotation_scope,
        )
        self.bound, self.optional = {arg.arg for arg in node.args.args}, {}
        self.namespace_bindings = dict(old_environment)
        # Python determines local shadowing for the entire function, including
        # uses before an assignment. Do not apply namespace/range policies there.
        for name in self._assigned_names(node.body) | self.bound:
            self.namespace_bindings.pop(name, None)
            self.namespace_bindings[name] = None

        body_annotation_aliases = {}
        annotation_locals = (
            local_names
            | introduced_names
            | {parameter.name for parameter in getattr(node, "type_params", [])}
        )

        class BodyAnnotationScope(ast.NodeTransformer):
            def __init__(self):
                self.locals = set(annotation_locals)

            def visit_Lambda(self, current):
                current.args.defaults = [self.visit(value) for value in current.args.defaults]
                current.args.kw_defaults = [
                    self.visit(value) if value is not None else None
                    for value in current.args.kw_defaults
                ]
                previous = self.locals
                arguments = [
                    *current.args.posonlyargs,
                    *current.args.args,
                    *current.args.kwonlyargs,
                ]
                arguments += [arg for arg in (current.args.vararg, current.args.kwarg) if arg]
                self.locals = previous | {argument.arg for argument in arguments}
                current.body = self.visit(current.body)
                self.locals = previous
                return current

            def visit_ListComp(self, current):
                previous = self.locals
                self.locals = set(previous)
                for generator in current.generators:
                    generator.iter = self.visit(generator.iter)
                    self.locals.update(
                        item.id for item in ast.walk(generator.target) if isinstance(item, ast.Name)
                    )
                    generator.ifs = [self.visit(value) for value in generator.ifs]
                if isinstance(current, ast.DictComp):
                    current.key, current.value = self.visit(current.key), self.visit(current.value)
                else:
                    current.elt = self.visit(current.elt)
                self.locals = previous
                return current

            visit_SetComp = visit_ListComp
            visit_DictComp = visit_ListComp
            visit_GeneratorExp = visit_ListComp

            def visit_Name(self, current):
                if not isinstance(current.ctx, ast.Load) or current.id in self.locals:
                    return current
                if current.id in (builder, transformer.parser_support_name):
                    return current
                name = current.id
                if name not in body_annotation_aliases:
                    body_annotation_aliases[name] = transformer.fresh("_annotation")
                # Only the host binding is retained in the factory. Resolve
                # symbols when this annotation executes in the function frame,
                # after its RHS, just like the original annotated assignment.
                captured = transformer._call(
                    record,
                    "capture",
                    [ast.Constant(name), transformer._name(body_annotation_aliases[name], current)],
                    current,
                )
                return transformer._call(
                    transformer.infrastructure_name,
                    "require_defined",
                    [captured, ast.Constant(name)],
                    current,
                )

        self.annotation_scope = lambda annotation: BodyAnnotationScope().visit(annotation)

        def nested(child):
            nested_kind, _ = self.function_metadata(child, allow_python=True)
            if nested_kind.python:
                return copy.deepcopy(child)
            statements, child_record, child_body = self.function_program(
                child, runtime, bindings, local=True
            )
            return [
                *statements,
                self._assign(child.name, self._attribute(child_record, "reference", child), child),
                self._statement(
                    self._call(child_record, "define", [self._name(child_body, child)], child),
                    child,
                ),
            ]

        self.nested_function = nested
        statements = self.transform_statements(node.body)
        (
            self.bound,
            self.optional,
            self.namespace_bindings,
            self.nested_function,
            self.annotation_scope,
        ) = (
            old_bound,
            old_optional,
            old_environment,
            old_nested,
            old_annotation_scope,
        )
        factory_body.extend(
            self._assign(alias, captured_value(name), node)
            for name, alias in body_annotation_aliases.items()
        )
        arguments = copy.deepcopy(node.args)
        for argument in arguments.args:
            argument.annotation = None
        arguments.defaults = []
        # Bare host TypeVars in annotations are resolved by constructor builders.
        # Capture their resulting values for free body names without inspecting
        # them here. Source locals/parameters keep Python's original shadowing.
        # Only syntactically referenced signature names can capture enclosing
        # symbols. An unrelated outer n must not constrain a local n declaration.
        capture_names = (annotation_names | introduced_names) - {
            parameter.name for parameter in getattr(node, "type_params", [])
        }
        record_expr.keywords.append(
            ast.keyword(
                "capture_names",
                ast.Tuple([ast.Constant(name) for name in sorted(capture_names)], ast.Load()),
            )
        )
        body_names = {
            item.id
            for statement in statements
            for item in ast.walk(statement)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
        }
        # Annotation-only reads now use their hygienic aliases, so only actual
        # body references participate in lexical capture defaults.
        # Signature symbols may need builder-owned TypeVar resolution. Every
        # other body reference keeps ordinary Python lexical lookup; namespace
        # membership never determines whether a name is a capture.
        free_names = (annotation_names | introduced_names) & body_names - self._assigned_names(
            node.body
        ) - {arg.arg for arg in node.args.args}
        # Recursive references use the factory's finalized function binding,
        # never an older outer function with the same spelling.
        free_names.discard(node.name)
        for name in sorted(free_names):
            arguments.kwonlyargs.append(ast.arg(name))
            arguments.kw_defaults.append(
                self._call(record, "capture", [ast.Constant(name), self._name(name, node)], node)
            )
        definition = self._located(
            ast.FunctionDef(body_name, arguments, statements or [ast.Pass()], [], None), node
        )
        if "type_params" in ast.FunctionDef._fields:
            definition.type_params = []
        if not local:
            definition._tvm_source_name = node.name
            definition._tvm_signature_names = (
                introduced_names
                | {parameter.name for parameter in getattr(node, "type_params", [])}
                | {node.name}
                | set(getattr(node, "_tvm_module_names", ()))
            )
        factory_body.append(definition)
        factory_body.append(
            self._located(
                ast.Return(
                    ast.Tuple([self._name(record, node), self._name(body_name, node)], ast.Load())
                ),
                node,
            )
        )
        factory_def = self._located(
            ast.FunctionDef(
                factory,
                ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(captures)],
                    kwonlyargs=[],
                    kw_defaults=[],
                    # Nested bodies may use namespaces from module globals
                    # without loading them into their Python local dictionary.
                    # Locals override globals exactly as lexical lookup does.
                    defaults=[
                        ast.Dict(
                            keys=[None, None, None],
                            values=[
                                self._call(self.infrastructure_name, "globals", [], node),
                                self._call(
                                    self.definition_scopes_name,
                                    "get",
                                    [ast.Constant(node.name), ast.Dict([], [])],
                                    node,
                                )
                                if self.definition_scopes_name is not None and not local
                                else ast.Dict([], []),
                                self._call(self.infrastructure_name, "locals", [], node),
                            ],
                        )
                    ],
                ),
                factory_body,
                [],
                None,
            ),
            node,
        )
        if "type_params" in ast.FunctionDef._fields:
            factory_def.type_params = []
        unpack = self._located(
            ast.Assign(
                [
                    ast.Tuple(
                        [self._name(record, node, True), self._name(body_name, node, True)],
                        ast.Store(),
                    )
                ],
                ast.Call(self._name(factory, node), [], []),
            ),
            node,
        )
        self.dialect_prefix = original_builder
        return [factory_def, unpack], record, body_name

    def program(self, tree, runtime, original_name, bindings):
        """Emit a complete module or function builder program from source AST.

        Parameters
        ----------
        tree : ast.Module
            Source module whose final statement is a function or module class.
        runtime : str
            Injected builder-runtime alias.
        original_name : str
            Injected binding for the original class object, if available.
        bindings : dict of str to object
            Caller-owned mapping receiving generated builder namespace aliases.

        Returns
        -------
        module : ast.Module
            Complete generated Python program, ready for compilation.
        result_name : str
            Generated binding containing the program's construction result.

        Raises
        ------
        SyntaxError
            If source structure, declarations, or function syntax is unsupported.

        Notes
        -----
        Generated names share the enclosing unit's allocator; execution remains
        separate from translation. Every wrapper inherits its source construct's
        range, while original descendants retain finer ranges. Missing location
        fields are filled without rendering and reparsing source code.
        """
        nodes = list(tree.body)
        if not nodes or not isinstance(nodes[-1], ast.FunctionDef | ast.ClassDef):
            self._error(tree.body[0], "Source must contain one function or module class")
        root, prefix = nodes[-1], nodes[:-1]
        is_module = isinstance(root, ast.ClassDef)
        members = root.body if is_module else [root]
        functions = [item for item in members if isinstance(item, ast.FunctionDef)]
        if len({item.name for item in functions}) != len(functions):
            self._error(root, "Duplicate function declaration")
        program, result = self.fresh("_t"), self.fresh("_t")
        context = self._call(
            runtime,
            "ModuleProgram",
            [
                ast.Constant(root.name if is_module else None),
                self._name(original_name, root) if is_module else ast.Constant(None),
                ast.Tuple(copy.deepcopy(root.bases) if is_module else [], ast.Load()),
            ],
            root,
        )
        body = [
            self._assign(
                item.name, self._call(program, "reserve", [ast.Constant(item.name)], item), item
            )
            for item in functions
        ]
        if is_module:
            body.append(self._assign(root.name, self._attribute(program, "namespace", root), root))
        for member in members:
            if isinstance(member, ast.FunctionDef):
                continue
            # Pattern: class-level statement -> same host statement, with module
            # member publication after named assignments. Builders classify values.
            body.append(self._scoped_host(member))
            targets = (
                member.targets
                if isinstance(member, ast.Assign)
                else [member.target]
                if isinstance(member, ast.AnnAssign)
                else []
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    body.append(
                        self._assign(
                            target.id,
                            self._call(
                                program,
                                "member",
                                [ast.Constant(target.id), self._name(target.id, target)],
                                target,
                            ),
                            target,
                        )
                    )
        definitions = []
        records = []
        for function in functions:
            kind, _ = self.function_metadata(function)
            if kind.python:
                # Pattern: @I.pyfunc def f(...): body -> unchanged host function
                # plus opaque builder registration.
                host = copy.deepcopy(function)
                host.decorator_list = []
                body.append(host)
                body.append(
                    self._assign(
                        function.name,
                        self._call(
                            program,
                            "python",
                            [
                                ast.Constant(function.name),
                                self._name(function.name, function),
                                ast.Constant(ast.unparse(function)),
                                self.span(function),
                            ],
                            function,
                        ),
                        function,
                    )
                )
                continue
            if is_module:
                function._tvm_module_names = [root.name, *[item.name for item in functions]]
            declaration, record, callback = self.function_program(function, runtime, bindings)
            body.extend(declaration)
            definitions.append(
                self._statement(
                    self._call(record, "define", [self._name(callback, function)], function),
                    function,
                )
            )
            records.append(record)
        body.extend(definitions)
        output = (
            self._attribute(program, "result", root)
            if is_module
            else self._attribute(records[0], "function", root)
        )
        translated = [
            # The frontend already executed imports and adapted their bindings.
            # Re-executing them here would restore original builder namespaces.
            *copy.deepcopy(
                [item for item in prefix if not isinstance(item, ast.Import | ast.ImportFrom)]
            ),
            self._with(context, body, root, self._name(program, root, True)),
            self._assign(result, output, root),
        ]
        return ast.fix_missing_locations(ast.Module(translated, [])), result
