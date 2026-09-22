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
"""Syntax-only rewriting of registered constructor argument policies.

Existing AST nodes retain all four Python location fields. Parsed strings use a
UTF-8 byte-offset map through the original literal spelling, including escapes
and physical newlines. If a literal cannot be recovered (for example a synthetic
AST or adjacent concatenated literals), ranges conservatively fall back to the
literal's complete original range. Rewriting never evaluates source expressions.
"""

import ast
import copy
import inspect
import linecache
import re

from . import protocol


class _LiteralParser:
    # filename is immutable for one expression rewrite and owns no source/IR cache.
    def __init__(self, filename):
        self.filename = filename

    def _string_expression(self, node):
        try:
            expression = ast.parse(node.value, mode="eval").body
        except SyntaxError as error:
            raise SyntaxError(f"Invalid annotation expression: {error.msg}") from error
        # Example: in R.Tensor(("n + 1",), "float32"), the generated resolve
        # call for n retains the byte range of n inside the quoted literal, not
        # the whole constructor. A triple-quoted physical newline advances lineno;
        # an escaped \n advances decoded input but maps back to the escape's real
        # source bytes. Generated call wrappers copy these four mapped fields.
        source = "".join(linecache.getlines(self.filename))
        literal = ast.get_source_segment(source, node) if source else None
        positions = self._literal_positions(literal, node) if literal else None
        lines = node.value.splitlines(keepends=True)
        for inner in ast.walk(expression):
            if not hasattr(inner, "lineno"):
                continue
            for line_field, column_field in (
                ("lineno", "col_offset"),
                ("end_lineno", "end_col_offset"),
            ):
                line, column = getattr(inner, line_field), getattr(inner, column_field)
                offset = len("".join(lines[: line - 1]).encode("utf-8")) + column
                if positions is not None and offset in positions:
                    line, column = positions[offset]
                else:
                    # Without a reliable spelling map (synthetic or concatenated
                    # literals), preserve the literal's entire real source range.
                    # Never invent physical lines from decoded escape sequences.
                    line = getattr(node, line_field)
                    column = getattr(node, column_field)
                setattr(inner, line_field, line)
                setattr(inner, column_field, column)
        return expression

    @staticmethod
    def _literal_positions(literal, node):
        """Map decoded expression byte offsets back through the literal's escapes."""
        match = re.match("(?i:([rub]*))([\"'])", literal)
        if match is None:
            return None
        prefix, quote = match.groups()
        width = 3 if literal[len(prefix) :].startswith(quote * 3) else 1
        start, stop = len(prefix) + width, len(literal) - width
        delimiter = quote * width
        positions, decoded, offset = {}, "", 0
        index = start

        def location(raw_index):
            before = literal[:raw_index]
            line = node.lineno + before.count("\n")
            column = len(before.rsplit("\n", 1)[-1].encode("utf-8"))
            return line, column + (node.col_offset if line == node.lineno else 0)

        while index < stop:
            end = index + 1
            if literal[index] == "\\" and "r" not in prefix.lower():
                escape = re.match(
                    r"\\(?:N\{[^}]*\}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|x[0-9a-fA-F]{2}|[0-7]{1,3}|\r?\n|.)",
                    literal[index:stop],
                )
                if escape:
                    end = index + len(escape.group())
            piece = literal[index:end]
            try:
                value = ast.literal_eval(prefix + delimiter + piece + delimiter)
            except (SyntaxError, ValueError):
                return None
            positions[offset] = location(index)
            for char in value:
                offset += len(char.encode("utf-8"))
                positions[offset] = location(end)
            decoded += value
            index = end
        return positions if decoded == node.value else None


def rewrite_expression(
    node, resolve, namespace, filename, *, annotation=False, parser_support="_PS"
):
    """Copy an expression and rewrite registered constructor arguments.

    Parameters
    ----------
    node : ast.expr
        Original source expression. It is never mutated.
    resolve : callable
        Static namespace lookup for Name and Attribute nodes. Returns host
        namespaces or callables, or None, without executing descriptors.
    namespace : str
        Generated builder binding used for ``resolve_type_var`` calls.
    filename : str
        Original compilation filename and linecache source key.
    annotation : bool, optional
        Parse a quoted whole annotation before constructor-field rewriting.
        Default is False. True also emits lexical bindings for names
        introduced by signature expression strings.
    parser_support : str, optional
        Hygienically allocated shared parser-support binding. Global-info
        arguments use its ``lookup_global_info`` helper. Default is ``"_PS"``.

    Returns
    -------
    ast.expr
        New expression AST preserving original source ranges.

    Raises
    ------
    SyntaxError
        If a marked string is not a Python expression.
    TypeError or ValueError
        If a registered constructor's signature cannot be inspected.

    Notes
    -----
    Rewriting never evaluates annotations or enters builder frames. Temporary
    literal and symbol transformers belong to this call only. String-derived
    nodes map decoded UTF-8 offsets back through the original literal spelling;
    unrecoverable spellings use the literal's full source range.

    Examples
    --------
    A registered shape string becomes a builder resolution call::

        R.Tensor(("n + 1",), "float32")
        # Rewritten shape: (X.resolve_type_var("n") + 1,)

    The unmarked dtype string remains unchanged.
    """
    literals = _LiteralParser(filename)

    class Symbols(ast.NodeTransformer):
        # Pattern: "n + m * 2" -> X.resolve_type_var("n") + X.resolve_type_var("m") * 2.
        def __init__(self, dtype):
            self.dtype = dtype

        def visit_Attribute(self, current):
            # A recognized namespace in T.max/T.Cast is a Python namespace,
            # not an expression-string symbol. Keep its attribute path intact.
            if resolve(current.value) is not None:
                return current
            return self.generic_visit(current)

        def visit_Call(self, current):
            # Constructor/operator names keep normal Python callable resolution;
            # symbol-bearing operands still resolve through the active builder.
            if resolve(current.func) is None:
                current.func = self.visit(current.func)
            current.args = [self.visit(value) for value in current.args]
            for keyword in current.keywords:
                keyword.value = self.visit(keyword.value)
            return current

        def visit_Name(self, current):
            call = ast.copy_location(
                ast.Call(
                    ast.Attribute(ast.Name(namespace, ast.Load()), "resolve_type_var", ast.Load()),
                    [ast.Constant(current.id)],
                    [] if self.dtype is None else [ast.keyword("dtype", ast.Constant(self.dtype))],
                ),
                current,
            )
            call._tvm_parser_support = True
            if annotation:
                # Signature strings introduce lexical names at their actual
                # evaluation point: ("n + 1", n) works; (n, "n") stays unbound.
                # This generated assignment expression contains only a builder
                # call and preserves the original symbol's source range.
                result = ast.copy_location(
                    ast.NamedExpr(ast.Name(current.id, ast.Store()), call), current
                )
                result._tvm_signature_binding = True
                return result
            return call

    class Rewrite(ast.NodeTransformer):
        def argument(self, current, policy, name):
            kind = policy.fields.get(name) if policy is not None else None
            if kind == "expr_str":
                return self.field(current, policy.expression)
            if kind == "global_info":
                # Keep reference spelling intact: the enclosing module owns
                # interpretation, and concrete values also pass through here.
                result = ast.copy_location(
                    ast.Call(
                        ast.Attribute(
                            ast.Name(parser_support, ast.Load()), "lookup_global_info", ast.Load()
                        ),
                        [self.visit(current)],
                        [],
                    ),
                    current,
                )
                result._tvm_parser_support = True
                return result
            return self.visit(current)

        def field(self, current, policy, nested=False):
            if isinstance(current, ast.Tuple | ast.List):
                current.elts = [self.field(value, policy, True) for value in current.elts]
                return current
            if isinstance(current, ast.Constant) and isinstance(current.value, str):
                if nested or policy.scalar_strings:
                    return Symbols(policy.dtype).visit(literals._string_expression(current))
            return self.visit(current)

        def visit_Call(self, current):
            # Pattern: R.Tensor(("n + 1",), dtype) ->
            # R.Tensor((X.resolve_type_var("n") + 1,), dtype).
            if getattr(current, "_tvm_args_policy_applied", False):
                # The enclosing expression and its children are translated in
                # separate passes. Keep each argument policy applied once while
                # still visiting children that need ordinary syntax rewriting.
                return self.generic_visit(current)
            constructor = resolve(current.func)
            policy = protocol.get_args_policy(constructor)
            parameters = []
            if policy is not None:
                parameters = [
                    p.name
                    for p in inspect.signature(constructor).parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                ]
            current.func = self.visit(current.func)
            positional = []
            known_position = True
            for index, value in enumerate(current.args):
                # An unpacked argument has no statically known parameter or
                # width. Preserve it and subsequent positional arguments;
                # explicit keyword parameters still have known policies.
                known_position = known_position and not isinstance(value, ast.Starred)
                name = parameters[index] if known_position and index < len(parameters) else None
                positional.append(self.argument(value, policy, name))
            current.args = positional
            for keyword in current.keywords:
                keyword.value = self.argument(keyword.value, policy, keyword.arg)
            if policy is not None:
                current._tvm_args_policy_applied = True
            return current

    result = copy.deepcopy(node)
    if annotation and isinstance(result, ast.Constant) and isinstance(result.value, str):
        result = literals._string_expression(result)
    return ast.fix_missing_locations(Rewrite().visit(result))
