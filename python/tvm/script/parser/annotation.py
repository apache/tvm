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
"""Decode source annotation strings and retain their literal source ranges.

Existing AST nodes retain all four Python location fields. Parsed strings use a
UTF-8 byte-offset map through the original literal spelling, including escapes
and physical newlines. If a literal cannot be recovered (for example a synthetic
AST or adjacent concatenated literals), ranges conservatively fall back to the
literal's complete original range. Rewriting never evaluates source expressions.
"""

from __future__ import annotations

import ast
import linecache
import re


class _LiteralParser:
    # filename is immutable for one expression rewrite and owns no source/IR cache.
    def __init__(self, filename: str) -> None:
        self.filename: str = filename

    def _parse_string_expression(self, node: ast.Constant) -> ast.expr:
        try:
            expression = ast.parse(node.value, mode="eval").body
        except SyntaxError as error:
            raise SyntaxError(
                f"Invalid annotation expression: {error.msg}",
                (
                    self.filename,
                    node.lineno,
                    node.col_offset + 1,
                    None,
                    node.end_lineno,
                    node.end_col_offset + 1,
                ),
            ) from error
        # In a quoted whole annotation such as "R.Tensor((n + 1,), 'float32')",
        # n retains its byte range inside the literal, not the whole constructor.
        # A triple-quoted physical newline advances lineno;
        # an escaped \n advances decoded input but maps back to the escape's real
        # source bytes. Generated call wrappers copy these four mapped fields.
        source = "".join(linecache.getlines(self.filename))
        literal = ast.get_source_segment(source, node) if source else None
        positions = self._read_literal_positions(literal, node) if literal else None
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
    def _read_literal_positions(
        literal: str, node: ast.Constant
    ) -> dict[int, tuple[int, int]] | None:
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

        def location(raw_index: int) -> tuple[int, int]:
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


def parse_annotation(node: ast.expr, filename: str) -> ast.expr:
    """Decode a quoted whole annotation without interpreting its Python names."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _LiteralParser(filename)._parse_string_expression(node)
    return node
