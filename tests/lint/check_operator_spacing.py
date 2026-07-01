#!/usr/bin/env python3
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

"""
Conservative operator-spacing checker for '<' and '>'.

It flags only mixed-side spacing for single-character '<' or '>' ops that are
likely to be binary comparisons, while attempting to avoid templates, casts,
macros, and stream/shift operators.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

CHECK_EXTENSIONS = {
    "cc",
    "c",
    "h",
    "hh",
    "hpp",
    "cu",
    "cuh",
    "cpp",
    "cxx",
}

# Match any single '<' or '>' with optional surrounding spaces and capture the symbol
_OP_RE = re.compile(r"(?P<left_char>.)(?P<op>\s*(?P<sym>[<>])\s*)(?P<right_char>.)")

ALNUM_UNDERSCORE = re.compile(r"[A-Za-z0-9_]")

# Tokens or patterns that strongly indicate the line is not a plain comparison
SKIP_KEYWORDS = [
    "static_cast",
    "std::",
    "::",
    "template",
    "TVM_FFI",
    "TVM_FFI_ICHECK",
    "TVM_FFI_THROW",
    "->",
]

def git_ls_files() -> List[str]:
    cmd = ["git", "ls-files"]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print("git ls-files failed:", proc.stderr.strip(), file=sys.stderr)
        sys.exit(2)
    return [p for p in proc.stdout.splitlines() if p.strip()]

def should_check_file(path: str) -> bool:
    suffix = Path(path).suffix.lstrip(".")
    return suffix in CHECK_EXTENSIONS

def is_preprocessor_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("#")

def contains_shift_or_stream(line: str) -> bool:
    return "<<" in line or ">>" in line

def contains_skip_keyword(line: str) -> bool:
    for k in SKIP_KEYWORDS:
        if k in line:
            return True
    return False

def is_part_of_two_char_operator(line: str, sym_pos: int) -> bool:
    # avoid matching <=, >=, <<=, >>= or combinations where '=' or another < or > is adjacent
    if sym_pos - 1 >= 0 and line[sym_pos - 1] in "<>=":
        return True
    if sym_pos + 1 < len(line) and line[sym_pos + 1] in "<>=":
        return True
    return False

def looks_like_template_close(line: str, match: re.Match) -> bool:
    # heuristics: if left char is alnum and the token after the op is an identifier start,
    # it's probably "vector<int> v" or similar. Check a few characters after op.
    left = match.group("left_char")
    start = match.start("op")
    after = line[start + len(match.group("op")) :]
    if left and ALNUM_UNDERSCORE.match(left):
        if after and (ALNUM_UNDERSCORE.match(after[0]) or (after[0].isspace() and len(after) > 1 and ALNUM_UNDERSCORE.match(after[1]))):
            return True
    return False

def check_file(path: str) -> List[Tuple[int, int, str]]:
    violations: List[Tuple[int, int, str]] = []
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if is_preprocessor_line(line):
            continue
        if contains_shift_or_stream(line):
            continue
        if contains_skip_keyword(line):
            continue

        for m in _OP_RE.finditer(line):
            sym = m.group("sym")
            # sym index in the original line:
            sym_index = m.start("op") + m.group("op").find(sym)
            # skip if it's part of <=, >=, <<=, >>=, etc.
            if is_part_of_two_char_operator(line, sym_index):
                continue

            op = m.group("op")
            left_has_space = op.startswith(" ")
            right_has_space = op.endswith(" ")

            # OK if both sides have spaces or neither side has space
            if (not left_has_space and not right_has_space) or (left_has_space and right_has_space):
                continue

            # Skip common non-comparison patterns: template-close, cast-like, punctuation adjacency
            if looks_like_template_close(line, m):
                continue

            left_char = m.group("left_char")
            right_char = m.group("right_char")
            # skip if punctuation suggests generic/template/angle-bracket use
            if left_char in ":,([{<" or right_char in ":,)]}>":
                continue

            # If the line contains 'template' or 'static_cast' we already skipped above,
            # but re-check broader patterns: if '(' immediately follows right side, it's often a cast or template call
            after = line[m.end("op"):]
            if after.startswith("(") or after.startswith("<"):
                continue

            col = m.start("op") + 1
            msg = f"Inconsistent spacing around '{sym}' operator (mixed sides): {line.strip()}"
            violations.append((lineno, col, msg))
    return violations

def main() -> int:
    files = git_ls_files()
    files_to_check = [f for f in files if should_check_file(f)]
    all_violations = {}
    for f in files_to_check:
        hits = check_file(f)
        if hits:
            all_violations[f] = hits

    if all_violations:
        print("Operator spacing violations found:")
        for f, hits in all_violations.items():
            for lineno, col, msg in hits:
                print(f"{f}:{lineno}:{col}: {msg}")
        print("\nRule: either `x<y` (no spaces) or `x < y` (spaces on both sides). Mixed forms like `x< y` or `x <y` are disallowed.")
        return 1
    print("check_operator_spacing.py: all checks passed.")
    return 0

if __name__ == "__main__":
    rc = main()
    sys.exit(rc)