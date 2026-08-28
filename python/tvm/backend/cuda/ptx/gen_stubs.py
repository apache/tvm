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
"""Thin generator 1: emit a ``.pyi`` stub for the ``T.ptx`` namespace.

Run manually (never at import time)::

    python -m tvm.backend.cuda.ptx.gen_stubs -o python/tvm/script/tirx.pyi

The output is a *module stub for* ``tvm.script.tirx``: that module is
virtual at runtime (dialect-registry ``__getattr__``), but a ``.pyi`` file
at ``python/tvm/script/tirx.pyi`` makes Pyright/Pylance resolve it — giving
VS Code completion for ``T.ptx.`` chains. All other members fall back to
``Any`` via the stub's module ``__getattr__``, matching today's behavior.

Each family's chain class lists every slot token flatly, so the stub also
type-checks chains the runtime would reject (e.g. repeated tokens) — the
correct trade-off for a completion-only stub; legality is enforced at trace
time.

Only imports :mod:`.table` (tvm-free), so it runs without a built tvm.
Regenerate whenever the instruction table changes (a unit test diffs the
checked-in stub against this generator).
"""

import argparse
import sys
import textwrap
from pathlib import Path

from .table import TABLE, InstructionEntry, escape_token

# The checked-in stub this module generates.
STUB_PATH = Path(__file__).resolve().parents[3] / "script" / "tirx.pyi"


# The signature a modifier-dependent operand list collapses to. It is also a
# catch-all for the trailing round-trip arguments, so a class that uses it must
# not also declare `*args`.
_CATCH_ALL = ["*__operands: Any"]


def _operand_params(entry: InstructionEntry) -> list[str]:
    """One stub parameter per call argument, from the single call-layout definition."""
    if any(callable(s.lanes) for s in entry.operands):
        # The group length depends on the modifiers, so there is no single
        # signature to write down.
        return list(_CATCH_ALL)
    out = []
    for slot in entry.operands:
        if slot.kind == "imm" and slot.literal is not None:
            continue
        for lane in range(slot.lanes):
            out.append(f"{slot.name}{lane}: Any" if slot.lanes > 1 else f"{slot.name}: Any")
    return out


def _chain_class(family: str, entries: list[InstructionEntry]) -> str:
    """One chain class per *family* — several entries may share a mnemonic."""
    cls = f"_Chain_{family}"
    tokens = sorted({tok for e in entries for slot in e.slots for tok in slot.choices})
    lines = [f"class {cls}:"]
    entry = entries[0]
    doc = "; ".join(
        f"{s.name}∈{{{','.join(s.choices)}}}{' (opt)' if s.optional else ''}" for s in entry.slots
    )
    if entry.check is not None and entry.check.__doc__:
        check_doc = " ".join(entry.check.__doc__.split())
        doc = f"{doc} — {check_doc}" if doc else check_doc
    if len(entries) > 1:
        shapes = dict.fromkeys(
            "(" + ", ".join(_operand_params(e)).replace(": Any", "") + ")" for e in entries
        )
        doc = (
            f"{len(entries)} entries sharing this mnemonic; PTX puts their difference in the "
            f"operand list, so the call selects one. Shapes: {'; '.join(shapes)}"
        )
    # *args covers the printed round-trip form (trailing modifier strings,
    # positional pred) so re-parsed scripts type-check too. Families with more
    # than one shape take their operands through *args for the same reason.
    params = ["self"]
    operands = _operand_params(entry) if len(entries) == 1 else []
    params += operands
    if operands != _CATCH_ALL:
        # A single entry whose group lengths are modifier-dependent already
        # spells its operands as the catch-all; a second one would not even
        # parse ("Only one '*' parameter allowed").
        params.append("*args: Any")
    params.append("pred: Any = None")
    if any(e.has_dst for e in entries):
        params.append("preserve_dst: bool = False")
    signature = f"def __call__({', '.join(params)}) -> None"
    # Emit the shape ruff format would produce, so the generated text needs no
    # formatter to be canonical: a docstring that fits on one line closes on
    # that line. 4 indent + 3 opening quotes + text + 3 closing quotes must
    # stay <= 100.
    doc_lines = textwrap.wrap(f"`{family}` — {doc or '(no modifiers)'}", width=88)
    if len(doc_lines) == 1:
        lines.append(f'    """{doc_lines[0]}"""')
    else:
        lines.append('    """' + doc_lines[0])
        lines.extend(f"    {line}" for line in doc_lines[1:])
        lines.append('    """')
    lines.append("")
    for tok in tokens:
        attr = escape_token(tok)
        if not attr.isidentifier():
            # A token like `16x64b` cannot be an attribute name, so the chain
            # form cannot reach it; those variants are written as strings,
            # `T.ptx["tcgen05.ld.sync.aligned.16x64b.x4.b32"](...)`.
            continue
        lines.append(f"    {attr}: {cls}")
    if len(f"    {signature}: ...") > 100:
        joined = ",\n        ".join(params)
        ret = signature[signature.rindex(")") + 1 :]
        signature = f"def __call__(\n        {joined},\n    ){ret}"
    lines.append(f"    {signature}: ...")
    return "\n".join(lines)


_ASF_HEADER = """\
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
# under the License.\
"""


def generate() -> str:
    out = [
        _ASF_HEADER,
        '"""Generated stub for T.ptx — do not edit.',
        "",
        "Regenerate:",
        "  python -m tvm.backend.cuda.ptx.gen_stubs -o python/tvm/script/tirx.pyi",
        '"""',
        "",
        "from typing import Any",
        "",
    ]
    families: dict[str, list[InstructionEntry]] = {}
    for entry in TABLE.values():
        families.setdefault(entry.family, []).append(entry)
    for family in sorted(families):
        out.append(_chain_class(family, families[family]))
        out.append("")
    out.append("class _PTX:")
    for family in sorted(families):
        out.append(f"    {escape_token(family)}: _Chain_{family}")
    out.append("    def addr(self, base: Any, byte_offset: Any) -> Any: ...")
    out.append("    def __getitem__(self, text: str) -> Any: ...")
    out.append("")
    out.append("ptx: _PTX")
    out.append("")
    out.append("# Every other tvm.script.tirx member stays dynamically typed, as before.")
    out.append("def __getattr__(name: str) -> Any: ...")
    # No formatter pass: every line above is emitted in the shape ruff format
    # produces, so this is byte-stable on a machine that has no ruff at all.
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default=None, help="output path (default: stdout)")
    args = parser.parse_args()
    text = generate()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
