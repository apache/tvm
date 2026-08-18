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
"""Thin generator 2: emit a markdown coverage table for the ptx dialect.

Run manually::

    python -m tvm.backend.cuda.ptx.gen_coverage

Only imports :mod:`.table` (tvm-free).
"""

import sys

from .table import TABLE


def _operand_tag(slot) -> str:
    """How one operand reads in the coverage table: its kind, or its direction
    when it is an ordinary register (where the direction is the interesting
    half and ``reg`` says nothing)."""
    return slot.rw if slot.kind == "reg" else slot.kind


def generate() -> str:
    lines = [
        "# ptx dialect coverage",
        "",
        "| instruction | modifiers | constraints | operands |",
        "|---|---|---|---|",
    ]
    for name in sorted(TABLE):
        e = TABLE[name]
        mods = (
            "<br>".join(
                f"`{s.name}` ∈ {{{', '.join(s.choices)}}}{' *(opt)*' if s.optional else ''}"
                for s in e.slots
            )
            or "—"
        )
        check_doc = (e.check.__doc__ or "").strip() if e.check is not None else "—"
        # `lanes` may be a function of the modifiers, which has no single number
        # to print here; the group marker only reports a fixed width.
        operands = ", ".join(
            f"{s.name}:{_operand_tag(s)}[{s.lanes}]"
            if not callable(s.lanes) and s.lanes > 1
            else f"{s.name}:{_operand_tag(s)}"
            for s in e.operands
        )
        lines.append(f"| `{name}` | {mods} | {check_doc} | {operands} |")
    lines.append("")
    families = {e.family for e in TABLE.values()}
    lines.append(f"{len(TABLE)} entries across {len(families)} instruction families.")
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.write(generate())


if __name__ == "__main__":
    main()
