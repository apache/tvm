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
"""Thin generator 3: dump the generated inline-CUDA helper of every instruction.

The engine renders each ``T.ptxd`` call's ``__device__`` helper on the fly at
codegen time — this tool renders the exact same helpers ahead of time so a
human can inspect what each registered instruction variant compiles to,
without building a kernel::

    python -m tvm.backend.cuda.ptx_dialect.gen_helpers            # everything
    python -m tvm.backend.cuda.ptx_dialect.gen_helpers ld st      # some families
    python -m tvm.backend.cuda.ptx_dialect.gen_helpers -o ptxd_helpers.cu

Only imports :mod:`.table` / :mod:`.render` (tvm-free).
"""

import argparse
import sys

from .render import render_variant
from .table import TABLE, renderings


def generate(families=None) -> str:
    unknown = set(families or ()) - set(TABLE)
    if unknown:
        raise ValueError(
            f"unknown instruction family: {', '.join(sorted(unknown))}; "
            f"known: {', '.join(sorted(TABLE))}"
        )
    chunks = []
    for name in sorted(TABLE):
        if families and name not in families:
            continue
        entry = TABLE[name]
        # `renderings` is the one place the axes are multiplied, so this dump
        # covers every helper the engine can emit -- dtype choices included.
        rendered = list(renderings(entry))
        chunks.append(f"// ========== {entry.name} — {len(rendered)} helper(s) ==========\n")
        for tokens, dtypes, predicated, imms in rendered:
            opcode, _, source = render_variant(entry, tokens, predicated, dtypes, imms)
            chunks.append(f"// {'@p ' if predicated else ''}{opcode}")
            chunks.append(source)
    return "\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tvm.backend.cuda.ptx_dialect.gen_helpers",
        description="Dump the generated inline-CUDA helper of every ptxd instruction variant.",
    )
    parser.add_argument("families", nargs="*", help="restrict to these families (default: all)")
    parser.add_argument("-o", "--output", default=None, help="output path (default: stdout)")
    args = parser.parse_args()
    try:
        text = generate(args.families or None)
    except ValueError as err:
        parser.error(str(err))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {len(text)} chars to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
