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

"""Warp atom: two m8n8 b16 matrices, carried as b32 pairs of columns.

Canonical coordinates: (*repeat, matrix=2, row=8, word=4).
Source: each (matrix, row) names a 16B-aligned contiguous row of four b32
words. Different rows may have arbitrary offsets expressible by TileLayout.
Destination: owner lane = 4*row + word; two matrix words per lane, placed
directly in the caller's register slots. A word holds two adjacent b16 values.

Address providers differ from result owners: lanes 0..7 provide matrix 0
rows, lanes 8..15 matrix 1 rows. Lanes 16..31 repeat these valid addresses.
"""

from tvm.script import tirx as T

SHAPE = (2, 8, 4)
SCOPE = "warp"
SRC_SCOPE = ("shared", "shared.dyn")


def check(operands, sctx):
    active = sctx.intra.get("laneid")
    if active is None or int(active[0]) != 32 or int(active[1]) != 0:
        raise ValueError("ldmatrix requires all 32 lanes of a warp to be active")
    for matrix in range(2):
        for row in range(8):
            base = operands.aligned_row((matrix, row, 0))
            for word in range(4):
                coord = (matrix, row, word)
                operands.require(
                    operands.memory("src", coord) == base + word,
                    "src: ldmatrix row needs four consecutive b32 words in canonical order",
                )
                operands.memory("dst", coord, lane=4 * row + word)


def emit(operands, instruction):
    lane = T.cuda.lane_id()
    return instruction(
        operands.value("dst", (0, lane // 4, lane % 4)),
        operands.value("dst", (1, lane // 4, lane % 4)),
        operands.pointer("src", ((lane // 8) % 2, lane % 8, 0)),
    )
