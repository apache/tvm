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

"""Thread atom: four contiguous b32 global words -> four independent registers.

Canonical coordinates: (*repeat, word), word in [0, 4).
The source address is base(repeat) + word, with each base aligned to 16B.
All four destination words belong to the executing thread. Their physical
register slots may be separated, and need not have the same order as memory.
"""

SHAPE = (4,)
SCOPE = "thread"
SRC_SCOPE = ("global",)


def check(operands, _sctx):
    base = operands.aligned_row((0,))
    for word in range(4):
        operands.require(
            operands.memory("src", (word,)) == base + word,
            "src: ld.global.v4.b32 needs four consecutive words in canonical order",
        )
        operands.memory("dst", (word,))


def emit(operands, instruction):
    return instruction(
        *[operands.value("dst", (word,)) for word in range(4)], operands.pointer("src", (0,))
    )
