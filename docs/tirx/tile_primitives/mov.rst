..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

mov
===

``mov`` copies register bits or assembles/disassembles a native register
group. It performs no numeric conversion and cannot change which thread
owns a value. Use :doc:`cvt` for conversion and :doc:`ld_st` for memory.

The scalar form ends both tensor operands in ``(1,)``. All earlier axes
after the execution-group prefix are repeats. A scalar source expression
can broadcast, for example to fill a register tile. Vector-group mov forms
instead use the group length declared by that PTX form; each input/output
word remains an explicit operand slot.

Example: filling and copying separated register slots
-----------------------------------------------------

.. code-block:: python

    from tvm.script import tirx as T
    from tvm.tirx.layout import S, TileLayout

    gapped = TileLayout(S[(4, 1) : (2, 1)])


    @T.function
    def register_move(Out: T.Tensor((4, 1), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        a = T.alloc_local((4, 1), "uint32", layout="default")
        b = T.alloc_local((4, 1), "uint32", layout=gapped)
        T.cuda.tile.apply(T.ptx["mov.b32"], a, T.uint32(17))
        T.cuda.tile.apply(T.ptx["mov.b32"], b, a)
        T.cuda.tile.apply(T.ptx["st.global.b32"], Out, b)

The first call emits four moves of the same constant. The second emits four
moves to offsets 0, 2, 4, and 6 of ``b``; its gaps require no extra remap
buffer. All four output values are 17. Ordinary compiler optimizations may
subsequently eliminate redundant moves; the tile expansion itself selects
no alternative instruction or ownership map.

``pred`` is an ordinary runtime predicate. For a predicated output that must
keep its old value, initialize it and pass ``preserve_dst=True``. A suppressed
move is not a zero fill. Destination repeats must not overlap each other.

For packed arithmetic, a bit reinterpretation such as
``halves.view("uint32")`` can expose the existing low/high halfword storage
without a mov. Use an explicit group mov when assembling separately stored
values. Group arity and carrier dtype must agree with the selected native
form; the pattern does not guess a packing arrangement.

Implementation: ``python/tvm/backend/cuda/tile/patterns/register.py``.
