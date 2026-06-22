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

Control flow
============

Control flow is ``if``, the loop family, and ``while`` — each maps to the obvious
CUDA.

if
--

A Python ``if`` / ``else`` becomes a CUDA ``if`` / ``else``. Guard work by a
thread/lane comparison, or elect a single issuing thread with
``T.ptx.elect_sync()``:

.. code-block:: python

    if tx < 128:
        A[tx] = A[tx] * T.float32(2.0)
    else:
        A[tx] = A[tx] + T.float32(1.0)

    if T.ptx.elect_sync():
        ...                              # one elected lane (e.g. to issue TMA/MMA)

.. code-block:: c++

    if (((int)threadIdx.x) < 128) {
      A_ptr[tx] = A_ptr[tx] * 2.0f;
    } else {
      A_ptr[tx] = A_ptr[tx] + 1.0f;
    }

For an expression-level choice (no branch), use ``T.if_then_else(cond, a, b)``.

loop
----

Loops come in four flavors; a plain Python ``range`` becomes ``T.serial``:

- ``T.serial(n)`` — a sequential loop (ptxas may still unroll it).
- ``T.unroll(n)`` — fully unrolled (expanded to straight-line statements).
- ``T.vectorized(n)`` — a vectorized loop.
- ``T.grid(*extents)`` — a nested loop nest.

``break`` / ``continue`` work inside loops.

.. code-block:: python

    for i, j in T.grid(8, 8):
        B[i, j] = T.max(A[i, j], T.float32(0.0))

.. code-block:: c++

    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j)
        B_ptr[i * 8 + j] = max(A_ptr[i * 8 + j], 0.0f);

``T.unroll(4)`` instead expands to four straight-line statements with no loop.

while
-----

A ``while`` loop runs until its condition is false. Use a mutable scalar counter
(see :doc:`buffers`):

.. code-block:: python

    i: T.int32 = 0
    while i < 64:
        A[i] = A[i] + T.float32(1.0)
        i += 1

It lowers to a ``while (1)`` with an early-exit ``break`` (the counter is a
one-element register buffer):

.. code-block:: c++

    int i_ptr[1];
    i_ptr[0] = 0;
    while (1) {
      if (!(i_ptr[0] < 64)) { break; }
      A_ptr[i_ptr[0]] = A_ptr[i_ptr[0]] + 1.0f;
      i_ptr[0] = i_ptr[0] + 1;
    }
