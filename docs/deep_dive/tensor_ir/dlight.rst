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

.. _dlight_gpu_scheduling:

DLight: Rule-Based GPU Scheduling
==================================

TIR functions produced by Relax legalization need GPU-specific scheduling --
thread binding, loop tiling, shared memory usage -- before they can run
efficiently on a GPU. TVM provides two approaches:

- **MetaSchedule** explores a search space to find the best schedule. High
  quality, but compilation takes minutes to hours.
- **DLight** applies pre-defined scheduling rules deterministically. No tuning
  required, compilation completes in seconds. Performance is excellent for
  well-known patterns (e.g., GEMM, GEMV in LLM workloads) and fair for the
  rest.

This page explains how DLight works internally, what rules are available, how
to diagnose scheduling quality, and how to extend it with custom rules.

.. contents:: Table of Contents
    :local:
    :depth: 1


How ApplyDefaultSchedule Works
------------------------------

``ApplyDefaultSchedule`` is an ``IRModule`` pass. Given a list of rules, it:

1. Iterates over every ``PrimFunc`` in the module.
2. Skips functions already marked ``tirx.is_scheduled``.
3. For each unscheduled function, tries rules **in order** -- the first rule
   whose ``apply()`` returns a non-``None`` schedule wins; subsequent rules are
   skipped.
4. Marks the scheduled function with ``tirx.is_scheduled`` to prevent
   re-scheduling by a later pass.

.. code-block:: python

    from tvm.s_tir import dlight as dl

    with target:
        mod = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(),           # GEMM: dense matrix multiplication
            dl.gpu.GEMV(),             # matrix-vector products
            dl.gpu.Reduction(),        # simple reductions (sum, max, ...)
            dl.gpu.GeneralReduction(), # compound reductions (softmax, layer norm, ...)
            dl.gpu.Fallback(),         # catch-all for anything unmatched above
        )(mod)

After scheduling, TIR functions will contain thread bindings
(``blockIdx.x``, ``threadIdx.x``, ...) and loop transformations.


Built-in GPU Rules
------------------

Each rule is a subclass of ``ScheduleRule`` and implements
``apply(func, target, tunable)`` -- returns a ``Schedule`` on match, or
``None`` to pass.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Rule
     - Pattern
     - Typical operators
   * - ``Matmul``
     - GEMM index pattern ``C[S,I,J] += A[S,I,K] * B[S,J,K]``
     - ``nn.Linear``, batched matmul
   * - ``GEMV``
     - Matrix-vector multiply (one dimension is 1)
     - single-batch decode in attention
   * - ``LowBatchGEMV``
     - Low-batch GEMM scheduled with a GEMV strategy
     - small-batch decode
   * - ``Reduction``
     - Simple accumulation ``X[...] += Y[...]``
     - sum, max, argmax
   * - ``GeneralReduction``
     - Spatial dims followed by reduction dims (``S* R*``)
     - softmax, layer norm, RMS norm
   * - ``Transpose``
     - Read/write indices are permutations of each other
     - 2-D transpose
   * - ``RMSNorm``
     - Contains an ``rsqrt`` operation
     - RMS normalization
   * - ``Fallback``
     - Any function (always matches)
     - generic catch-all

Why Rule Order Matters
~~~~~~~~~~~~~~~~~~~~~~

``ApplyDefaultSchedule`` stops at the first match, so:

- Put **specialized** rules first (``Matmul``, ``GEMV``) -- they have strict
  matching conditions but produce high-quality schedules.
- Put **general** rules later (``GeneralReduction``, ``Fallback``) -- they
  match broadly but with less optimal schedules.
- If you put ``Fallback`` first, it would claim every function and no
  specialized rule would ever run.


Diagnosing Schedule Quality
---------------------------

``ApplyDefaultSchedule`` does not log which rule handled which function.
You can figure it out by applying rules one at a time:

**Step 1** -- Apply each rule individually and record which functions it claims:

.. code-block:: python

    from collections import OrderedDict
    from tvm.s_tir import dlight as dl

    rules = OrderedDict([
        ("Matmul",          dl.gpu.Matmul()),
        ("GEMV",            dl.gpu.GEMV()),
        ("LowBatchGEMV",   dl.gpu.LowBatchGEMV()),
        ("Reduction",       dl.gpu.Reduction()),
        ("GeneralReduction", dl.gpu.GeneralReduction()),
        ("Transpose",       dl.gpu.Transpose()),
        ("RMSNorm",         dl.gpu.RMSNorm()),
    ])

    rule_assignment = {}
    for rule_name, rule in rules.items():
        with target:
            test_mod = dl.ApplyDefaultSchedule(rule)(mod)
        for gv, func in test_mod.functions_items():
            if isinstance(func, tirx.PrimFunc) and gv.name_hint not in rule_assignment:
                if "tirx.is_scheduled" in func.attrs and func.attrs["tirx.is_scheduled"] == 1:
                    rule_assignment[gv.name_hint] = rule_name

**Step 2** -- Functions not claimed by any specialized rule will fall through
to ``Fallback``:

.. code-block:: python

    all_tir_funcs = [
        gv.name_hint
        for gv, func in mod.functions_items()
        if isinstance(func, tirx.PrimFunc)
    ]
    fallback_funcs = [name for name in all_tir_funcs if name not in rule_assignment]

If an important kernel lands in the Fallback bucket, you have three options:

1. Write a **custom DLight rule** (see below).
2. Use **MetaSchedule** to auto-tune that specific function.
3. Manually schedule it with the ``tvm.s_tir.Schedule`` API.


DLight vs MetaSchedule
----------------------

The two systems are complementary, not competing:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - DLight
     - MetaSchedule
   * - Mechanism
     - Deterministic rule matching
     - Search-space exploration
   * - Compile time
     - Seconds
     - Minutes to hours
   * - Performance
     - Excellent on known patterns, fair otherwise
     - Near-optimal with sufficient search budget
   * - Best for
     - Default path, rapid iteration, CI
     - Hot-spot tuning in production

A practical workflow:

1. Run ``ApplyDefaultSchedule`` with the full rule set to cover all functions.
2. Profile the compiled model to identify hot-spot kernels.
3. Use ``MetaScheduleTuneTIR`` to auto-tune only those kernels.

.. note::

   ``MetaScheduleTuneTIR`` does **not** automatically skip functions already
   scheduled by DLight -- it processes every ``PrimFunc`` in the module. To
   avoid the extra search cost, filter the module or use
   ``MetaScheduleTuneIRMod`` with ``op_names`` to target specific functions.


Writing a Custom Rule
---------------------

You can extend DLight by writing your own ``ScheduleRule``. The simplest way
is ``ScheduleRule.from_callable``, which wraps a plain function into a rule
**instance**:

.. code-block:: python

    from tvm import s_tir, tirx
    from tvm.s_tir.dlight.analysis import normalize_prim_func
    from tvm.s_tir.dlight.base.schedule_rule import ScheduleRule

    @ScheduleRule.from_callable("MyTileAndBind")
    def my_tile_and_bind(func, target, tunable):
        """For single-block injective functions, tile and bind to GPU threads."""
        if not isinstance(func, tirx.PrimFunc):
            return None
        sch = s_tir.Schedule(func)
        # normalize_prim_func returns SBlockInfo with correct spatial/reduction
        # classification -- the same analysis used by built-in DLight rules.
        block_infos = normalize_prim_func(sch)
        if block_infos is None or len(block_infos) != 1:
            return None
        info = block_infos[0]
        if not info.is_injective():
            return None  # dom_kind() uses iter_type, not loop kind
        loops = sch.get_loops(info.block_rv)
        if len(loops) == 0:
            return None
        fused = sch.fuse(*loops)
        bx, tx = sch.split(fused, factors=[None, 256])
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")
        return sch

Insert the custom rule into the rule chain. Note that ``from_callable``
returns an **instance** -- pass it directly, do not call
``my_tile_and_bind()`` again:

.. code-block:: python

    with target:
        mod = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(),
            dl.gpu.GeneralReduction(),
            my_tile_and_bind,   # custom rule, tried before Fallback
            dl.gpu.Fallback(),
        )(mod)

For production-quality rules, subclass ``ScheduleRule`` directly and implement
``apply()`` with full analysis logic. See ``tvm.s_tir.dlight.gpu.Matmul`` for
a reference implementation.


Summary
-------

- **DLight** provides fast, deterministic GPU scheduling via rule matching.
- Rules are tried in order; the first match wins. Put specialized rules before
  general ones.
- Use the **single-rule probing** technique to diagnose which rule handles
  each function.
- Combine DLight with MetaSchedule: DLight for baseline coverage, MetaSchedule
  for hot-spot tuning.
- Extend DLight by writing custom ``ScheduleRule`` implementations.

For DLight's role in the broader optimization pipeline, see
:ref:`customize_opt`.
