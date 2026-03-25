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

.. _security-advisory-subroutine-hash-collision:

Security Advisory: Subroutine Cache Hash Collision
===================================================

Summary
-------

``SubroutineMixin._get_subroutine()`` in ``python/tvm/relax/frontend/nn/subroutine.py``
used ``ir.structural_hash`` as the sole cache lookup key without a subsequent
``structural_equal`` verification. If two different ``arg_sinfo`` values produced the
same 64-bit hash, the cache would return a previously compiled function with
mismatched parameter shapes, leading to silently incorrect compiled output.

Severity
--------

**Low.** The ``structural_hash`` function returns a 64-bit integer. A natural hash
collision requires approximately 2^32 distinct inputs (birthday bound), making
accidental collision extremely unlikely in normal compilation workflows. The issue
is primarily a **correctness defect** rather than a practically exploitable security
vulnerability.

Affected Code
-------------

- **File**: ``python/tvm/relax/frontend/nn/subroutine.py``
- **Method**: ``SubroutineMixin._get_subroutine()``
- **Trigger condition**: ``define_subroutine = True`` on an ``nn.Module`` subclass,
  with two or more calls using different input shapes within the same compilation session.

Root Cause
----------

The subroutine cache (``cls._gvar``) was keyed by
``(structural_hash(arg_sinfo, map_free_vars=True), is_dataflow)``.
A hash match was treated as proof of structural equality, skipping the necessary
``structural_equal`` check. This is inconsistent with the pattern used elsewhere in
TVM (e.g., ``block_builder.cc`` uses ``StructuralHash`` + ``StructuralEqual`` together
in ``std::unordered_map``).

Impact
------

If a collision occurred:

1. The cache returned a ``GlobalVar`` bound to a function compiled for a different
   input shape.
2. The caller would invoke this wrong function with mismatched arguments.
3. The compiled Relax IR module would contain an incorrect function call.
4. At inference time, the model would produce wrong numerical results **without
   any error or warning**.

Fix
---

The cache now stores a list of ``(arg_sinfo, result)`` pairs per hash bucket.
On lookup, each candidate is verified with ``structural_equal`` before returning.
This follows the standard hash-table pattern: hash for bucket selection, equality
for final verification.

Recommendations
---------------

- Update to the patched version of TVM.
- If you maintain custom code that caches TVM IR nodes by ``structural_hash``,
  ensure that a ``structural_equal`` check is always performed on cache hits.
