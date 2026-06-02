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

"""Packed-vector emit functions for elementwise ops.

Each module here exposes one or more ``VecImpl`` instances that an ``OpSpec``
(in ``ops/``) attaches to its ``vec_impls`` list. ``reg.py``/``smem.py`` then
pick the widest matching one at dispatch time, mirroring how copy picks
``copy_{Nb}`` from a menu.

VecImpl emit contract:
  emit(dst_buf, dst_lane_indices, src_args, extras) -> PrimExpr

  * dst_buf: Buffer
  * dst_lane_indices: list[list[Expr]] of length ``vec_len``; each entry is the
                     multi-dim indices for one lane (precomputed by schedule).
  * src_args[i]: one of
      - PrimExpr (scalar src — broadcast across all lanes)
      - tuple (Buffer, list[list[Expr]] of length ``vec_len``) — buffer src
                                                                with per-lane indices
  * extras: dict (rounding_mode, etc.)

  Returns the PTX/CUDA call result; the schedule wraps in ``Tx.evaluate`` at
  the call site. All Python-side shape branching (scalar vs buffer src) happens
  in this emit function -- collapses the old 4x2 schema.py factory explosion.
"""
