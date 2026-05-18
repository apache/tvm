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

"""Unified elementwise dispatch for CUDA.

Three schedules cover all elementwise ops (unary / binary / cast / fma):

  per_thread:       scope == thread; one thread runs vectorized serial loop
  tile_local:       scope > thread; local buffer with layout describing
                    thread->element mapping; threads cooperatively cover the
                    tile via per-thread views (buf.local(*shape))
  shared_distributed: scope > thread; shared buffer; fused-tid distribution
                      with scope-level barrier at the end

Phase 1 covers unary ops. Binary / cast / fma to follow.
"""

from .register import *
