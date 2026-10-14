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
"""Trainium-specific TIRx layout helpers."""

from __future__ import annotations

from tvm.tirx.layout import Layout, TileLayout

_TRN_MEMORY_AXES = {"F", "P", "Bank"}


def is_trainium_layout(layout: Layout | None) -> bool:
    """Return whether a layout uses only Trainium memory axes."""
    if not isinstance(layout, TileLayout):
        return False
    return not any(
        iter.axis.is_memory() and iter.axis.name not in _TRN_MEMORY_AXES for iter in layout.shard
    )


__all__ = ["is_trainium_layout"]
