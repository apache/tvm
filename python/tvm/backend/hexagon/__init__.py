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
"""Hexagon-owned backend hooks."""

from importlib import import_module

_LAZY_SUBMODULES = {"target_tags"}


def register_backend():
    """Register Hexagon-owned Python semantics."""
    from tvm.backend.loader import _load_runtime_sidecar

    _load_runtime_sidecar("hexagon")
    import_module(f"{__name__}.target_tags")


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["register_backend", "target_tags"]
