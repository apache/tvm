# isort: skip_file
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
"""Trainium-specific TIRX transformations."""
# pylint: disable=invalid-name

# Fork-only TIRX-specific passes. They decorate their pass body with
# `@prim_func_pass(...)` at module-load time, which triggers an FFI call to
# construct PassInfo -- not runtime-safe. Loading them lazily preserves
# apache's discipline that `import tvm.tirx.transform.trn` performs no
# compiler-side FFI calls (required for `TVM_USE_RUNTIME_LIB=1`).
_LAZY_TRANSFORMS = {
    "TrnNaiveAllocator": ".naive_allocator",
    "TrnPrivateBufferAlloc": ".private_buffer_alloc",
}


def __getattr__(name):
    target = _LAZY_TRANSFORMS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module  # pylint: disable=import-outside-toplevel

    return getattr(import_module(target, __name__), name)
