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
"""Codegen registry for CUDA HW ops.

User-facing Python wrappers are hand-written in :mod:`tvm.tirx.op` so that
editors / static analyzers (Cursor, Pyright) can see their signatures. This
module only handles the backend codegen side.
"""

import functools

import tvm_ffi

CODEGEN_REGISTRY = {}
CODEGEN_MANIFEST = {}


def _canonical_device_intrin_name(op_name: str) -> str:
    if not op_name.startswith("tirx."):
        return op_name
    basename = op_name[len("tirx.") :]
    if "." in basename:
        return op_name
    for prefix, namespace in (
        ("cuda_", "cuda"),
        ("ptx_", "ptx"),
        ("nvshmem_", "nvshmem"),
        ("nki_", "nki"),
    ):
        if basename.startswith(prefix):
            return f"tirx.{namespace}.{basename[len(prefix) :]}"
    return op_name


def _codegen_namespace(op_name: str) -> str:
    basename = op_name[len("tirx.") :] if op_name.startswith("tirx.") else op_name
    if basename.startswith("_"):
        return "internal"
    if "." in basename:
        return basename.split(".", 1)[0]
    if "_" in basename:
        return basename.split("_", 1)[0]
    return basename


def _record_codegen(canonical_op_name, op_names, backend):
    CODEGEN_MANIFEST[canonical_op_name] = {
        "aliases": tuple(sorted(op_names - {canonical_op_name})),
        "backend": backend,
        "namespace": _codegen_namespace(canonical_op_name),
    }


@tvm_ffi.register_global_func("tirx.intrinsics.cuda.get_codegen")
def get_codegen(op):
    """get the codegen function for a given op"""
    return CODEGEN_REGISTRY.get(op, None)


def list_registered_codegen(include_internal=False):
    """Return registered codegen ops grouped by namespace.

    The result is intended for inspection and tests.  The C++ codegen still
    dispatches through ``get_codegen`` / ``CODEGEN_REGISTRY``.
    """

    result = {}
    for op_name, metadata in sorted(CODEGEN_MANIFEST.items()):
        if not include_internal and metadata["namespace"] == "internal":
            continue
        entry = {
            "op": op_name,
            "aliases": metadata["aliases"],
            "backend": metadata["backend"],
        }
        result.setdefault(metadata["namespace"], []).append(entry)
    return {namespace: tuple(entries) for namespace, entries in sorted(result.items())}


def register_codegen(op, backend="cuda"):
    """Register a codegen function for a given op.

    The codegen function should return a ``cuda_func_call`` statement, and
    optionally a list of tags that the codegen function needs.
    """

    def decorator(func):
        full_op_name = "tirx." + op
        canonical_op_name = _canonical_device_intrin_name(full_op_name)
        op_names = {full_op_name, canonical_op_name}

        @functools.wraps(func)
        def wrapper(arg_list):
            res = func(*arg_list)  # pylint: disable=not-callable
            if isinstance(res, tuple):
                return res[0], res[1]
            return res, list()

        for op_name in op_names:
            CODEGEN_REGISTRY[op_name] = wrapper
        _record_codegen(canonical_op_name, op_names, backend)
        return wrapper

    return decorator
