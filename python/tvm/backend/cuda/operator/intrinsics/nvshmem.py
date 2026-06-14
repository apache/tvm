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
# pylint: disable=redefined-builtin, invalid-name
"""NVSHMEM intrinsics. Each backend call is one ``device_intrinsic(...)``."""

from ._schema import device_intrinsic
from .registry import CODEGEN_REGISTRY, register_codegen

_NVSHMEM = ("nvshmem",)

# =============================================================================
# No-arg helpers: PE queries, quiet, fence, barrier_all.
# =============================================================================
for _op, _call, _ret, _tvm_ret in [
    ("nvshmem_my_pe", "nvshmem_my_pe", "int32_t", "int32"),
    ("nvshmem_n_pes", "nvshmem_n_pes", "int32_t", "int32"),
    ("nvshmem_quiet", "nvshmem_quiet", "void", None),
    ("nvshmem_fence", "nvshmem_fence", "void", None),
    ("nvshmem_barrier_all", "nvshmem_barrier_all", "void", None),
]:
    device_intrinsic(
        _op,
        body=("    " + (f"return {_call}();" if _ret != "void" else f"{_call}();")),
        return_type=_ret,
        tvm_return_type=_tvm_ret,
        extra_deps=_NVSHMEM,
    )
del _op, _call, _ret, _tvm_ret


# =============================================================================
# RMA get/put (thread/warp/block).
# =============================================================================
_RMA_SIG = "(void *dest, const void *source, size_t nelems, int pe)"
for _op, _backend_call in [
    ("nvshmem_getmem_nbi", "nvshmem_getmem_nbi"),
    ("nvshmem_putmem_nbi", "nvshmem_putmem_nbi"),
    ("nvshmem_getmem_nbi_warp", "nvshmemx_getmem_nbi_warp"),
    ("nvshmem_putmem_nbi_warp", "nvshmemx_putmem_nbi_warp"),
    ("nvshmem_getmem_nbi_block", "nvshmemx_getmem_nbi_block"),
    ("nvshmem_putmem_nbi_block", "nvshmemx_putmem_nbi_block"),
]:
    device_intrinsic(
        _op,
        c_signature=_RMA_SIG,
        body=f"    {_backend_call}(dest, source, nelems, pe);",
        extra_deps=_NVSHMEM,
    )
del _op, _backend_call


# =============================================================================
# Signal / wait_until — each backend call is one device_intrinsic. String
# attrs (sig_op, cmp) are mapped to NVSHMEM integer constants in the
# user-facing dispatcher below.
# =============================================================================

_SIG_OP_VAL = {"set": 0, "add": 1}
_CMP_VAL = {"eq": 0, "ne": 1, "gt": 2, "ge": 3, "lt": 4, "le": 5}


def _resolve_attr(value, table, label):
    s = value if isinstance(value, str) else value.value
    if s not in table:
        raise ValueError(f"Unsupported {label}: {s}")
    return table[s]


device_intrinsic(
    "_nvshmem_signal_op_impl",
    helper_name="tvm_builtin_nvshmem_signal_op",
    c_signature="(uint64_t* sig_addr, uint64_t signal, int sig_op, int pe)",
    body="    nvshmemx_signal_op(sig_addr, signal, sig_op, pe);",
    extra_deps=_NVSHMEM,
)


@register_codegen("nvshmem_signal_op")
def codegen_nvshmem_signal_op(sig_addr, signal, sig_op, pe):
    """Map ``sig_op`` (string) to its NVSHMEM int constant, then forward."""
    sig_op_int = _resolve_attr(sig_op, _SIG_OP_VAL, "signal op")
    result = CODEGEN_REGISTRY["tirx._nvshmem_signal_op_impl"]([sig_addr, signal, sig_op_int, pe])
    return result


# nvshmem_<type>_wait_until — one device_intrinsic per supported type.
_WAIT_UNTIL_TYPES = {"uint64_t": "uint64", "uint64": "uint64"}

for _c_type, _suffix in [("uint64_t", "uint64")]:
    device_intrinsic(
        f"_nvshmem_{_suffix}_wait_until_impl",
        helper_name=f"tvm_builtin_nvshmem_{_suffix}_wait_until",
        c_signature=f"({_c_type}* ivar, int cmp, {_c_type} cmp_value)",
        body=f"    nvshmem_{_suffix}_wait_until(ivar, cmp, cmp_value);",
        extra_deps=_NVSHMEM,
    )
del _c_type, _suffix


@register_codegen("nvshmem_wait_until")
def codegen_nvshmem_wait_until(ivar, cmp, cmp_value, type):
    """Dispatch to the type-specific wait_until helper after mapping ``cmp``
    (string) to its NVSHMEM int constant."""
    type_str = type if isinstance(type, str) else type.value
    if type_str not in _WAIT_UNTIL_TYPES:
        raise ValueError(f"Unsupported type for nvshmem_wait_until: {type_str}")
    suffix = _WAIT_UNTIL_TYPES[type_str]
    cmp_int = _resolve_attr(cmp, _CMP_VAL, "cmp operation")
    result = CODEGEN_REGISTRY[f"tirx._nvshmem_{suffix}_wait_until_impl"]([ivar, cmp_int, cmp_value])
    return result


# putmem_signal_nbi (thread / warp / block) — three scope-specific helpers.
_PUTMEM_SIG_SIG = (
    "(void* dest, const void* source, size_t nelems, "
    "uint64_t* sig_addr, uint64_t signal, int sig_op, int pe)"
)
for _scope_suffix, _backend_call in [
    ("", "nvshmem_putmem_signal_nbi"),
    ("_warp", "nvshmemx_putmem_signal_nbi_warp"),
    ("_block", "nvshmemx_putmem_signal_nbi_block"),
]:
    device_intrinsic(
        f"_nvshmem_putmem_signal_nbi{_scope_suffix}_impl",
        helper_name=f"tvm_builtin_nvshmem_putmem_signal_nbi{_scope_suffix}",
        c_signature=_PUTMEM_SIG_SIG,
        body=f"    {_backend_call}(dest, source, nelems, sig_addr, signal, sig_op, pe);",
        extra_deps=_NVSHMEM,
    )
del _scope_suffix, _backend_call


def _make_putmem_signal_dispatcher(scope_suffix):
    @register_codegen(f"nvshmem_putmem_signal_nbi{scope_suffix}")
    def _codegen(dest, source, nelems, sig_addr, signal, sig_op, pe):
        sig_op_int = _resolve_attr(sig_op, _SIG_OP_VAL, "signal op")
        result = CODEGEN_REGISTRY[f"tirx._nvshmem_putmem_signal_nbi{scope_suffix}_impl"](
            [dest, source, nelems, sig_addr, signal, sig_op_int, pe]
        )
        return result

    return _codegen


for _suffix in ("", "_warp", "_block"):
    _make_putmem_signal_dispatcher(_suffix)
del _suffix
