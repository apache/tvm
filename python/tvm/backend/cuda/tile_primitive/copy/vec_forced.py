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

"""Explicit fixed-width vector copy dispatches.

Cache-semantics config (all variants, read from ``op_call.config``):

- ``cache``: ``None`` (default, plain ``T.ptxd.ld``) or ``"nc"`` (load via
  ``T.ptxd.ld.global_.nc``, i.e. PTX ``ld.global.nc``). Requires a global src.
- ``l1_evict`` / ``l2_evict`` / ``prefetch_size``: L1/L2 cache hint kwargs
  forwarded verbatim to the emitted load (same values ``T.ptxd.ld`` /
  ``T.ptxd.ld.global_.nc`` accept, e.g. ``"L1::no_allocate"``,
  ``"L2::evict_first"``, ``"L2::256B"``). Requires a global src.

Example::

    Tx.copy(dst_local[0:8], src_global[base : base + 8], dispatch="vec_256b",
            cache="nc", l1_evict="L1::no_allocate", prefetch_size="L2::256B")
"""

from tvm.arith.analyzer import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as T
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.operator.tile_primitive.dispatcher import predicate, register_dispatch
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.stmt import BufferRegion
from tvm.tirx.tile_primitive import TilePrimitiveCall

from ._common import copy_ptxd_form, copy_ptxd_ld_chain
from .utils import _scope_allowed


def _region_start(buffer_region: BufferRegion):
    return [r.min for r in buffer_region.region]


def _region_elements(buffer_region: BufferRegion):
    product = 1
    for r in buffer_region.region:
        product *= r.extent
    return product


def _can_prove_equal(lhs, rhs) -> bool:
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs == rhs
    return Analyzer().can_prove_equal(lhs, rhs)


def _ptx_space(scope: str) -> str:
    if scope.startswith("shared"):
        return "shared"
    return scope


_LD_CACHE_HINT_KEYS = ("l1_evict", "l2_evict", "prefetch_size")


def _ld_cache_config(op_call: TilePrimitiveCall) -> tuple[str | None, dict[str, str]]:
    """Read the cache-semantics config from ``op_call.config``.

    Returns ``(cache, hints)``: ``cache`` is ``None`` or ``"nc"``, and
    ``hints`` maps the ``T.ptxd.ld``/``T.ptxd.ld.global_.nc`` L1/L2 hint kwargs
    (``l1_evict``, ``l2_evict``, ``prefetch_size``) to their string values.
    """
    cache = op_call.config.get("cache", None)
    if cache is not None:
        cache = str(cache)
    hints: dict[str, str] = {}
    for key in _LD_CACHE_HINT_KEYS:
        value = op_call.config.get(key, None)
        if value is not None and str(value):
            hints[key] = str(value)
    return cache, hints


def _is_forced_vec_copy(
    op_call: TilePrimitiveCall,
    sctx: DispatchContext,
    *,
    variant: str,
    num_bytes: int,
):
    if getattr(op_call, "dispatch", None) != variant:
        return False, f"requires explicit dispatch={variant!r}"
    if sctx.scope_kind != "thread":
        return False, f"expected thread exec_scope, got {sctx.scope_kind}"

    scope_ok, scope_reason = _scope_allowed(op_call, sctx)
    if not scope_ok:
        return False, scope_reason

    op_call = TilePrimitiveCall.downcast(op_call)
    src: Buffer = op_call.src.buffer
    dst: Buffer = op_call.dst.buffer
    if src.dtype != dst.dtype:
        return False, f"dtype mismatch: src={src.dtype}, dst={dst.dtype}"

    elem_bits = DataType(src.dtype).bits
    width_bits = num_bytes * 8
    if width_bits % elem_bits != 0:
        return False, f"{variant} is not an integral number of {src.dtype} elements"
    expected_elements = width_bits // elem_bits

    src_elements = _region_elements(op_call.src)
    dst_elements = _region_elements(op_call.dst)
    if not _can_prove_equal(src_elements, expected_elements):
        return False, f"src region does not contain exactly {expected_elements} elements"
    if not _can_prove_equal(dst_elements, expected_elements):
        return False, f"dst region does not contain exactly {expected_elements} elements"

    cache, hints = _ld_cache_config(op_call)
    if cache not in (None, "nc"):
        return False, f"unsupported cache config {cache!r}; expected None or 'nc'"
    if (cache is not None or hints) and src.scope() != "global":
        return False, (
            "cache/L1/L2 hint config applies to global loads; "
            f"requires src scope 'global', got {src.scope()!r}"
        )
    return True, None


def _emit_forced_vec_copy(op_call: TilePrimitiveCall, _sctx: DispatchContext, num_bytes: int):
    op_call = TilePrimitiveCall.downcast(op_call)
    src: Buffer = op_call.src.buffer
    dst: Buffer = op_call.dst.buffer
    src_scope = src.scope()
    dst_scope = dst.scope()
    src_is_local = src_scope == "local"
    dst_is_local = dst_scope == "local"
    src_space = _ptx_space(src_scope)
    dst_space = _ptx_space(dst_scope)
    src_ptr = src.ptr_to(_region_start(op_call.src))
    dst_ptr = dst.ptr_to(_region_start(op_call.dst))
    elem_bits = DataType(src.dtype).bits
    tail, lanes, reg_dtype = copy_ptxd_form(num_bytes)
    container_bits = num_bytes * 8 // lanes

    def _words(buf, region):
        """The container lvalues a local region's registers land in.

        The instruction moves whole containers, so the element region start is
        scaled to a container index; the vector alignment rule the caller
        already enforced makes the division exact.
        """
        flat = buf.view(-1).view(reg_dtype)
        start, stride = 0, 1
        for idx, extent in zip(reversed(_region_start(region)), reversed(list(buf.shape))):
            start = start + idx * stride
            stride = stride * extent
        base = (
            start * (elem_bits // container_bits)
            if elem_bits > container_bits
            else start // (container_bits // elem_bits)
        )
        return [flat[base + i] for i in range(lanes)]

    # Chains are built here: a Python string bound inside the traced body is
    # not something the parser can carry. `nc` folds into the chain, so the
    # use_nc branches collapse.
    st_chain = f"st.{dst_space}.{tail}"
    cache, hints = _ld_cache_config(op_call)
    use_nc = cache == "nc"
    l1_evict = hints.get("l1_evict", "")
    l2_evict = hints.get("l2_evict", "")
    prefetch_size = hints.get("prefetch_size", "")
    ld_chain = copy_ptxd_ld_chain(
        src_space,
        tail,
        nc=use_nc,
        l1_evict=l1_evict,
        l2_evict=l2_evict,
        prefetch_size=prefetch_size,
    )

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def impl():
        if src_is_local:
            T.ptxd[st_chain](dst_ptr, *_words(src, op_call.src))
        elif dst_is_local:
            T.ptxd[ld_chain](*_words(dst, op_call.dst), src_ptr)
        else:
            tmp = T.alloc_local((lanes,), reg_dtype)
            T.ptxd[ld_chain](*[tmp[i] for i in range(lanes)], src_ptr)
            T.ptxd[st_chain](dst_ptr, *[tmp[i] for i in range(lanes)])
    # fmt: on
    return impl


def _register_forced_vec_copy(variant: str, num_bytes: int) -> None:
    @register_dispatch(
        "copy",
        "cuda",
        variant=variant,
        priority=20,
        when=[
            predicate(
                f"{variant}_applicable",
                _is_forced_vec_copy,
                variant=variant,
                num_bytes=num_bytes,
            )
        ],
    )
    def _copy_schedule_forced_vec(
        op_call: TilePrimitiveCall,
        sctx: DispatchContext,
        _num_bytes=num_bytes,
    ) -> PrimFunc:
        return _emit_forced_vec_copy(op_call, sctx, _num_bytes)


_register_forced_vec_copy("vec_256b", 32)
_register_forced_vec_copy("vec_128b", 16)
_register_forced_vec_copy("vec_64b", 8)
_register_forced_vec_copy("vec_32b", 4)
_register_forced_vec_copy("vec_16b", 2)
