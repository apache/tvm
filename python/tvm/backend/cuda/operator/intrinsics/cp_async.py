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
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments, too-many-locals, too-many-positional-arguments
"""PTX cp.async / cp.async.bulk / cp.async.bulk.tensor intrinsics.

Each PTX form table entry is registered as one ``device_intrinsic``.
User-facing wrappers in ``tvm.tirx.op`` keep their v1 signatures;
``register_codegen`` dispatchers below decode the (cp_size, fill_mode,
predicate) / (dim, cta_mask, tile_mode) arguments to pick the right form.
Bodies are hand-written ``asm volatile(...)`` strings.  The file is grouped
as cp.async, cp.async.bulk.tensor, cp.async.bulk non-TMA, and CUDA
compatibility helpers.
"""

import tvm
from tvm.backend.cuda.op import cuda_func_call

from ._schema import device_intrinsic
from .registry import CODEGEN_REGISTRY, register_codegen
from .utils import parse_str

_PREFETCH_CHOICES = ("", "64", "128", "256")
_DIM_CHOICES = (1, 2, 3, 4, 5)
_TILE_MODE_CHOICES = ("tile", "tile_gather4")


def _safe(s):
    return s.replace("::", "_").replace(".", "_")


# =============================================================================
# cp.async forms from the PTX Syntax block.
#
# Includes commit/wait plus the non-bulk shared/global copy forms.
# =============================================================================
device_intrinsic(
    "ptx_cp_async_commit_group",
    helper_name="tvm_builtin_ptx_cp_async_commit_group",
    body='    asm volatile("cp.async.commit_group;");',
)
device_intrinsic(
    "ptx_cp_async_wait_group",
    n_attrs=1,
    helper_name=lambda n: f"tvm_builtin_ptx_cp_async_wait_group_{int(n)}",
    body=lambda n: f'    asm volatile("cp.async.wait_group {int(n)};");',
)


# cp.async non-bulk copy forms:
#   Form 1: cp.async.ca.shared.global ... [dst], [src], cp-size{, src-size}{, cache-policy}
#   Form 2: cp.async.cg.shared.global ... [dst], [src], 16{, src-size}{, cache-policy}
#   Form 3: cp.async.ca.shared.global ... [dst], [src], cp-size{, ignore-src}{, cache-policy}
#   Form 4: cp.async.cg.shared.global ... [dst], [src], 16{, ignore-src}{, cache-policy}


def _cp_async_modifier_str(has_cache_hint, prefetch_size):
    s = ""
    if has_cache_hint:
        s += ".L2::cache_hint"
    if prefetch_size:
        s += f".L2::{prefetch_size}B"
    return s


def _make_form_parts(ca_or_cg, fixed_cp_size, extra):
    """Build a parts callable for one of the cp.async PTX forms.

    Args layout: (dst, src [, extra_int], cache_policy, has_cache, prefetch_size [, cp_size_attr])
    Forwarded operands: dst, src [, extra_int], cache_policy.
    Trailing attrs: has_cache, prefetch_size [, cp_size if .ca].
    """
    n_op = 3 if extra is not None else 2
    n_attrs = 2 if fixed_cp_size is not None else 3
    extra_in_name = f"_with_{extra}" if extra is not None else ""

    def _parts(*args):
        # Operand args (forwarded) come first, then attr args.
        attr_args = args[-n_attrs:]
        has_cache = _bool_attr(attr_args[0])
        prefetch_size = parse_str(attr_args[1])
        cp_size = fixed_cp_size if fixed_cp_size is not None else int(attr_args[2])
        modifier = _cp_async_modifier_str(has_cache, prefetch_size)
        cache_operand = ', "l"(cache_policy)' if has_cache else ""
        # name parts
        name_cache = "_cache_hint" if has_cache else ""
        name_prefetch = f"_prefetch_{prefetch_size}" if prefetch_size else ""
        name = (
            f"tvm_builtin_ptx_cp_async_{ca_or_cg}_{cp_size}"
            f"{name_cache}{name_prefetch}{extra_in_name}"
        )
        sig = (
            "(void* dst, void* src"
            + (f", int {extra}" if extra else "")
            + ", unsigned long long cache_policy)"
        )
        instr_base = f"cp.async.{ca_or_cg}.shared.global{modifier}"
        if extra is None:
            cache_arg = ", %2" if has_cache else ""
            body = (
                "    unsigned int dst_addr = __cvta_generic_to_shared(dst);\n"
                f'    asm volatile("{instr_base} [%0], [%1], {cp_size}{cache_arg};\\n"\n'
                f'                 :: "r"(dst_addr), "l"(src){cache_operand} : "memory");'
            )
        else:
            cache_arg = ", %3" if has_cache else ""
            body = (
                "    unsigned int dst_addr = __cvta_generic_to_shared(dst);\n"
                f'    asm volatile("{instr_base} [%0], [%1], {cp_size}, %2{cache_arg};\\n"\n'
                f'                 :: "r"(dst_addr), "l"(src), "r"({extra})'
                f'{cache_operand} : "memory");'
            )
        return name, sig, body

    return _parts, n_op + n_attrs - n_op  # n_attrs


def _register_nb_form(op_name, ca_or_cg, fixed_cp_size, extra):
    parts_fn, n_attrs = _make_form_parts(ca_or_cg, fixed_cp_size, extra)
    n_op = 3 if extra is not None else 2
    sig_static = (
        "(void* dst, void* src"
        + (f", int {extra}" if extra else "")
        + ", unsigned long long cache_policy)"
    )
    device_intrinsic(
        f"ptx_cp_async_{op_name}",
        n_attrs=n_attrs,
        c_signature=sig_static,  # static — depends on `extra` not on attrs
        helper_name=lambda *a, fn=parts_fn: fn(*a)[0],
        body=lambda *a, fn=parts_fn: fn(*a)[2],
    )
    return n_op


# Form 1: .ca + src-size (cp-size ∈ {4, 8}). src-size is required when present.
_register_nb_form("ca_src_size", "ca", fixed_cp_size=None, extra="src_size")
# Form 2: .cg + src-size (cp-size = 16).
_register_nb_form("cg_src_size", "cg", fixed_cp_size=16, extra="src_size")
# Form 3: .ca + ignore-src.
_register_nb_form("ca_ignore_src", "ca", fixed_cp_size=None, extra="ignore_src")
# Form 4: .cg + ignore-src.
_register_nb_form("cg_ignore_src", "cg", fixed_cp_size=16, extra="ignore_src")
# Plain degenerate of forms 1+2 with optional src-size omitted.
_register_nb_form("ca", "ca", fixed_cp_size=None, extra=None)
_register_nb_form("cg", "cg", fixed_cp_size=16, extra=None)


def _make_setp_at_p_helper(ca_or_cg, cp_size, has_cache, prefetch):
    """Wrapper convenience: ``setp+@p`` around a form 1/2 cp.async (predicate-
    gated skip with dst untouched on false). Not a PTX form — emitted directly
    here as a one-off helper rather than a separate device_intrinsic."""
    modifier = _cp_async_modifier_str(has_cache, prefetch)
    cache_arg = ", %4" if has_cache else ""
    cache_operand = ', "l"(cache_policy)' if has_cache else ""
    func_name = (
        f"tvm_builtin_ptx_cp_async_{cp_size}"
        + ("_cache_hint" if has_cache else "")
        + (f"_prefetch_{prefetch}" if prefetch else "")
        + "_predicate"
    )
    body = (
        "  unsigned int dst_addr = __cvta_generic_to_shared(dst);\n"
        "  __asm__ __volatile__(\n"
        '    "{\\n"\n'
        '    " .reg .pred p;\\n"\n'
        '    " setp.eq.u32 p, %3, 1;\\n"\n'
        f'    " @p cp.async.{ca_or_cg}.shared.global{modifier}'
        f' [%0], [%1], %2{cache_arg};\\n"\n'
        '    "}\\n"\n'
        f'    :: "r"(dst_addr), "l"(src), "n"({cp_size}), "r"(predicate){cache_operand}\n'
        "  );"
    )
    source_code = (
        f"\n__forceinline__ __device__ void {func_name}"
        "(void* dst, void* src, int predicate, unsigned long long cache_policy) {\n"
        f"{body}\n"
        "}\n"
    )
    return func_name, source_code


@register_codegen("ptx_cp_async")
def codegen_ptx_cp_async(*args):
    """Map the wrapper API to the 4 PTX form table entries.

    Accepts three call shapes (sorted by arity):

    * 5 args ``(dst_ptr, dst_offset, src_ptr, src_offset, cp_size)`` —
      the legacy form emitted by
      ``tvm.backend.cuda.transform.InjectPTXAsyncCopy``.
      Offsets are folded into the pointers via ``tvm_access_ptr`` (in
      bytes; offsets are pre-scaled by the pass) and the call is
      forwarded with default cache / predicate / fill_mode.
    * 6 args ``(dst_ptr, dst_offset, src_ptr, src_offset, cp_size,
      predicate)`` — same as 5-arg form with an explicit predicate,
      zero-filling the destination when the predicate is false.
    * 8 args ``(dst_ptr, src_ptr, cp_size, cache_policy, has_cache_hint,
      prefetch_size, predicate, fill_mode)`` — the fork-native wrapper
      API.

    The three resulting form_kinds:

    * ``fill_mode == "zero"`` -> form 1/2 (src-size = predicate ? cp_size : 0)
    * ``predicate != -1`` and no fill_mode -> form 1/2 wrapped in setp+@p
      (wrapper convenience; not a PTX form)
    * else -> form 1/2 with src-size omitted (the "plain" degenerate)
    """
    from tvm.tirx.op import if_then_else

    if len(args) in (5, 6):
        # Legacy InjectPTXAsyncCopy emission: (dst_ptr, dst_off, src_ptr,
        # src_off, cp_size [, predicate]). Offsets are element indices into
        # the typed buffers (the pass uses index_factor=1 except for the
        # shared.dyn-merged byte-buffer path). Emit a C helper that scales
        # the offset by the buffer element size, then runs cp.async.
        #
        # PTX plain form for both .ca and .cg is just
        # ``cp.async.<v>.shared.global [dst], [src], cp_size;`` — three
        # operands, no trailing src-size / cache-policy.
        from tvm import DataType

        dst_ptr_in, dst_offset, src_ptr_in, src_offset, cp_size = args[:5]
        predicate = args[5] if len(args) == 6 else -1
        cp_size_v = int(cp_size)
        ca_or_cg = "cg" if cp_size_v == 16 else "ca"

        # Recover the per-side element dtype from each pointer's type
        # type (Var has ty = PointerType(PrimType(dtype))).
        # InjectPTXAsyncCopy emits offsets in element-units of each side's
        # buffer dtype (dst gets dst_offset * src_elem_size only when dst is a
        # merged shared.dyn byte buffer, in which case dst_elem_dtype is uint8
        # and the resulting scale-by-1 is a no-op).
        def _elem_bytes(ptr):
            ta = getattr(ptr, "ty", None)
            if ta is None or getattr(ta, "element_type", None) is None:
                return 1
            et = ta.element_type
            if not hasattr(et, "dtype"):
                return 1
            bits = DataType(str(et.dtype)).bits
            assert bits % 8 == 0, f"non-byte element dtype: {et.dtype}"
            return bits // 8

        dst_elem_bytes = _elem_bytes(dst_ptr_in)
        src_elem_bytes = _elem_bytes(src_ptr_in)
        has_predicate = not (
            (isinstance(predicate, int) and predicate == -1)
            or (hasattr(predicate, "value") and int(predicate.value) == -1)
        )

        def _scale(n):
            return "" if n == 1 else f" * {n}"

        dst_scale = _scale(dst_elem_bytes)
        src_scale = _scale(src_elem_bytes)
        if has_predicate:
            func_name = (
                f"ptx_cp_async_legacy_pred_{ca_or_cg}_{cp_size_v}_{dst_elem_bytes}_{src_elem_bytes}"
            )
            if cp_size_v == 4:
                zero_fill = '    " @!p st.shared.u32 [%0], {%4};\\n"\n'
            elif cp_size_v == 8:
                zero_fill = '    " @!p st.shared.v2.u32 [%0], {%4, %4};\\n"\n'
            elif cp_size_v == 16:
                zero_fill = '    " @!p st.shared.v4.u32 [%0], {%4, %4, %4, %4};\\n"\n'
            else:
                raise ValueError(f"unsupported legacy predicated cp.async size: {cp_size_v}")
            body = (
                f"  uint8_t* dst_p = (uint8_t*)dst + dst_off{dst_scale};\n"
                f"  uint8_t* src_p = (uint8_t*)src + src_off{src_scale};\n"
                "  unsigned int dst_addr = __cvta_generic_to_shared(dst_p);\n"
                "  __asm__ __volatile__(\n"
                '    "{\\n"\n'
                '    " .reg .pred p;\\n"\n'
                '    " setp.eq.u32 p, %3, 1;\\n"\n'
                f'    " @p cp.async.{ca_or_cg}.shared.global'
                ' [%0], [%1], %2;\\n"\n'
                f"{zero_fill}"
                '    "}\\n"\n'
                f'    :: "r"(dst_addr), "l"(src_p), "n"({cp_size_v}), "r"(predicate), "r"(0)\n'
                "  );"
            )
            source_code = (
                f"\n__forceinline__ __device__ void {func_name}"
                "(void* dst, int dst_off, void* src, int src_off, int predicate) {\n"
                f"{body}\n"
                "}\n"
            )
            return cuda_func_call(
                func_name,
                dst_ptr_in,
                dst_offset,
                src_ptr_in,
                src_offset,
                predicate,
                source_code=source_code,
            )
        # No predicate — plain cp.async.
        func_name = f"ptx_cp_async_legacy_{ca_or_cg}_{cp_size_v}_{dst_elem_bytes}_{src_elem_bytes}"
        body = (
            f"  uint8_t* dst_p = (uint8_t*)dst + dst_off{dst_scale};\n"
            f"  uint8_t* src_p = (uint8_t*)src + src_off{src_scale};\n"
            "  unsigned int dst_addr = __cvta_generic_to_shared(dst_p);\n"
            f'  asm volatile("cp.async.{ca_or_cg}.shared.global'
            ' [%0], [%1], %2;"\n'
            f'    :: "r"(dst_addr), "l"(src_p), "n"({cp_size_v}));'
        )
        source_code = (
            f"\n__forceinline__ __device__ void {func_name}"
            "(void* dst, int dst_off, void* src, int src_off) {\n"
            f"{body}\n"
            "}\n"
        )
        return cuda_func_call(
            func_name,
            dst_ptr_in,
            dst_offset,
            src_ptr_in,
            src_offset,
            source_code=source_code,
        )
    elif len(args) == 8:
        (
            dst_ptr,
            src_ptr,
            cp_size,
            cache_policy,
            has_cache_hint,
            prefetch_size,
            predicate,
            fill_mode,
        ) = args
    else:
        raise ValueError(f"ptx_cp_async codegen expects 5/6/8 args, got {len(args)}")

    cp_size_v = int(cp_size)
    ca_or_cg = "cg" if cp_size_v == 16 else "ca"
    pref = "" if int(prefetch_size) == -1 else str(int(prefetch_size))
    fill = parse_str(fill_mode)
    has_cache = _bool_attr(has_cache_hint)
    has_predicate = not (
        (isinstance(predicate, int) and predicate == -1)
        or (hasattr(predicate, "value") and int(predicate.value) == -1)
    )

    if fill == "zero":
        src_size = if_then_else(predicate != 0, cp_size_v, 0)
        op = f"tirx.ptx_cp_async_{ca_or_cg}_src_size"
        if cp_size_v == 16:
            args = [dst_ptr, src_ptr, src_size, cache_policy, has_cache, pref]
        else:
            args = [dst_ptr, src_ptr, src_size, cache_policy, has_cache, pref, cp_size_v]
        result = CODEGEN_REGISTRY[op](args)
        return result[0] if isinstance(result, tuple) else result

    if has_predicate:
        func_name, source_code = _make_setp_at_p_helper(ca_or_cg, cp_size_v, has_cache, pref)
        return cuda_func_call(
            func_name, dst_ptr, src_ptr, predicate, cache_policy, source_code=source_code
        )

    # Plain — form 1/2 with src-size omitted.
    op = f"tirx.ptx_cp_async_{ca_or_cg}"
    if cp_size_v == 16:
        args = [dst_ptr, src_ptr, cache_policy, has_cache, pref]
    else:
        args = [dst_ptr, src_ptr, cache_policy, has_cache, pref, cp_size_v]
    result = CODEGEN_REGISTRY[op](args)
    return result[0] if isinstance(result, tuple) else result


CODEGEN_REGISTRY["tirx.ptx.cp_async_raw"] = CODEGEN_REGISTRY["tirx.ptx.cp_async"]


# =============================================================================
# cp.async.bulk.tensor (TMA) — one device_intrinsic per arity variant of each
# PTX form. Per-dim coord operands materialise via the ``c_signature`` callable.
# =============================================================================


def _is_sm100_or_higher():
    target = tvm.target.Target.current()
    if target is None:
        return False
    arch = target.arch[3:]
    if not arch[-1].isdigit():
        arch = arch[:-1]
    return int(arch) >= 100


def _resolve_cta_group_str(cta_group):
    if cta_group == 2 or (cta_group != -1 and _is_sm100_or_higher()):
        return f".cta_group::{cta_group}"
    return ""


def _coord_template(coord_count, start_slot):
    inner = ", ".join(f"%{start_slot + i}" for i in range(coord_count))
    return f"{{{inner}}}"


def _coord_constraints(coord_count):
    return ", ".join(f'"r"(coord{i})' for i in range(coord_count))


def _coord_sig(n):
    return ", ".join(f"int coord{i}" for i in range(n))


# PTX cp.async.bulk.tensor global -> shared::cluster form:
#   cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism
#       {.multicast}{.cta_group}{.level::cache_hint}
#       [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo}
#       {, ctaMask} {, cache-policy}
#   .dst = {.shared::cluster}; .src = {.global}
#   .completion_mechanism = {.mbarrier::complete_tx::bytes}
#   .multicast = {.multicast::cluster}
#   .cta_group = {.cta_group::1, .cta_group::2}
#   .load_mode = {.tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128}
#   .level::cache_hint = {.L2::cache_hint}
# This registration supports tile/tile::gather4 modes; ctaMask is only used
# when the optional ``.multicast::cluster`` modifier is enabled.
def _g2cluster_parts(*args):
    attrs = args[-6:]
    dim = int(attrs[0])
    cta_group = int(attrs[1])
    has_cache = _bool_attr(attrs[2])
    tile_mode = parse_str(attrs[3])
    bar_is_addr = _bool_attr(attrs[4])
    multicast = _bool_attr(attrs[5])
    coord_count = 5 if tile_mode == "tile_gather4" else dim
    bar_type = "unsigned int bar_addr" if bar_is_addr else "void* bar"
    sig = (
        f"(void* dst, {bar_type}, unsigned long long tensormap_addr, "
        "uint16_t cta_mask, unsigned long long cache_policy"
        + (", " + _coord_sig(coord_count) if coord_count else "")
        + ")"
    )
    name = (
        f"ptx_cp_async_bulk_tensor_g2cluster_{tile_mode}_{dim}d"
        f"{'_multicast' if multicast else ''}"
        f"{'_cache_hint' if has_cache else ''}{'_bar_addr' if bar_is_addr else ''}"
    )
    tile_modifier = ".tile::gather4" if tile_mode == "tile_gather4" else ""
    cta_group_str = _resolve_cta_group_str(cta_group)
    multicast_inst = ".multicast::cluster" if multicast else ""
    cache_inst = ".L2::cache_hint" if has_cache else ""
    mask_arg = ',\n          "h"(cta_mask)' if multicast else ""
    cache_arg = ',\n          "l"(cache_policy)' if has_cache else ""
    mask_slot = ", %3" if multicast else ""
    cache_slot = ", %4" if multicast and has_cache else ", %3" if has_cache else ""
    coord_start = 5 if multicast and has_cache else 4 if multicast or has_cache else 3
    coord_tpl = _coord_template(coord_count, coord_start)
    instr = (
        f"cp.async.bulk.tensor.{dim}d.shared::cluster.global{tile_modifier}"
        f".mbarrier::complete_tx::bytes{multicast_inst}"
        f"{cta_group_str}{cache_inst}"
    )
    bar_addr_decl = (
        "" if bar_is_addr else "    unsigned int bar_addr = __cvta_generic_to_shared(bar);\n"
    )
    body = (
        "    unsigned int dst_addr = __cvta_generic_to_shared(dst);\n"
        f"{bar_addr_decl}"
        "    asm volatile(\n"
        f'        "{instr} [%0], [%1, {coord_tpl}], [%2]{mask_slot}{cache_slot};"\n'
        "        :\n"
        f'        : "r"(dst_addr), "l"(tensormap_addr), "r"(bar_addr){mask_arg}{cache_arg},\n'
        f"          {_coord_constraints(coord_count)}\n"
        '        : "memory"\n'
        "    );"
    )
    return name, sig, body


device_intrinsic(
    "ptx_cp_async_bulk_tensor_g2cluster",
    n_attrs=6,
    helper_name=lambda *a: _g2cluster_parts(*a)[0],
    c_signature=lambda *a: _g2cluster_parts(*a)[1],
    body=lambda *a: _g2cluster_parts(*a)[2],
)


# PTX cp.async.bulk.tensor shared::cta -> global form:
#   cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism
#       {.level::cache_hint}
#       [tensorMap, tensorCoords], [srcMem] {, cache-policy}
#   .dst = {.global}; .src = {.shared::cta}
#   .completion_mechanism = {.bulk_group}
#   .load_mode = {.tile, .tile::scatter4, .im2col_no_offs}
#   .level::cache_hint = {.L2::cache_hint}
# This registration supports tile mode; cache-policy is a real operand.
def _s2g_parts(*args):
    attrs = args[-2:]
    dim = int(attrs[0])
    has_cache = _bool_attr(attrs[1])
    sig = (
        "(void* src, unsigned long long tensormap_addr, unsigned long long cache_policy"
        + (", " + _coord_sig(dim) if dim else "")
        + ")"
    )
    name = f"ptx_cp_async_bulk_tensor_shared_to_global_{dim}d{'_cache_hint' if has_cache else ''}"
    cache_inst = ".L2::cache_hint" if has_cache else ""
    cache_arg = ', "l"(cache_policy)' if has_cache else ""
    cache_slot = ", %2" if has_cache else ""
    coord_start = 3 if has_cache else 2
    coord_tpl = _coord_template(dim, coord_start)
    instr = f"cp.async.bulk.tensor.{dim}d.global.shared::cta.tile.bulk_group{cache_inst}"
    body = (
        "    unsigned int src_addr = __cvta_generic_to_shared(src);\n"
        "    asm volatile(\n"
        f'        "{instr} [%0, {coord_tpl}], [%1]{cache_slot};"\n'
        "        :\n"
        f'        : "l"(tensormap_addr), "r"(src_addr){cache_arg},\n'
        f"          {_coord_constraints(dim)}\n"
        '        : "memory"\n'
        "    );"
    )
    return name, sig, body


device_intrinsic(
    "ptx_cp_async_bulk_tensor_s2g",
    n_attrs=2,
    helper_name=lambda *a: _s2g_parts(*a)[0],
    c_signature=lambda *a: _s2g_parts(*a)[1],
    body=lambda *a: _s2g_parts(*a)[2],
)


# PTX cp.async.bulk.prefetch.tensor form:
#   cp.async.bulk.prefetch.tensor.dim.L2.src{.load_mode}{.level::cache_hint}
#       [tensorMap, tensorCoords] {, im2colInfo} {, cache-policy}
#   .src = {.global}
#   .load_mode = {.tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128}
#   .level::cache_hint = {.L2::cache_hint}
# This registration supports tile mode; cache-policy is a real operand.
def _prefetch_parts(*args):
    attrs = args[-2:]
    dim = int(attrs[0])
    has_cache = _bool_attr(attrs[1])
    sig = (
        "(unsigned long long tensormap_addr, unsigned long long cache_policy"
        + (", " + _coord_sig(dim) if dim else "")
        + ")"
    )
    name = (
        f"ptx_cp_async_bulk_tensor_global_to_cluster_prefetch_{dim}d"
        f"{'_cache_hint' if has_cache else ''}"
    )
    cache_inst = ".L2::cache_hint" if has_cache else ""
    cache_arg = ', "l"(cache_policy)' if has_cache else ""
    cache_slot = ", %1" if has_cache else ""
    coord_start = 2 if has_cache else 1
    coord_tpl = _coord_template(dim, coord_start)
    instr = f"cp.async.bulk.prefetch.tensor.{dim}d.L2.global.tile{cache_inst}"
    body = (
        "    asm volatile(\n"
        f'        "{instr} [%0, {coord_tpl}]{cache_slot};"\n'
        "        :\n"
        f'        : "l"(tensormap_addr){cache_arg},\n'
        f"          {_coord_constraints(dim)}\n"
        '        : "memory"\n'
        "    );"
    )
    return name, sig, body


device_intrinsic(
    "ptx_cp_async_bulk_tensor_prefetch",
    n_attrs=2,
    helper_name=lambda *a: _prefetch_parts(*a)[0],
    c_signature=lambda *a: _prefetch_parts(*a)[1],
    body=lambda *a: _prefetch_parts(*a)[2],
)


# PTX cp.reduce.async.bulk.tensor shared::cta -> global form:
#   cp.reduce.async.bulk.tensor.dim.dst.src.redOp{.load_mode}.completion_mechanism
#       {.level::cache_hint}
#       [tensorMap, tensorCoords], [srcMem] {, cache-policy}
#   .dst = {.global}; .src = {.shared::cta}
#   .completion_mechanism = {.bulk_group}
#   .redOp = {.add, .min, .max, .inc, .dec, .and, .or, .xor}
#   .level::cache_hint = {.L2::cache_hint}
# This registration supports tile mode; redOp is syntax, cache-policy is an operand.
def _reduce_parts(*args):
    attrs = args[-3:]
    dim = int(attrs[0])
    has_cache = _bool_attr(attrs[1])
    red_op = parse_str(attrs[2])
    sig = (
        "(void* src, unsigned long long tensormap_addr, unsigned long long cache_policy"
        + (", " + _coord_sig(dim) if dim else "")
        + ")"
    )
    name = (
        f"ptx_cp_async_bulk_tensor_shared_to_global_reduce_{dim}d"
        f"{'_cache_hint' if has_cache else ''}"
    )
    cache_inst = ".L2::cache_hint" if has_cache else ""
    cache_arg = ', "l"(cache_policy)' if has_cache else ""
    cache_slot = ", %2" if has_cache else ""
    coord_start = 3 if has_cache else 2
    coord_tpl = _coord_template(dim, coord_start)
    instr = (
        f"cp.reduce.async.bulk.tensor.{dim}d.global.shared::cta"
        f".{red_op}.tile.bulk_group{cache_inst}"
    )
    body = (
        "    unsigned int src_addr = __cvta_generic_to_shared(src);\n"
        "    asm volatile(\n"
        f'        "{instr} [%0, {coord_tpl}], [%1]{cache_slot};"\n'
        "        :\n"
        f'        : "l"(tensormap_addr), "r"(src_addr){cache_arg},\n'
        f"          {_coord_constraints(dim)}\n"
        '        : "memory"\n'
        "    );"
    )
    return name, sig, body


device_intrinsic(
    "ptx_cp_async_bulk_tensor_reduce",
    n_attrs=3,
    helper_name=lambda *a: _reduce_parts(*a)[0],
    c_signature=lambda *a: _reduce_parts(*a)[1],
    body=lambda *a: _reduce_parts(*a)[2],
)


# User-facing dispatchers for tensor global -> shared::cluster.  The same
# backend root handles the optional ``.multicast::cluster`` modifier.


def _g2c_dispatch(dim, dst_ptr, bar, tensormap, *args, tile_mode):
    cta_mask, cta_group, cache_policy, has_cache, *rest = args
    coord_count = 5 if tile_mode == "tile_gather4" else int(dim)
    if len(rest) == coord_count + 1:
        bar_is_addr = _bool_attr(rest[0])
        coords = rest[1:]
    else:
        bar_is_addr = False
        coords = rest
    is_unicast = isinstance(cta_mask, tvm.tirx.IntImm) and bin(int(cta_mask)).count("1") <= 1
    cg = int(cta_group)
    op = "tirx.ptx_cp_async_bulk_tensor_g2cluster"
    call_args = [
        dst_ptr,
        bar,
        tensormap,
        cta_mask,
        cache_policy,
        *coords,
        int(dim),
        cg,
        has_cache,
        tile_mode,
        bar_is_addr,
        int(not is_unicast),
    ]
    result = CODEGEN_REGISTRY[op](call_args)
    return result[0] if isinstance(result, tuple) else result


@register_codegen("ptx_cp_async_bulk_tensor_global_to_cluster")
def codegen_g2c(dim, dst_ptr, bar, tensormap, *args):
    return _g2c_dispatch(dim, dst_ptr, bar, tensormap, *args, tile_mode="tile")


@register_codegen("ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster")
def codegen_g2c_gather4(dim, dst_ptr, bar, tensormap, *args):
    return _g2c_dispatch(dim, dst_ptr, bar, tensormap, *args, tile_mode="tile_gather4")


@register_codegen("ptx_cp_async_bulk_tensor_shared_to_global")
def codegen_s2g(dim, src_ptr, tensormap, *args):
    cache_policy, has_cache, *coords = args
    result = CODEGEN_REGISTRY["tirx.ptx_cp_async_bulk_tensor_s2g"](
        [src_ptr, tensormap, cache_policy, *coords, int(dim), has_cache]
    )
    return result[0] if isinstance(result, tuple) else result


@register_codegen("ptx_cp_async_bulk_tensor_global_to_cluster_prefetch")
def codegen_prefetch(dim, tensormap, *args):
    cache_policy, has_cache, *coords = args
    result = CODEGEN_REGISTRY["tirx.ptx_cp_async_bulk_tensor_prefetch"](
        [tensormap, cache_policy, *coords, int(dim), has_cache]
    )
    return result[0] if isinstance(result, tuple) else result


@register_codegen("ptx_cp_async_bulk_tensor_shared_to_global_reduce")
def codegen_reduce(dim, src_ptr, tensormap, *args):
    cache_policy, has_cache, red_op, *coords = args
    result = CODEGEN_REGISTRY["tirx.ptx_cp_async_bulk_tensor_reduce"](
        [src_ptr, tensormap, cache_policy, *coords, int(dim), has_cache, red_op]
    )
    return result[0] if isinstance(result, tuple) else result


# =============================================================================
# cp.async.bulk non-TMA forms from the PTX Syntax block. Each form is one
# device_intrinsic; optional PTX modifiers are attrs, not separate fixed ops.
# =============================================================================
device_intrinsic(
    "ptx_cp_async_bulk_commit_group",
    helper_name="ptx_cp_async_bulk_tensor_commit_group",
    body='    asm volatile("cp.async.bulk.commit_group;");',
)


def _ptx_cp_async_bulk_wait_group_parts(n, read):
    n = int(n)
    read_b = bool(int(read)) if hasattr(read, "value") else bool(read)
    return (
        f"ptx_cp_async_bulk_wait_group{'_read' if read_b else ''}_{n}",
        f'    asm volatile("cp.async.bulk.wait_group{".read" if read_b else ""} {n};");',
    )


device_intrinsic(
    "ptx_cp_async_bulk_wait_group",
    n_attrs=2,
    helper_name=lambda n, read: _ptx_cp_async_bulk_wait_group_parts(n, read)[0],
    body=lambda n, read: _ptx_cp_async_bulk_wait_group_parts(n, read)[1],
)


def _bool_attr(value):
    return bool(int(value)) if hasattr(value, "value") else bool(value)


def _bulk_cache_operand_constraint(has_cache):
    return ', "l"(cache_policy)' if has_cache else ""


def _bulk_cache_operand_suffix(has_cache):
    return ".L2::cache_hint" if has_cache else ""


# PTX cp.async.bulk global -> shared::cta form:
#   cp.async.bulk.dst.src.completion_mechanism{.level::cache_hint}{.ignore_oob}
#       [dstMem], [srcMem], size{, ignoreBytesLeft, ignoreBytesRight}, [mbar] {, cache-policy}
#   .dst = {.shared::cta}; .src = {.global}
#   .completion_mechanism = {.mbarrier::complete_tx::bytes}
#   .level::cache_hint = {.L2::cache_hint}
def _bulk_g2s_cta_parts(*args):
    has_cache = _bool_attr(args[-2])
    ignore_oob = _bool_attr(args[-1])
    instr = (
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
        f"{_bulk_cache_operand_suffix(has_cache)}{'.ignore_oob' if ignore_oob else ''}"
    )
    if ignore_oob:
        asm_args = (
            '"r"(dst), "l"(src_ptr), "r"(num_bytes), "r"(ignore_bytes_left), '
            '"r"(ignore_bytes_right), "r"(mbarrier)'
        )
        operands = "%2, %3, %4, [%5]"
        cache_slot = ", %6" if has_cache else ""
    else:
        asm_args = '"r"(dst), "l"(src_ptr), "r"(num_bytes), "r"(mbarrier)'
        operands = "%2, [%3]"
        cache_slot = ", %4" if has_cache else ""
    body = (
        "    unsigned int dst = (unsigned int)__cvta_generic_to_shared(dst_ptr);\n"
        "    unsigned int mbarrier = (unsigned int)__cvta_generic_to_shared(mbarrier_ptr);\n"
        f'    asm volatile("{instr} [%0], [%1], {operands}{cache_slot};"\n'
        "                 :\n"
        f"                 : {asm_args}{_bulk_cache_operand_constraint(has_cache)}\n"
        '                 : "memory");'
    )
    name = (
        "tvm_builtin_ptx_cp_async_bulk_g2s_cta"
        f"{'_cache_hint' if has_cache else ''}{'_ignore_oob' if ignore_oob else ''}"
    )
    return name, body


device_intrinsic(
    "ptx_cp_async_bulk_g2s_cta",
    n_attrs=2,
    helper_name=lambda *a: _bulk_g2s_cta_parts(*a)[0],
    c_signature=(
        "(void* dst_ptr, void* src_ptr, unsigned int num_bytes, "
        "unsigned int ignore_bytes_left, unsigned int ignore_bytes_right, "
        "void* mbarrier_ptr, unsigned long long cache_policy)"
    ),
    body=lambda *a: _bulk_g2s_cta_parts(*a)[1],
)


# PTX cp.async.bulk global -> shared::cluster form:
#   cp.async.bulk.dst.src.completion_mechanism{.multicast}{.level::cache_hint}
#       [dstMem], [srcMem], size, [mbar] {, ctaMask} {, cache-policy}
#   .dst = {.shared::cluster}; .src = {.global}
#   .completion_mechanism = {.mbarrier::complete_tx::bytes}
#   .level::cache_hint = {.L2::cache_hint}
#   .multicast = {.multicast::cluster}
def _bulk_g2s_cluster_parts(*args):
    has_cache = _bool_attr(args[-2])
    multicast = _bool_attr(args[-1])
    instr = (
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        f"{'.multicast::cluster' if multicast else ''}{_bulk_cache_operand_suffix(has_cache)}"
    )
    cta_constraint = ', "h"(cta_mask)' if multicast else ""
    mask_slot = ", %4" if multicast else ""
    cache_slot = ", %5" if multicast and has_cache else ", %4" if has_cache else ""
    body = (
        "    unsigned int dst = (unsigned int)__cvta_generic_to_shared(dst_ptr);\n"
        "    unsigned int mbarrier = (unsigned int)__cvta_generic_to_shared(mbarrier_ptr);\n"
        f'    asm volatile("{instr} [%0], [%1], %2, [%3]'
        f'{mask_slot}{cache_slot};"\n'
        "                 :\n"
        '                 : "r"(dst), "l"(src_ptr), "r"(num_bytes), "r"(mbarrier)'
        f"{cta_constraint}{_bulk_cache_operand_constraint(has_cache)}\n"
        '                 : "memory");'
    )
    name = (
        "tvm_builtin_ptx_cp_async_bulk_g2s_cluster"
        f"{'_multicast' if multicast else ''}{'_cache_hint' if has_cache else ''}"
    )
    return name, body


device_intrinsic(
    "ptx_cp_async_bulk_g2s_cluster",
    n_attrs=2,
    helper_name=lambda *a: _bulk_g2s_cluster_parts(*a)[0],
    c_signature=(
        "(void* dst_ptr, void* src_ptr, unsigned int num_bytes, "
        "void* mbarrier_ptr, unsigned short cta_mask, unsigned long long cache_policy)"
    ),
    body=lambda *a: _bulk_g2s_cluster_parts(*a)[1],
)


# PTX cp.async.bulk shared::cta -> shared::cluster form:
#   cp.async.bulk.dst.src.completion_mechanism [dstMem], [srcMem], size, [mbar]
#   .dst = {.shared::cluster}; .src = {.shared::cta}
#   .completion_mechanism = {.mbarrier::complete_tx::bytes}
device_intrinsic(
    "ptx_cp_async_bulk_s2s_cluster",
    helper_name="tvm_builtin_ptx_cp_async_bulk_s2s_cluster",
    c_signature="(uint64_t dst, void* src, int size, uint64_t mbar)",
    body=r"""    unsigned int dst_addr = static_cast<unsigned int>(dst);
    unsigned int src_addr = __cvta_generic_to_shared(src);
    unsigned int mbar_addr = static_cast<unsigned int>(mbar);
    asm volatile(
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :
        : "r"(dst_addr), "r"(src_addr), "r"(size), "r"(mbar_addr)
        : "memory");""",
)


@register_codegen("ptx_cp_async_bulk_shared_to_cluster")
def codegen_ptx_cp_async_bulk_shared_to_cluster(dst_ptr, src_ptr, size, mbar):
    result = CODEGEN_REGISTRY["tirx.ptx_cp_async_bulk_s2s_cluster"]([dst_ptr, src_ptr, size, mbar])
    return result[0] if isinstance(result, tuple) else result


# PTX cp.async.bulk shared::cta -> global form:
#   cp.async.bulk.dst.src.completion_mechanism{.level::cache_hint}{.cp_mask}
#       [dstMem], [srcMem], size {, cache-policy} {, byteMask}
#   .dst = {.global}; .src = {.shared::cta}
#   .completion_mechanism = {.bulk_group}
#   .level::cache_hint = {.L2::cache_hint}
def _bulk_s2g_parts(*args):
    has_cache = _bool_attr(args[-2])
    cp_mask = _bool_attr(args[-1])
    if cp_mask and not has_cache:
        raise ValueError("cp.async.bulk shared::cta -> global .cp_mask requires .L2::cache_hint")
    instr = f"cp.async.bulk.global.shared::cta.bulk_group{_bulk_cache_operand_suffix(has_cache)}"
    if cp_mask:
        instr += ".cp_mask"
    cache_slot = ", %3" if has_cache else ""
    mask_slot = ", %4" if cp_mask else ""
    mask_constraint = ', "r"(byte_mask)' if cp_mask else ""
    body = (
        "    unsigned int src = (unsigned int)__cvta_generic_to_shared(src_ptr);\n"
        f'    asm volatile("{instr} [%0], [%1], %2'
        f'{cache_slot}{mask_slot};"\n'
        "                 :\n"
        '                 : "l"(dst_ptr), "r"(src), "r"(num_bytes)'
        f"{_bulk_cache_operand_constraint(has_cache)}{mask_constraint}\n"
        '                 : "memory");'
    )
    name = (
        "tvm_builtin_ptx_cp_async_bulk_s2g"
        f"{'_cache_hint' if has_cache else ''}{'_cp_mask' if cp_mask else ''}"
    )
    return name, body


device_intrinsic(
    "ptx_cp_async_bulk_s2g",
    n_attrs=2,
    helper_name=lambda *a: _bulk_s2g_parts(*a)[0],
    c_signature=(
        "(void* dst_ptr, void* src_ptr, unsigned int num_bytes, "
        "unsigned int byte_mask, unsigned long long cache_policy)"
    ),
    body=lambda *a: _bulk_s2g_parts(*a)[1],
)
