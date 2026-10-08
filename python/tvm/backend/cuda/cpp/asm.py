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
"""Device helpers whose bodies are hand-written inline asm.

These do not fit the ``T.ptx`` instruction table: each wraps something
other than one instruction with operands -- a spin-wait loop around
``mbarrier.try_wait``, an arrive/wait pair, an empty asm block used purely
as a compiler barrier, a special-register read whose name is baked into
the PTX text, or the offset-scaling ``cp.async`` form the legacy
``InjectPTXAsyncCopy`` pass emits.
"""

import re

from tvm import ir
from tvm.backend.cuda.op import cuda_func_call

from ..codegen.registry import CODEGEN_REGISTRY, register_codegen
from ..codegen.schema import device_intrinsic
from ..codegen.types import PTXDataType
from ..codegen.utils import parse_str

# =============================================================================
# Declared synchronization words. The four direct forms emit exactly what their
# raw PTX spellings do; only the operation's identity differs, which is what
# lets a checker tell a protocol's own accesses from a stray one. The wait is
# the one that generates something new: a loop, which is why it lives here and
# not in the instruction table.
# =============================================================================
_WAIT_UNTIL_SCALARS = {
    "int32": "s32",
    "uint32": "u32",
    "int64": "s64",
    "uint64": "u64",
    "uint128": "b128",
    "int128": "b128",
}


_WAIT_UNTIL_PTX_WIDTH = {
    "b32": 32,
    "s32": 32,
    "u32": 32,
    "b64": 64,
    "s64": 64,
    "u64": 64,
    "b128": 128,
}


def _wait_until_scalar_suffix(ty, requested=""):
    """The PTX type this access is spelled with.

    Defaults to the word's own signedness. A caller may ask for another
    spelling of the same width -- `b32` for a load or a store that only moves
    the value, which is how kernels ordinarily write one -- but not for another
    width, which would access a different number of bytes. Whether the
    instruction has the requested form at all is settled by the instruction
    table: `add` has no bit-typed form and rejects one there.
    """
    suffix = _WAIT_UNTIL_SCALARS.get(str(ty))
    if suffix is None:
        raise TypeError(f"sync word must be a 32/64/128-bit integer scalar, got {ty}")
    if not requested:
        return suffix
    width = _WAIT_UNTIL_PTX_WIDTH.get(requested)
    if width is None:
        raise TypeError(f"invalid ptx_type {requested!r}")
    if width != _WAIT_UNTIL_PTX_WIDTH[suffix]:
        raise TypeError(
            f"ptx_type {requested!r} is {width} bits but the sync word is "
            f"{_WAIT_UNTIL_PTX_WIDTH[suffix]} bits"
        )
    return requested


def _wait_until_pointee(ptr, what):
    if not isinstance(ptr.ty, ir.PointerType):
        raise TypeError(f"{what} ptr must be a pointer to the synchronization word")
    return ptr.ty.element_type


def _wait_until_word_suffix(ptr, requested, what):
    """The PTX type the word is accessed as.

    Normally the word's own type decides it and `ptx_type` may respell it at
    the same width. A kernel that addresses its workspace by byte offset has
    no pointee type to read -- the address is an untyped handle -- and there
    `ptx_type` is not a respelling but the statement of how wide the word is,
    so it is required.
    """
    pointee = str(_wait_until_pointee(ptr, what))
    # A 128-bit word spans two 64-bit elements, so a pointer into the pair is
    # how a kernel names it; asking for `.b128` there is not a respelling of
    # the pointee but the statement that the word is the wider one.
    if requested == "b128" and _WAIT_UNTIL_PTX_WIDTH.get(_WAIT_UNTIL_SCALARS.get(pointee, "")) in (
        32,
        64,
    ):
        return requested
    if pointee in _WAIT_UNTIL_SCALARS:
        return _wait_until_scalar_suffix(pointee, requested)
    if not requested:
        raise TypeError(
            f"{what} ptr is an untyped address; pass ptx_type to say how wide the word is"
        )
    if requested not in _WAIT_UNTIL_PTX_WIDTH:
        raise TypeError(f"invalid ptx_type {requested!r}")
    return requested


def _wait_until_same_width(dst_ty, suffix, what):
    """The destination register must be as wide as the word.

    Not the same type: a bit-typed access moves 32 or 64 bits into whatever
    register of that width the caller named, which is how kernels ordinarily
    read a counter they treat as signed out of an unsigned word.
    """
    dst = _wait_until_scalar_suffix(dst_ty)
    if _WAIT_UNTIL_PTX_WIDTH[dst] != _WAIT_UNTIL_PTX_WIDTH[suffix]:
        raise TypeError(
            f"{what} destination is {_WAIT_UNTIL_PTX_WIDTH[dst]} bits but the "
            f"sync word is {_WAIT_UNTIL_PTX_WIDTH[suffix]} bits"
        )


def _wait_until_thread_local_scalar(dst, what):
    if not isinstance(dst, ir.TensorLoad) or dst.source.scope() not in {
        "local",
        "local_scalar",
        "register",
        "reg",
    }:
        raise TypeError(f"{what} dst must be a writable thread-local scalar")
    return dst.ty


def _wait_until_forward(spelling, *args):
    """Emit one instruction-table PTX operation and return its codegen result."""
    from ..ptx import PTXNamespace  # pylint: disable=import-outside-toplevel

    call = PTXNamespace()[spelling](*args)
    return CODEGEN_REGISTRY[call.op.name](call.args)


@register_codegen("cuda_wait_until")
def cuda_wait_until(dst, ptr, condition, scope, space, ptx_type, backoff_ns):
    """Lower a declared wait to a pre-tested loop around one scoped load."""
    scope, space, ptx_type = (parse_str(x) for x in (scope, space, ptx_type))
    dtype = _wait_until_thread_local_scalar(dst, "wait_until")
    suffix = _wait_until_word_suffix(ptr, ptx_type, "wait_until")
    _wait_until_same_width(dtype, suffix, "wait_until")

    # The wait polls relaxed and closes with a single `ld.acquire`, rather than
    # paying acquire semantics on every poll. Measured against the acquiring
    # poll over every benchmarkable kernel that owns a spin wait, interleaved,
    # round 1 dropped:
    #
    #     sm100_fp8_fp4_mega_moe                 8 sites   -1.04%
    #     radix_topk_multi_cta                   1 site    -0.50%
    #     agent_evolved_moe_fp8_blockscale_dsv3  3 sites   +0.03%
    #     agent_evolved_kda_backward_packed      4 sites   -0.08%
    #     cudnn_sm100_flex_attention_backward    1 site    -0.25%
    #
    # The closing read is what takes the edge, and it may observe a value later
    # than the one that satisfied the predicate. That is still the edge the
    # protocol means: these words are published by `red`/`atom` release RMWs,
    # so every contribution sits in one release sequence and an acquire reading
    # any of them synchronizes with all the earlier ones. It reads into a
    # discarded temporary precisely so the later value cannot reach `dst` --
    # the wait's exit value stays the value the predicate accepted, which
    # matters because predicates here are not all monotone (`mega_moe`'s grid
    # barrier tests a sign-bit flip, and the ring waits test equality).
    #
    # This is the wait's only lowering. `ld.volatile` polls the same way -- PTX
    # ISA 8.4.2 puts it and `.relaxed` in one class, and measured they are
    # within 0.1% everywhere -- but `.relaxed.<scope>` says the scope out loud
    # instead of resting on `volatile` meaning `.sys`.
    load_call, tags = _wait_until_forward(f"ld.relaxed.{scope}.{space}.{suffix}", dst, ptr)
    # The helper is named after the poll, which is the load that repeats; the
    # closing acquire hangs off that name with an `_acquire` suffix.
    load_name = parse_str(load_call.args[0])
    name = load_name.replace("ptx_ld_", "cuda_wait_until_")
    source = load_call.args[-1].value.replace(load_name, name + "_load")
    acquire_call, _ = _wait_until_forward(f"ld.acquire.{scope}.{space}.{suffix}", dst, ptr)
    acquire_name = parse_str(acquire_call.args[0])
    acquire_source = acquire_call.args[-1].value.replace(acquire_name, name + "_acquire")
    source += "\n" + acquire_source
    # NVRTC has no `__typeof__`, and `decltype` on the destination yields a
    # reference that cannot be declared uninitialized, so the scratch takes
    # the C type the generated helper already spells in its signature.
    scratch_type = re.search(rf"void\s+{re.escape(name)}_acquire\(\s*([\w:]+)\s*&", acquire_source)
    if scratch_type is None:  # pragma: no cover - the helper shape is fixed
        raise RuntimeError(f"cannot read the destination type of {name}_acquire")
    closing = (
        f" {{ {scratch_type.group(1)} __tirx_wait_edge; "
        f"{name}_acquire(__tirx_wait_edge, (ptr)); (void)__tirx_wait_edge; }}"
    )
    # The predicate stays at the call site: an ordinary bool argument would be
    # evaluated once, before the load ever updates the destination.
    # Load first, test after, which is how every spin loop in this repository
    # is written: `ld; while (!done) { ld; }` reads once before it can decide
    # anything, so a pre-tested loop would need the caller to seed the
    # destination -- and the only way to seed it honestly is another load of
    # the same word, which is one more unguaranteed read for a checker to
    # judge. A `do`/`while` needs no seed.
    #
    # `unroll 1` is what a hand-written spin carries, and what the loop this
    # replaces emitted. Without it nothing stops ptxas from duplicating the
    # load into an unrolled body, which changes how often a waiter polls even
    # though the instruction sequence is the same one.
    #
    # The first load and test are peeled out of the loop. The executed sequence
    # is the same either way, but a single `do`/`while` gives ptxas one body to
    # schedule and it stops emitting the early-exit branch a hand-written
    # `ld; while (!done) { ld; }` gets -- so a waiter whose predicate already
    # holds pays a second load and a poll it did not pay before. Measured on
    # `sm100_fp8_fp4_mega_moe`'s grid barrier at +2.2% and on
    # `radix_topk_multi_cta` at +1.25%, reproduced on two idle B200s with the
    # states interleaved. Peeling costs nothing when the wait does spin.
    #
    # A backoff of zero is no backoff: the macro, and so the emitted code, is
    # the one a wait without one produces, down to the argument list.
    backoff = 0 if not hasattr(backoff_ns, "value") else int(backoff_ns.value)
    if backoff == 0:
        source += (
            f"\n#define {name}(dst, ptr, predicate) "
            f"do {{ {name}_load((dst), (ptr)); if (!(predicate)) {{ "
            f'_Pragma("unroll 1") '
            f"do {{ {name}_load((dst), (ptr)); }} while (!(predicate)); }}"
            f"{closing} }} while (0)\n"
        )
        operands = (condition,)
    else:
        # Between polls, never before the first one and never after the last:
        # the shape `allgather_gemm` and `gemm_reduce_scatter` write by hand.
        name = f"{name}_backoff"
        source = source.replace(f"{name[: -len('_backoff')]}_load", f"{name}_load")
        source += (
            f"\n#define {name}(dst, ptr, predicate, backoff_ns) "
            f"do {{ {name}_load((dst), (ptr)); if (!(predicate)) {{ "
            f'_Pragma("unroll 1") '
            f"while (1) {{ __nanosleep(backoff_ns); {name}_load((dst), (ptr)); "
            f"if (predicate) break; }} }}"
            f"{closing} }} while (0)\n"
        )
        operands = (condition, backoff_ns)
    return cuda_func_call(name, *load_call.args[1:-1], *operands, source_code=source), tags


# =============================================================================
# mbarrier waits — ``mbarrier.try_wait`` only polls once, so the body wraps it
# in a branch loop that retries until the parity flips. The magic
# ``ticks = 0x989680`` is the per-attempt timeout hint in ns.
# =============================================================================
def _mbarrier_wait_parts(barrier, *_rest):
    """Dispatch on the barrier operand's dtype, as the retired op did.

    A ``uint32`` is already a shared-window address (the caller ran cvta once
    and carries offsets in integer space); converting it again would corrupt
    it, so the raw form binds it directly. Anything else is a generic pointer
    and gets the cvta here.
    """
    raw = str(getattr(barrier, "ty", "")) == "uint32"
    return (
        ("_raw_u32" if raw else ""),
        ("(unsigned int barrier, int phase)" if raw else "(void* barrier, int phase)"),
        (
            "    unsigned int barrier_addr_int = barrier;\n"
            if raw
            else "    unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);\n"
        ),
    )


device_intrinsic(
    "cuda_mbarrier_wait",
    helper_name=lambda *a: f"tvm_builtin_cuda_mbarrier_wait{_mbarrier_wait_parts(*a)[0]}",
    c_signature=lambda *a: _mbarrier_wait_parts(*a)[1],
    body=lambda *a: (
        _mbarrier_wait_parts(*a)[2] + "    unsigned int ticks = 0x989680;\n"
        "    asm volatile(\n"
        '        "{\\n"\n'
        '        ".reg .pred                P1;\\n"\n'
        '        "LAB_WAIT:\\n"\n'
        '        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\\n"\n'
        '        "@P1                       bra.uni DONE;\\n"\n'
        '        "bra.uni                   LAB_WAIT;\\n"\n'
        '        "DONE:\\n"\n'
        '        "}\\n"\n'
        '        :: "r"(barrier_addr_int), "r"(phase), "r"(ticks) : "memory");'
    ),
)


# mbarrier.try_wait.parity.acquire.cluster — cluster-scope acquire wait used for
# cross-CTA barrier handshakes (e.g. the tmem-finished handoff).
device_intrinsic(
    "cuda_mbarrier_wait_acquire_cluster",
    c_signature="(void* barrier, int phase)",
    body=(
        "    unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);\n"
        "    asm volatile(\n"
        '        "{\\n"\n'
        '        ".reg .pred                P1;\\n"\n'
        '        "LAB_WAIT_AC:\\n"\n'
        '        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\\n"\n'
        '        "@P1                       bra.uni DONE_AC;\\n"\n'
        '        "bra.uni                   LAB_WAIT_AC;\\n"\n'
        '        "DONE_AC:\\n"\n'
        '        "}\\n"\n'
        '        :: "r"(barrier_addr_int), "r"(phase) : "memory");'
    ),
)


# =============================================================================
# Cluster / warpgroup barriers — open-coded arrive+wait and ``bar.sync``.
# =============================================================================
device_intrinsic(
    "cuda_cluster_sync",
    body=('    asm("barrier.cluster.arrive.aligned;");\n    asm("barrier.cluster.wait.aligned;");'),
)
device_intrinsic(
    "cuda_warpgroup_sync",
    c_signature="(int name_bar_id)",
    body='    asm volatile("bar.sync %0, 128;" : : "r"(name_bar_id));',
)


# =============================================================================
# mov.u32/u64 from a special register. Each (bits, reg) emits a distinct helper
# because the register name is baked into the PTX text rather than passed as an
# operand, so it cannot be a table row parameterized on its operands.
# =============================================================================


def _cuda_mov_sreg_body(bits):
    spec = "l" if bits == 64 else "r"

    def _body(reg):
        reg = parse_str(reg)
        return (
            f"    uint{bits}_t x;\n"
            f'    asm volatile("mov.u{bits} %0, %{reg};" : "={spec}"(x));\n'
            f"    return (int{bits}_t)x;"
        )

    return _body


for _bits in (32, 64):
    device_intrinsic(
        f"cuda_mov_sreg_{_bits}",
        n_attrs=1,
        helper_name=(
            lambda *a, bits=_bits: (
                f"tvm_builtin_ptx_fetch_register_"
                f"{parse_str(a[-1]).replace('::', '_').replace('.', '_')}"
            )
        ),
        return_type=f"int{_bits}_t",
        body=_cuda_mov_sreg_body(_bits),
    )
del _bits


@register_codegen("cuda_mov_sreg")
def codegen_cuda_mov_sreg(bits, reg):
    bits = int(bits)
    reg = parse_str(reg)
    if bits not in (32, 64):
        raise ValueError(f"Only support 32/64 bits for cuda_mov_sreg, but got {bits}.")
    result = CODEGEN_REGISTRY[f"tirx.cuda_mov_sreg_{bits}"]([reg])
    return result[0] if isinstance(result, tuple) else result


# =============================================================================
# wgmma_noop_barrier — empty asm with one inout register operand. Two
# device_intrinsic calls, one per supported dtype; dispatcher picks the form
# based on the operand's runtime dtype.
# =============================================================================
device_intrinsic(
    "cuda_wgmma_noop_barrier_uint32",
    helper_name="ptx_wgmma_fence_uint32_t",
    c_signature="(uint32_t reg)",
    body='    asm volatile("" : "+r"(reg) :: "memory");',
)
device_intrinsic(
    "cuda_wgmma_noop_barrier_float32",
    helper_name="ptx_wgmma_fence_float",
    c_signature="(float reg)",
    body='    asm volatile("" : "+f"(reg) :: "memory");',
)


@register_codegen("cuda_wgmma_noop_barrier")
def codegen_cuda_wgmma_noop_barrier(reg):
    dtype = str(reg.dtype)
    dtype_enum = PTXDataType.from_string(dtype)
    if dtype_enum == PTXDataType.UINT32:
        op_name = "tirx.cuda_wgmma_noop_barrier_uint32"
    elif dtype_enum == PTXDataType.FLOAT32:
        op_name = "tirx.cuda_wgmma_noop_barrier_float32"
    else:
        raise ValueError(f"Only support uint32/float32 for wgmma_fence, but got {dtype}.")
    result = CODEGEN_REGISTRY[op_name]([reg])
    return result[0] if isinstance(result, tuple) else result


# =============================================================================
# The raw cp.async op behind ``InjectPTXAsyncCopy``. All user-issued
# cp.async / cp.async.bulk copies go through ``T.ptx`` instead.
# =============================================================================
@register_codegen("s_tir_cp_async_raw")
def codegen_s_tir_cp_async_raw(*args):
    """The raw cp.async op InjectPTXAsyncCopy emits (new copies are ptx).

    Accepts two call shapes:

    * 5 args ``(dst_ptr, dst_offset, src_ptr, src_offset, cp_size)`` —
      offsets are element indices the helper scales by the buffer element
      size.
    * 6 args — the same with an explicit predicate, zero-filling the
      destination when the predicate is false.
    """
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
    else:
        raise ValueError(f"cp_async_raw codegen expects 5/6 args, got {len(args)}")
