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

from tvm.backend.cuda.op import cuda_func_call

from ..codegen.registry import CODEGEN_REGISTRY, register_codegen
from ..codegen.schema import device_intrinsic
from ..codegen.types import PTXDataType
from ..codegen.utils import parse_str


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
