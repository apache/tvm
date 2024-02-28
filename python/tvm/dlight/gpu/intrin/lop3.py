import tvm
from tvm.runtime import convert
from tvm.tir.function import TensorIntrin
from tvm.script import tir as T
import numpy as np
from typing import Dict, Optional, Literal

lift = convert

decode_i4_to_f16 = """
template <typename T1, typename T2>
__device__ void decode_i4s_to_f16(T1 *_i4s, T2 *B_local_decode, const int N = 8)
{
  uint *h = reinterpret_cast<uint *>(B_local_decode);

  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint BOTTOM_MASK = 0x000f000f;
  static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
  uint const i4s = *reinterpret_cast<uint *>(_i4s);
#pragma unroll
  // decode 2 elems at one time.
  for (int i = 0; i < (N / 2); i++)
  {

    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                 : "=r"(h[i])
                 : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(FP16_TOP_MAGIC_NUM), "n"(immLut));
    asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i]) : "r"(h[i]), "r"(FP16_TOP_MAGIC_NUM));
  }
}
"""


decode_i1s_to_i8s_l16 = """template <typename T1, typename T2>
__device__ void decode_i1s_to_i8s_l16(T1 *_i1s, T2 *_i8s, const int N = 16)
{
  int *i8s = reinterpret_cast<int *>(_i8s);
  int16_t i1s_i16 = *reinterpret_cast<int16_t *>(_i1s);
  // permutate: {e0,e4,e8,e12,e2,e6,e10,e14,e1,e5,e9,e13,e3,e7,e11,e15}
  // into: {e0,e4,e8,e12,x,x,x,x,e1,e5,e9,x,x,x,x,e13,e2,e6,e10,e14,e1,e5,e9,e13,e3,e7,e11,e15,x,x,x,x}
  int i1s = (i1s_i16 & 0x0f0f);
  i1s |= ((i1s_i16 & 0xf0f0) << 12); 
  // i1s        {0..,e15,e14,e13,e12,e11,e10,e9,e8,e7,e6,e5,e4,e3,e2,e1,e0}
  // interleave {0..,e15,e13,e11,e9,e7,e5,e3,e1,e14,e12,e10,e8,e6,e4,e2,e0}
  // First, we extract the i1s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa; // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x01010101;      // 0x1 -> 0b01 select 0,1
  static constexpr uint I8s_MAGIC_NUM = 0x00000000;

  for (int i = 0; i < N / 4; i++)
  {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                 : "=r"(i8s[i])
                 : "r"(i1s >> i), "n"(BOTTOM_MASK), "n"(I8s_MAGIC_NUM), "n"(immLut));
  }
}
"""

decode_i2s_to_i8s = """template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2s, T2 *_i8s, const int N = 16)
{
  // convert 8 int2b_t to 8 int8b_t -> 2 int32
  uint *i8s = reinterpret_cast<uint *>(_i8s);

  // i2s = {e7,e6,e5,e4,e3,e2,e1,e0}
  // also require interleave {e7,e3,e6,e2,e5,e1,e4,e0}
  uint const i2s = *_i2s;

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;     // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x03030303;          // 0xf -> 0b11 select 0,3
  static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; // 1024

#pragma unroll
  for (int i = 0; i < (N / 2); i++)
  {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                 : "=r"(i8s[i])
                 : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
  }
}
"""

decode_i4s_to_i8s = """template <typename T1, typename T2>
__device__ void decode_i4s_to_i8s(T1 *_i4s, T2 *_i8s, const int N = 8)
{
  uint *i8s = reinterpret_cast<uint *>(_i8s);
  uint i4s = *_i4s;
  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;     // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x0f0f0f0f;          // 0xf -> 0b1111 select 0,4
  static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; // 1024

#pragma unroll
  for (int i = 0; i < (N / 4); i++)
  {
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                 : "=r"(i8s[i])
                 : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
  }
}
"""


def get_fast_decode_intrin(
    storage_nbit=4, storage_dtype="int8", target_dtype="float16", loops_extent=8
):
    """
    loops extent is the number of elements to be decoded in one stage
    for memory friendly process, the loops_extent should be a multiple of (sizeof(int) // 8).
    However, for the case of int1b, it is not possible to decode 8 elements in one stage, so we have to use 16.
    """
    if target_dtype == "float16":
        d4f = "f16"
    elif target_dtype == "int8":
        d4f = "i8s"
    else:
        raise ValueError("Unsupported target dtype: {}".format(target_dtype))
    func_name = "decode_i{}s_to_{}".format(storage_nbit, d4f)

    def _tir_u8_to_int_to_float(
        nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str
    ):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)

    assert storage_dtype == "int8"
    elem_per_i8 = 8 // storage_nbit
    n_storage_elems = loops_extent // elem_per_i8

    @T.prim_func
    def fast_decode_desc(compressed: T.handle, decompressed: T.handle) -> None:
        Compressed = T.match_buffer(
            compressed,
            [
                n_storage_elems,
            ],
            dtype=storage_dtype,
            scope="local",
        )
        Decompressed = T.match_buffer(
            decompressed,
            [
                loops_extent,
            ],
            dtype=target_dtype,
            scope="local",
        )

        with T.block("root"):
            T.reads(Compressed[0:n_storage_elems])
            T.writes(Decompressed[0:loops_extent])
            for i in T.grid(loops_extent):
                with T.block("decode"):
                    vi = T.axis.remap("S", [i])
                    Decompressed[vi] = _tir_u8_to_int_to_float(
                        storage_nbit,
                        Compressed[vi // elem_per_i8],
                        vi % elem_per_i8,
                        dtype=target_dtype,
                    )

    @T.prim_func
    def fast_decode_impl(compressed: T.handle, decompressed: T.handle) -> None:
        Compressed = T.match_buffer(
            compressed,
            [
                n_storage_elems,
            ],
            dtype=storage_dtype,
            scope="local",
        )
        Decompressed = T.match_buffer(
            decompressed,
            [
                loops_extent,
            ],
            dtype=target_dtype,
            scope="local",
        )

        with T.block("root"):
            T.reads(Compressed[0:n_storage_elems])
            T.writes(Decompressed[0:loops_extent])
            T.call_extern("handle", func_name, Compressed.data, Decompressed.data, loops_extent)

    return fast_decode_desc, fast_decode_impl


LOP3_FAST_DECODE_INT4_TO_FP16_L8_INTRIN = "lop3_fast_decode_i4_to_f16_l8_"
TensorIntrin.register(
    LOP3_FAST_DECODE_INT4_TO_FP16_L8_INTRIN,
    *get_fast_decode_intrin(
        storage_nbit=4, storage_dtype="int8", target_dtype="float16", loops_extent=8
    ),
)

LOP3_FAST_DECODE_INT4_TO_INT8_L8_INTRIN = "lop3_fast_decode_i4_to_i8_l8_"
TensorIntrin.register(
    LOP3_FAST_DECODE_INT4_TO_INT8_L8_INTRIN,
    *get_fast_decode_intrin(
        storage_nbit=4, storage_dtype="int8", target_dtype="int8", loops_extent=8
    ),
)


LOP3_FAST_DECODE_INT2_TO_INT8_L16_INTRIN = "lop3_fast_decode_i2_to_i8_l16_"
TensorIntrin.register(
    LOP3_FAST_DECODE_INT2_TO_INT8_L16_INTRIN,
    *get_fast_decode_intrin(
        storage_nbit=2, storage_dtype="int8", target_dtype="int8", loops_extent=16
    ),
)

LOP3_FAST_DECODE_INT1_TO_INT8_L16_INTRIN = "lop3_fast_decode_int1_to_i8_l16_"
TensorIntrin.register(
    LOP3_FAST_DECODE_INT1_TO_INT8_L16_INTRIN,
    *get_fast_decode_intrin(
        storage_nbit=1, storage_dtype="int8", target_dtype="int8", loops_extent=16
    ),
)


def get_lop3_intrin_group(
    in_dtype: Literal["int8"],
    out_dtype: Literal["float16", "int8"],
    storage_nbit: int = 4,
) -> Dict[str, str]:
    """
    This function is used to get the intrinsic group of the LOP3 operation to avoid the overhead of fast decoding.
    LOP3 is a type of logic operation that takes three inputs. The intrinsic group refers to the set of
    intrinsic operations that can be performed on these inputs. This function retrieves and returns this group.

    Parameters
    ----------
    in_dtype : Literal["int8"]
        The data type of the input. It should be "int8".

    out_dtype : Literal["float16", "int8"]
        The data type of the output. It can be either "float16" or "int8".

    storage_nbit : int, optional
        The number of bits used for storage. By default, it is 4.

    with_scale : bool, optional
        A boolean parameter that indicates whether scaling should be applied. By default, it is False.

    Returns
    -------
    Dict[str, str]
        A dictionary mapping the names of the intrinsics to their corresponding implementations.
    """
    assert in_dtype in ["int8"]
    assert out_dtype in ["float16", "int8"]

    dtype_mapping = {"float16": "f16", "int8": "i8", "int32": "i32"}
    target_dtype = dtype_mapping[out_dtype]
    target_bits = tvm.DataType(out_dtype).bits
    loop_extent = min(128 // target_bits, 32 // storage_nbit)
    _intrin = f"lop3_fast_decode_i{storage_nbit}_to_{target_dtype}_l{loop_extent}_"
    import_c_map = {
        "i4_to_f16": decode_i4_to_f16,
        "i1_to_i8": decode_i1s_to_i8s_l16,
        "i2_to_i8": decode_i2s_to_i8s,
        "i4_to_i8": decode_i4s_to_i8s,
    }
    return {
        "c_source": import_c_map[f"i{storage_nbit}_to_{target_dtype}"],
        "compute": _intrin,
    }


# interleave weight numpy code
def interleave_weight(qweight, nbits=4, target_dtype="float16"):
    assert target_dtype in ["float16", "int8"]
    # reinterpret the data type of qweight to int32
    qweight = qweight.view(np.int32)
    new_qweight = np.zeros_like(qweight)
    bits_stride = 8 if target_dtype == "int8" else 16
    mask = (1 << nbits) - 1  # for 4bit the val is 0x0000000f
    num_groups = 32 // bits_stride
    elems_per_group = bits_stride // nbits
    for i in range(num_groups):
        for j in range(elems_per_group):
            offset = i * elems_per_group + j
            shift = (offset % num_groups) * bits_stride + (offset // num_groups) * nbits
            new_qweight |= ((qweight >> (nbits * offset)) & mask) << shift

    if nbits == 1 and target_dtype == "int8":
        # special handling for 1b interleave
        n16_weight = new_qweight & np.int32(0xF0F00F0F)
        n16_weight |= ((new_qweight & np.int32(0x000000F0)) >> 4) << 16
        n16_weight |= ((new_qweight & np.int32(0x0000F000)) >> 12) << 24
        n16_weight |= ((new_qweight & np.int32(0x000F0000)) >> 16) << 4
        n16_weight |= ((new_qweight & np.int32(0x0F000000)) >> 24) << 12
        return n16_weight.view(np.int8)
    elif nbits == 2 and target_dtype == "float16":
        n8_weight = new_qweight & np.int32(0xFF0000FF)
        n8_weight |= ((new_qweight & np.int32(0x0000FF00)) >> 8) << 16
        n8_weight |= ((new_qweight & np.int32(0x00FF0000)) >> 16) << 8
        return n8_weight.view(np.int8)
    elif nbits == 1 and target_dtype == "float16":
        n8_weight = new_qweight & 0xF000000F
        n8_weight |= ((new_qweight & 0x000000F0) >> 4) << 8
        n8_weight |= ((new_qweight & 0x00000F00) >> 8) << 16
        n8_weight |= ((new_qweight & 0x0000F000) >> 12) << 24
        n8_weight |= ((new_qweight & 0x000F0000) >> 16) << 4
        n8_weight |= ((new_qweight & 0x00F00000) >> 20) << 12
        n8_weight |= ((new_qweight & 0x0F000000) >> 24) << 20

    return new_qweight.view(np.int8)
