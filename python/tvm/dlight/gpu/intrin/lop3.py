import tvm
from tvm.runtime import convert
from tvm.tir.function import TensorIntrin
from tvm.script import tir as T

lift = convert

decode_i4_to_f16 = """
template <typename T1, typename T2>
__device__ void decode_i4s_to_f16(T1 *_i4s, T2* B_local_decode, const int N = 8) {
  uint* h = reinterpret_cast<uint*>(B_local_decode);
  
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint BOTTOM_MASK = 0x000f000f;
  static constexpr uint TOP_MASK = 0x00f000f0;
  static constexpr uint I4s_TO_F16s_MAGIC_NUM = 0x64006400;
  static constexpr uint FP16_TOP_MAGIC_NUM = 0x64086408;
  static constexpr uint ONE_SIXTEENTH = 0x2c002c00;
  static constexpr uint NEG_72 = 0xd480d480;
  uint const i4s = *reinterpret_cast<uint *>(_i4s);
#pragma unroll
  for (int i = 0; i < (N / 4); i++)
  {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                 : "=r"(h[i * 2 + 0])
                 : "r"(i4s >> (8 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                 : "=r"(h[i * 2 + 1])
                 : "r"(i4s >> (8 * i)), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    asm volatile("sub.f16x2 %0, %1, %2;\\n" : "=r"(h[i * 2 + 0]) : "r"(h[i * 2 + 0]), "r"(FP16_TOP_MAGIC_NUM));
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i * 2 + 1]) : "r"(h[i * 2 + 1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
  }
}

"""

lop3_import_c = decode_i4_to_f16

def get_fast_decode_intrin(storage_nbit=4, storage_dtype="int8", target_dtype="float16", loops_extent=8):
    if target_dtype == "float16":
        d4f = "f16"
    elif target_dtype == "int8":
        d4f = "i8s"
    else:
        raise ValueError("Unsupported target dtype: {}".format(target_dtype))
    func_name = "decode_i{}s_to_{}".format(storage_nbit, d4f)
    def _tir_u8_to_int_to_float(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)
    
    assert storage_dtype == "int8"
    elem_per_i8 = 8 // storage_nbit
    n_storage_elems = loops_extent // elem_per_i8
    @T.prim_func
    def fast_decode_desc(compressed: T.handle, decompressed: T.handle) -> None:
        Compressed = T.match_buffer(
            compressed, [n_storage_elems,], dtype=storage_dtype, scope="local"
        )
        Decompressed = T.match_buffer(
            decompressed, [loops_extent,], dtype=target_dtype, scope="local"
        )
    
        with T.block("root"):
            T.reads(Compressed[0:n_storage_elems])
            T.writes(Decompressed[0:loops_extent])
            for i in T.grid(loops_extent):
                with T.block("decode"):
                    vi = T.axis.remap("S", [i])
                    Decompressed[vi] = _tir_u8_to_int_to_float(storage_nbit, Compressed[vi // elem_per_i8], vi % elem_per_i8, dtype=target_dtype)

    @T.prim_func
    def fast_decode_impl(compressed: T.handle, decompressed: T.handle) -> None:
        Compressed = T.match_buffer(
            compressed, [n_storage_elems,], dtype=storage_dtype, scope="local"
        )
        Decompressed = T.match_buffer(
            decompressed, [loops_extent,], dtype=target_dtype, scope="local"
        )
    
        with T.block("root"):
            T.reads(Compressed[0:n_storage_elems])
            T.writes(Decompressed[0:loops_extent])
            T.call_extern("handle", func_name, Compressed.data, Decompressed.data, loops_extent)
    
    return fast_decode_desc, fast_decode_impl

LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN = "lop3_fast_decode_int4_to_fp16"
TensorIntrin.register(
    LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN,
    *get_fast_decode_intrin(storage_nbit=4, storage_dtype="int8", target_dtype="float16"),
)
