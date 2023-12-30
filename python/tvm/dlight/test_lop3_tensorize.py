import numpy as np
import tvm
import time
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import DefaultPolicy
from tvm.dlight.base.roller.policy.default import PrimFuncNode
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import ElementWise, GeneralReduction, GEMV
from tvm.dlight.gpu import Fallback
from tvm.dlight.base.utils import apply_and_build_parallel, apply_and_build
from tvm import te
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass


def lop3_decode_i4(N, K, dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int_to_float(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)
    
    B = te.placeholder((N, K // 8 * bit), name='B', dtype='int8')
    
    def decode_func(n, k):
        w = _tir_u8_to_int_to_float(bit, B[n, k // n_float_per_i8], k % n_float_per_i8, dtype=dtype)
        return w

    B_decode = te.compute(
        (N, K),
        decode_func,
        name='B_decode'
    )
    func = te.create_prim_func([B, B_decode])
    return tvm.IRModule.from_expr(func)

N = 1
K = 8
benchmark_results = {}
ir_module = lop3_decode_i4(N, K, "float16")

sch = tvm.tir.Schedule(ir_module)
block_b = sch.get_block("B_decode")
j, k = sch.get_loops(block_b)
sch.bind(j, 'threadIdx.x')
block_cache_b = sch.cache_read(block_b, 0, 'local')
block_decode_fp16 = sch.cache_write(block_b, 0, 'local')
sch.compute_at(block_cache_b, j)
sch.reverse_compute_at(block_decode_fp16, j)
from tvm.dlight.gpu.intrin.lop3 import (
    LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN
)
decode_i4_to_f16 = """
template <typename T1, typename T2>
__device__ void decode_i4s_to_f16(T1 *_i4s, T2* B_local_decode, const int N = 8) {
  uint* h = reinterpret_cast<uint*>(B_local_decode);
  
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint BOTTOM_MASK = 0x000f000f;
  static constexpr uint TOP_MASK = 0x00f000f0;
  static constexpr uint I4s_TO_F16s_MAGIC_NUM = 0x64006400;
  static constexpr uint FP16_TOP_MAGIC_NUM = 0x64006400;
  static constexpr uint ONE_SIXTEENTH = 0x2c002c00;
  static constexpr uint NEG_64 = 0xd400d400;
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
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\\n" : "=r"(h[i * 2 + 1]) : "r"(h[i * 2 + 1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
  }
}

"""
sch.annotate(block_cache_b, ann_key="pragma_import_c", ann_val=decode_i4_to_f16)
sch.tensorize(sch.get_loops(block_b)[-1], LOP3_FAST_DECODE_INT4_TO_FP16_INTRIN)
print(sch.mod)

target = tvm.target.Target("nvidia/nvidia-a100")
mod = tvm.build(sch.mod["main"], target=target)

int4_data = [1, 2, 3, 4, 5, 6, 7, 8]
def compress_int8(int4_data):
    int8_data = []
    for i in range(0, len(int4_data), 2):
        int8_data.append(int4_data[i] + (int4_data[i+1] << 4))
    return int8_data

def decompress_int8(int8_data):
    int4_data = []
    for i in range(len(int8_data)):
        int4_data.append(int8_data[i] & 0x0f)
        int4_data.append((int8_data[i] >> 4) & 0x0f)
    return int4_data

int8_data = compress_int8(int4_data)
d_int4_data = decompress_int8(int8_data)
print(int4_data)
print(int8_data)
print(d_int4_data)

def interleave_weight(qweight):
    # reinterpret the data type of qweight to int32
    qweight = qweight.view(np.int32)
     
    new_qweight = np.zeros_like(qweight)
    new_qweight |= (qweight & 0x0000000f)
    new_qweight |= (qweight & 0x000000f0) << 12
    new_qweight |= (qweight & 0x00000f00) >> 4
    new_qweight |= (qweight & 0x0000f000) << 8
    new_qweight |= (qweight & 0x000f0000) >> 8
    new_qweight |= (qweight & 0x00f00000) << 4
    new_qweight |= (qweight & 0x0f000000) >> 12
    new_qweight |= (qweight & 0xf0000000)
    return new_qweight.view(np.int8)

permutated_b = interleave_weight(np.array(int8_data, dtype=np.int8))

ctx = tvm.cuda()
tvm_b = tvm.nd.array(np.array(permutated_b, dtype=np.int8).reshape((N, K//8*4)), device=ctx)
tvm_b_decode = tvm.nd.array(np.zeros((N, K), dtype=np.float16), device=ctx)

mod(tvm_b, tvm_b_decode)

print(tvm_b_decode.asnumpy())