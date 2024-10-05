import tvm
from tvm import tl

def _tir_packed_to_unsigned_convert(storage_type="uint", storage_nbit=8):
    storage_dtype = storage_type + str(storage_nbit)

    def f_convert(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == storage_dtype, f"{val.dtype} != {storage_dtype}"
        mask = tvm.tir.const((1 << nbit) - 1, storage_dtype)
        return ((val >> (pos * nbit).astype(storage_dtype)) & mask).astype(dtype)

    return f_convert

def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
    num_bits=4,
):
    num_elems_per_byte = 8 // num_bits
    storage_dtype = "int8"
    A_shape = (M, K)
    B_shape = (N, K // num_elems_per_byte)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K // num_elems_per_byte)
    B_dequantize_shared_shape = (block_N, block_K)

    import tvm.tl.language as T

    @T.prim_func
    def main(
            A: T.Buffer(A_shape, in_dtype),
            B: T.Buffer(B_shape, storage_dtype),
            C: T.Buffer((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
            B_local = T.alloc_local([8], storage_dtype)
            B_dequantize_local = T.alloc_local([16], in_dtype)
            B_dequantize_shared = T.alloc_shared(B_dequantize_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            tx = T.thread_binding(0, threads, thread="threadIdx.x")

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)

                # for i, j in T.Parallel(block_N, block_K // num_elems_per_byte):
                #     B_shared[i, j] = B[bx * block_N + i, k * block_K // num_elems_per_byte + j]

                for i in T.serial(block_N * block_K // num_elems_per_byte // (threads * 4)):
                    for v in T.vectorized(0, 4):
                        vi = (i * threads * 4 + tx * 4 + v) // (block_K // num_elems_per_byte)
                        vj = (i * threads * 4 + tx * 4 + v) % (block_K // num_elems_per_byte)
                        B_local[v] = B_shared[vi, vj]
                    for v in T.serial(0, 8):
                        B_dequantize_local[v] = _tir_packed_to_unsigned_convert("int", 8)(
                            num_bits,
                            B_local[v // 2],
                            v % 2,
                            dtype=in_dtype,
                        )
                    for v in T.vectorized(0, 8):
                        vi = (i * threads * 8 + tx * 8 + v) // (block_K)
                        vj = (i * threads * 8 + tx * 8 + v) % (block_K)
                        B_dequantize_shared[vi, vj] = B_dequantize_local[v]
                T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm(
    M,
    N,
    K,
    dtypeAB,
    dtypeC,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=3,
    num_threads=128,
):
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        dtypeAB,
        dtypeC,
        dtypeAccum,
        num_stages,
        num_threads,
    )
    print(program)

    mod, params = tl.lower(program)
    mod = tl.Profiler(mod, params, [2], tl.TensorSupplyType.Integer)

    out = mod.run_once()

    print(f"output is {out}")

    def ref_program(A, qB):
        import torch

        B = (
            torch.zeros(qB.shape[0], qB.shape[1] * 8 // 4,
                        dtype=torch.half).to(torch.half).to(A.device))
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B[i][j] = ((qB[i][j // 2] >> (4 * (j % 2))) & 0xF).to(torch.half)
        C = torch.matmul(A.to(torch.float), B.T.to(torch.float))
        C = C.to(torch.__getattribute__(dtypeC))
        return C

    mod.assert_allclose(ref_program)


def test_run_dequantize_gemm():
    run_gemm(256, 256, 256, "int8", "int32", "int32", 128, 128, 64, num_threads=128)
    # run_gemm(256, 256, 256, "float16", "float16", "float32", 128, 128, 64, num_threads=128)


if __name__ == "__main__":
    # bitblas.testing.main()
    test_run_dequantize_gemm()
