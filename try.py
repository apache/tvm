import tvm
from tvm.script import tir as T

@T.prim_func
def vectorize_add_fp16(A: T.Buffer([128], "bfloat16"), B: T.Buffer([128], "bfloat16")) -> None:

    for i in range(128):
        with T.block("blk"):
            vi = T.axis.remap("S", [i])
            B[vi] = A[vi] + T.abs(A[vi])


sch = tvm.tir.Schedule(vectorize_add_fp16, debug_mask="all")
blk = sch.get_block("blk")
i, = sch.get_loops(blk)
io, ii, v = sch.split(i, [None, 32, 2])
sch.vectorize(v)
sch.bind(ii, "threadIdx.x")
sch.bind(io, "blockIdx.x")

print(sch.mod["main"])
f = tvm.build(sch.mod["main"], target="cuda")
print(f.imported_modules[0].get_source())