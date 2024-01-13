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


def gemv_i4(M, N, K, dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int_to_float(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)
    
    A = te.placeholder((M, K), name='A', dtype=dtype)
    B = te.placeholder((N, K // 8 * bit), name='B', dtype='int8')
    
    def decode_func(n, k):
        w = _tir_u8_to_int_to_float(bit, B[n, k // n_float_per_i8], k % n_float_per_i8, dtype=dtype)
        return w

    B_decode = te.compute(
        (N, K),
        decode_func,
        name='B_decode'
    )

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name='k')
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B_decode[j, k], axis=k),
        name='C'
    )
    func = te.create_prim_func([A, B, C]).with_attr("inconsistent", {
        'B': {
            'decode_block': 'B_decode',
            'fast_decoding': True,
            'source_format':{
                'bits': 4,
                'format': 'int',
            },
            'target_format':{
                'bits': 16,
                'format': 'float',
            }
        }
    })
    return tvm.IRModule.from_expr(func)



from tvm.script import relax as R
from tvm.script import tir as T
from tvm.script import ir as I
from tvm import relax
shapes = [
    # M, N, K
    # (1, 43008, 14336),
    # (1, 14336, 14336),
    # (1, 57344, 14336),
    # (1, 14336, 57344),
    # (16, 43008, 14336),
    # (16, 14336, 14336),
    (16, 57344, 14336),
    # (16, 14336, 57344),
    # (32, 43008, 14336),
    # (32, 14336, 14336),
    # (32, 57344, 14336),
    # (32, 14336, 57344),
    # (64, 43008, 14336),
    # (64, 14336, 14336),
    # (64, 57344, 14336),
    # (64, 14336, 57344),
    # (128, 43008, 14336),
    # (128, 14336, 14336),
    # (128, 57344, 14336),
    # (128, 14336, 57344),
    # (4096, 43008, 14336),
    # (4096, 14336, 14336),
    # (4096, 57344, 14336),
    # (4096, 14336, 57344),
    # (8192, 43008, 14336),
    # (8192, 14336, 14336),
    # (8192, 57344, 14336),
    # (8192, 14336, 57344),

]
# M = 1
# N = 16384
# K = 16384
perf_map = []
for M, N, K in shapes:
        
    @I.ir_module
    class Module:
        @T.prim_func
        def decode(
            A: T.Buffer((T.int64(K), T.int64(N//2)), "int8"),
            B: T.Buffer((T.int64(N),), "float16"),
            decode_1: T.Buffer((T.int64(K), T.int64(N)), "float16"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i, j in T.grid(T.int64(K), T.int64(N)):
                with T.block("decode"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(A[v_i, v_j // T.int64(2)], B[v_j])
                    T.writes(decode_1[v_i, v_j])
                    decode_1[v_i, v_j] = (
                        T.Cast(
                            "float16",
                            T.shift_right(
                                T.shift_left(
                                    T.bitwise_and(
                                        T.shift_right(
                                            T.Cast("int32", A[v_i, v_j // T.int64(2)]),
                                            T.Cast("int32", v_j % T.int64(2)) * 4,
                                        ),
                                        15,
                                    ),
                                    28,
                                ),
                                28,
                            ),
                        )
                        * B[v_j]
                    )

        @T.prim_func
        def encode(
            A: T.Buffer((T.int64(N), T.int64(K)), "float16"),
            w_gathered: T.Buffer((T.int64(K), T.int64(N // 2)), "int8"),
            compute: T.Buffer((T.int64(N),), "float16"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            max_abs_value = T.alloc_buffer((T.int64(N),), "float16")
            scale = T.alloc_buffer((T.int64(N),))
            for i, k in T.grid(T.int64(N), T.int64(K)):
                with T.block("max_abs_value"):
                    v_i, v_k = T.axis.remap("SR", [i, k])
                    T.reads(A[v_i, v_k])
                    T.writes(max_abs_value[v_i])
                    with T.init():
                        max_abs_value[v_i] = T.float16(-65504)
                    max_abs_value[v_i] = T.max(max_abs_value[v_i], T.fabs(A[v_i, v_k]))
            for i in range(T.int64(N)):
                with T.block("scale"):
                    v_i = T.axis.spatial(T.int64(N), i)
                    T.reads(max_abs_value[v_i])
                    T.writes(scale[v_i])
                    scale[v_i] = T.max(
                        T.Cast("float32", max_abs_value[v_i]), T.float32(0.0001)
                    ) * T.float32(0.125)
            for j, i, k in T.grid(T.int64(K), T.int64(N//2), T.int64(2)):
                with T.block("w_gathered"):
                    v_j, v_i, v_k = T.axis.remap("SSR", [j, i, k])
                    T.reads(A[v_i * T.int64(2) + v_k, v_j], scale[v_i * T.int64(2) + v_k])
                    T.writes(w_gathered[v_j, v_i])
                    with T.init():
                        w_gathered[v_j, v_i] = T.int8(0)
                    w_gathered[v_j, v_i] = T.bitwise_or(
                        w_gathered[v_j, v_i],
                        T.if_then_else(
                            v_i * T.int64(2) + v_k < T.int64(N),
                            T.shift_left(
                                T.bitwise_and(
                                    T.Cast(
                                        "int8",
                                        T.min(
                                            T.max(
                                                T.round(
                                                    T.Cast(
                                                        "float32", A[v_i * T.int64(2) + v_k, v_j]
                                                    )
                                                    / scale[v_i * T.int64(2) + v_k]
                                                ),
                                                T.float32(-8),
                                            ),
                                            T.float32(7),
                                        ),
                                    ),
                                    T.int8(15),
                                ),
                                T.Cast("int8", v_k) * T.int8(4),
                            ),
                            T.int8(0),
                        ),
                    )
            for i0 in range(T.int64(N)):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(N), i0)
                    T.reads(scale[v_i0])
                    T.writes(compute[v_i0])
                    compute[v_i0] = T.Cast("float16", scale[v_i0])

        @R.function
        def main(
            x: R.Tensor((M, K), dtype="float16"),
            y: R.Tensor((N, K), dtype="float16"),
        ) -> R.Tensor((M, N), dtype="float16"):
            R.func_attr({"num_input": 1})
            cls = Module
            with R.dataflow():
                lv = R.call_tir(
                    cls.encode,
                    (y,),
                    out_sinfo=[R.Tensor((K, N//2), dtype="int8"), R.Tensor((N,), dtype="float16")],
                )
                lv1 = lv[0]
                lv2 = R.call_pure_packed(
                    "cutlass.ft_preprocess_weight",
                    lv1,
                    R.prim_value(80),
                    R.prim_value(0),
                    sinfo_args=(R.Tensor((K, N//2), dtype="int8"),),
                )
                lv3: R.Tensor((N,), dtype="float16") = lv[1]
                lv6 = R.call_tir(
                    cls.decode, (lv2, lv3), out_sinfo=R.Tensor((K, N), dtype="float16")
                )
                lv1_1: R.Tensor((M, N), dtype="float16") = R.matmul(x, lv6, out_dtype="float16")
                R.output(lv1_1)
            return lv1_1
    x_shape = (M, K)
    y_shape = (N, K)
    mod = partition_for_cutlass(Module)
    func_names = [name.name_hint for (name, _) in mod.functions.items()]

    mod = relax.transform.RunCodegen(
                {"cutlass": {"sm": 80, "find_first_valid": False}},
                entry_functions=["main"],
    )(mod)

    mod = relax.pipeline.get_pipeline()(mod)
    mod = relax.transform.LiftTransformParams()(mod)
    def split_transform_deploy_mod(mod):
        mod_transform = tvm.IRModule()
        mod_deploy = tvm.IRModule().with_attrs(mod.attrs)

        transform_func_name = None

        for gv, func in mod.functions.items():
            if "transform_params" in gv.name_hint:
                transform_func_name = gv.name_hint
                mod_transform[gv] = func
            elif isinstance(func, tvm.tir.PrimFunc):
                mod_transform[gv] = func
            else:
                mod_deploy[gv] = func

        assert transform_func_name is not None
        return mod_transform, mod_deploy, transform_func_name

    _, mod_deploy, _ = split_transform_deploy_mod(mod)

    print(mod_deploy)
    dev = tvm.device("cuda", 0)
    ex = relax.build(mod_deploy, target="nvidia/nvidia-a100")
    vm = relax.vm.VirtualMachine(ex, dev)
    tvm_a = tvm.nd.array(np.random.uniform(0, 1, [M, K]).astype("float16"), device=dev)
    tvm_b = tvm.nd.array(np.random.randint(-127, 128, [K, N//2]).astype("int8"), device=dev)
    tvm_scales = tvm.nd.array(np.ones(N, dtype="float16"), dev)
    vm_output =vm["main"](tvm_a, tvm_b, tvm_scales)
    print(vm_output)

    # warm up
    for i in range(5):
        vm["main"](tvm_a, tvm_b, tvm_scales)

    iters = 10
    dev.sync()
    start = time.time()
    for i in range(iters):
        vm["main"](tvm_a, tvm_b, tvm_scales)
    dev.sync()
    end = time.time()

    cost = (end - start) / iters * 1000
    print("cost: ", cost)
    key = f"{M}_{N}_{K}"
    perf_map.append((key, cost))
    
for key, cost in perf_map:
    print(f"{key}: {cost}")