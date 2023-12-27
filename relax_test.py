import numpy as np
import tvm
from tvm.script import tir as T
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.script import ir as I
from tvm import relax

M = 1
N = 16384
K = 16384
@I.ir_module
class Module:
    @T.prim_func
    def decode(
        A: T.Buffer((T.int64(64), T.int64(64)), "int8"),
        B: T.Buffer((T.int64(128),), "float16"),
        decode_1: T.Buffer((T.int64(64), T.int64(128)), "float16"),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j in T.grid(T.int64(64), T.int64(128)):
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
        A: T.Buffer((T.int64(128), T.int64(64)), "float16"),
        w_gathered: T.Buffer((T.int64(64), T.int64(64)), "int8"),
        compute: T.Buffer((T.int64(128),), "float16"),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        max_abs_value = T.alloc_buffer((T.int64(128),), "float16")
        scale = T.alloc_buffer((T.int64(128),))
        for i, k in T.grid(T.int64(128), T.int64(64)):
            with T.block("max_abs_value"):
                v_i, v_k = T.axis.remap("SR", [i, k])
                T.reads(A[v_i, v_k])
                T.writes(max_abs_value[v_i])
                with T.init():
                    max_abs_value[v_i] = T.float16(-65504)
                max_abs_value[v_i] = T.max(max_abs_value[v_i], T.fabs(A[v_i, v_k]))
        for i in range(T.int64(128)):
            with T.block("scale"):
                v_i = T.axis.spatial(T.int64(128), i)
                T.reads(max_abs_value[v_i])
                T.writes(scale[v_i])
                scale[v_i] = T.max(
                    T.Cast("float32", max_abs_value[v_i]), T.float32(0.0001)
                ) * T.float32(0.125)
        for j, i, k in T.grid(T.int64(64), T.int64(64), T.int64(2)):
            with T.block("w_gathered"):
                v_j, v_i, v_k = T.axis.remap("SSR", [j, i, k])
                T.reads(A[v_i * T.int64(2) + v_k, v_j], scale[v_i * T.int64(2) + v_k])
                T.writes(w_gathered[v_j, v_i])
                with T.init():
                    w_gathered[v_j, v_i] = T.int8(0)
                w_gathered[v_j, v_i] = T.bitwise_or(
                    w_gathered[v_j, v_i],
                    T.if_then_else(
                        v_i * T.int64(2) + v_k < T.int64(128),
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
        for i0 in range(T.int64(128)):
            with T.block("compute"):
                v_i0 = T.axis.spatial(T.int64(128), i0)
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

x = np.random.randn(*x_shape).astype("float16")
y = np.random.normal(0, 0.002, size=y_shape).astype("int8")
tvm_x = tvm.nd.array(x, dev)
tvm_y = tvm.nd.array(y, dev)
tvm_scales = tvm.nd.array(np.ones(N, dtype="float16"), dev)
vm["main"](tvm_x, tvm_y, tvm_scales)
print(tvm_scales)