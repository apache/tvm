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

#  type: ignore

from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T
import tvm
from tvm import relax
from tvm.ir import assert_structural_equal
import tvm.testing


@I.ir_module
class MLP:
    I.module_attrs({"device_num": 10})
    I.module_global_infos(
        {
            "mesh": [
                R.device_mesh((2,), I.Range(0, 2)),  # mesh[0]
                R.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
            ]
        }
    )

    @R.function
    def foo(
        x: R.Tensor((128, 128), "float32"),
        weight1: R.Tensor((128, 128), "float32"),
        weight2: R.Tensor((128, 128), "float32"),
    ) -> R.Tensor((128, 128), "float32"):
        lv0 = R.matmul(x, weight1)
        lv1 = R.nn.gelu(lv0)
        lv2 = R.dist.annotate_sharding(lv1, device_mesh="mesh[0]", placement="S[1]")
        lv3 = R.matmul(lv2, weight2)
        return lv3


@I.ir_module
class ShardedMLP:
    I.module_attrs({"device_num": 10})
    I.module_global_infos(
        {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
    )

    @R.function
    def foo(
        x: R.DTensor((128, 128), "float32", "mesh[0]", "R"),
        weight1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]"),
        weight2: R.DTensor((128, 128), "float32", "mesh[0]", "S[0]"),
    ) -> R.DTensor((128, 128), "float32", "mesh[0]", "R"):
        lv0: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.matmul(
            x, weight1, out_dtype="void"
        )
        lv1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.nn.gelu(lv0)
        lv2: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = lv1
        lv3: R.DTensor((128, 128), "float32", "mesh[0]", "R") = R.matmul(
            lv2, weight2, out_dtype="void"
        )
        return lv3


@I.ir_module
class PipelineMLP:
    I.module_attrs({"device_num": 10})
    I.module_global_infos(
        {
            "mesh": [
                R.device_mesh((2,), I.Range(0, 2)),  # mesh[0]
                R.device_mesh((2,), I.Range(4, 6)),  # mesh[1]
            ]
        }
    )

    @R.function
    def foo(
        x: R.Tensor((128, 128), "float32"),
        weight1: R.Tensor((128, 128), "float32"),
        weight2: R.Tensor((128, 128), "float32"),
        weight3: R.Tensor((128, 128), "float32"),
        weight4: R.Tensor((128, 128), "float32"),
    ) -> R.Tensor((128, 128), "float32"):
        lv0 = R.matmul(x, weight1)
        lv1 = R.nn.gelu(lv0)
        lv2 = R.dist.annotate_sharding(lv1, device_mesh="mesh[0]", placement="S[1]")
        lv3 = R.matmul(lv2, weight2)
        lv4 = R.dist.annotate_sharding(lv3, device_mesh="mesh[1]", placement="R")
        lv5 = R.matmul(lv4, weight3)
        lv6 = R.nn.gelu(lv5)
        lv7 = R.dist.annotate_sharding(lv6, device_mesh="mesh[1]", placement="S[1]")
        lv8 = R.matmul(lv7, weight4)
        return lv8


@I.ir_module
class ShardedPipelineMLP:
    I.module_attrs({"device_num": 10})
    I.module_global_infos(
        {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((2,), I.Range(4, 6))]}
    )

    @R.function
    def foo(
        x: R.DTensor((128, 128), "float32", "mesh[0]", "R"),
        weight1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]"),
        weight2: R.DTensor((128, 128), "float32", "mesh[0]", "S[0]"),
        weight3: R.DTensor((128, 128), "float32", "mesh[1]", "S[1]"),
        weight4: R.DTensor((128, 128), "float32", "mesh[1]", "S[0]"),
    ) -> R.DTensor((128, 128), "float32", "mesh[1]", "R"):
        lv0: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.matmul(
            x, weight1, out_dtype="void"
        )
        lv1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.nn.gelu(lv0)
        lv2: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = lv1
        lv3: R.DTensor((128, 128), "float32", "mesh[0]", "R") = R.matmul(
            lv2, weight2, out_dtype="void"
        )
        lv4: R.DTensor((128, 128), "float32", "mesh[1]", "R") = R.dist.redistribute(
            lv3, device_mesh="mesh[1]", placement="R"
        )
        lv5: R.DTensor((128, 128), "float32", "mesh[1]", "S[1]") = R.matmul(
            lv4, weight3, out_dtype="void"
        )
        lv6: R.DTensor((128, 128), "float32", "mesh[1]", "S[1]") = R.nn.gelu(lv5)
        lv7: R.DTensor((128, 128), "float32", "mesh[1]", "S[1]") = lv6
        lv8: R.DTensor((128, 128), "float32", "mesh[1]", "R") = R.matmul(
            lv7, weight4, out_dtype="void"
        )
        return lv8


@I.ir_module
class MLPWithConst:
    I.module_attrs({"device_num": 10})
    I.module_global_infos(
        {
            "mesh": [
                R.device_mesh((2,), I.Range(0, 2)),  # mesh[0]
                R.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
            ]
        }
    )

    @R.function
    def foo(
        x: R.Tensor((128, 128), "float32"),
        weight1: R.Tensor((128, 128), "float32"),
        weight2: R.Tensor((128, 128), "float32"),
    ) -> R.Tensor((128, 128), "float32"):
        lv0 = R.matmul(x, weight1)
        lv1 = R.nn.gelu(lv0)
        lv2 = R.add(lv1, R.const(2, "float32"))
        lv3 = R.dist.annotate_sharding(lv2, device_mesh="mesh[0]", placement="S[1]")
        lv4 = R.matmul(lv3, weight2)
        return lv4


@I.ir_module
class ShardedMLPWithConst:
    I.module_attrs({"device_num": 10})
    I.module_global_infos(
        {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
    )

    @R.function
    def foo(
        x: R.DTensor((128, 128), "float32", "mesh[0]", "R"),
        weight1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]"),
        weight2: R.DTensor((128, 128), "float32", "mesh[0]", "S[0]"),
    ) -> R.DTensor((128, 128), "float32", "mesh[0]", "R"):
        lv0: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.matmul(
            x, weight1, out_dtype="void"
        )
        lv1: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.nn.gelu(lv0)
        lv2: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = R.add(
            lv1, R.dist.const(2, R.DTensor((), "float32", "mesh[0]", "R"))
        )
        lv3: R.DTensor((128, 128), "float32", "mesh[0]", "S[1]") = lv2
        lv4: R.DTensor((128, 128), "float32", "mesh[0]", "R") = R.matmul(
            lv3, weight2, out_dtype="void"
        )
        return lv4


@I.ir_module
class MLPDynamicShape:
    I.module_attrs({"device_num": 10})
    I.module_global_infos(
        {
            "mesh": [
                R.device_mesh((2,), I.Range(0, 2)),  # mesh[0]
                R.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
            ]
        }
    )

    @R.function
    def foo(
        x: R.Tensor(("m", "k0"), "float32"),
        weight1: R.Tensor(("k0", "k1"), "float32"),
        weight2: R.Tensor(("k1", "n"), "float32"),
    ) -> R.Tensor(("m", "n"), "float32"):
        lv0 = R.matmul(x, weight1)
        lv1 = R.nn.gelu(lv0)
        lv2 = R.dist.annotate_sharding(lv1, device_mesh="mesh[0]", placement="S[1]")
        lv3 = R.matmul(lv2, weight2)
        return lv3


@I.ir_module
class ShardedMLPDynamicShape:
    I.module_attrs({"device_num": 10})
    I.module_global_infos(
        {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
    )

    @R.function
    def foo(
        x: R.DTensor(("m", "k0"), "float32", "mesh[0]", "R"),
        weight1: R.DTensor(("k0", "k1"), "float32", "mesh[0]", "S[1]"),
        weight2: R.DTensor(("k1", "n"), "float32", "mesh[0]", "S[0]"),
    ) -> R.DTensor(("m", "n"), "float32", "mesh[0]", "R"):
        m = T.int64()
        n = T.int64()
        k0 = T.int64()
        k1 = T.int64()
        lv0: R.DTensor((m, k1), "float32", "mesh[0]", "S[1]") = R.matmul(
            x, weight1, out_dtype="void"
        )
        lv1: R.DTensor((m, k1), "float32", "mesh[0]", "S[1]") = R.nn.gelu(lv0)
        lv2: R.DTensor((m, k1), "float32", "mesh[0]", "S[1]") = lv1
        lv3: R.DTensor((m, n), "float32", "mesh[0]", "R") = R.matmul(lv2, weight2, out_dtype="void")
        return lv3


# only have static shape support for now
@I.ir_module
class LlamaAttentionLayer:
    I.module_attrs({"device_num": 10})
    I.module_global_infos(
        {
            "mesh": [
                R.device_mesh((2,), I.Range(0, 2)),  # mesh[0]
                R.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
            ]
        }
    )

    @T.prim_func
    def rms_norm(var_A: T.handle, B: T.Buffer((T.int64(4096),), "float16"), var_rms_norm: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})

        A = T.match_buffer(var_A, (T.int64(1), 256, T.int64(4096)), "float16")
        rms_norm_1 = T.match_buffer(var_rms_norm, (T.int64(1), 256, T.int64(4096)), "float16")
        # with T.block("root"):
        Ared_temp = T.alloc_buffer((T.int64(1), 256))
        for bsz, i, k in T.grid(T.int64(1), 256, T.int64(4096)):
            with T.block("Ared_temp"):
                v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
                T.reads(A[v_bsz, v_i, v_k])
                T.writes(Ared_temp[v_bsz, v_i])
                with T.init():
                    Ared_temp[v_bsz, v_i] = T.float32(0)
                Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast(
                    "float32", A[v_bsz, v_i, v_k]
                ) * T.Cast("float32", A[v_bsz, v_i, v_k])
        for bsz, i, k in T.grid(T.int64(1), 256, T.int64(4096)):
            with T.block("rms_norm"):
                v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
                T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
                T.writes(rms_norm_1[v_bsz, v_i, v_k])
                rms_norm_1[v_bsz, v_i, v_k] = T.Cast(
                    "float16",
                    T.Cast("float32", B[v_k])
                    * (
                        T.Cast("float32", A[v_bsz, v_i, v_k])
                        / T.sqrt(
                            Ared_temp[v_bsz, v_i] * T.float32(0.000244140625)
                            + T.float32(9.9999999999999995e-07)
                        )
                    ),
                )

    @T.prim_func
    def rotary_embedding(
        var_A: T.handle,
        B: T.Buffer((T.int64(2048), T.int64(128)), "float16"),
        C: T.Buffer((T.int64(2048), T.int64(128)), "float16"),
        var_rotary: T.handle,
    ):
        T.func_attr({"tir.noalias": T.bool(True)})

        A = T.match_buffer(var_A, (T.int64(1), 256, T.int64(32), T.int64(128)), "float16")
        rotary = T.match_buffer(var_rotary, (T.int64(1), 256, T.int64(32), T.int64(128)), "float16")
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), 256, T.int64(32), T.int64(128)):
            with T.block("rotary"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(
                    B[256 + v_i1 - 256, v_i3],
                    A[v_i0, v_i1, v_i2, v_i3 - T.int64(64) : v_i3 - T.int64(64) + T.int64(129)],
                    C[256 + v_i1 - 256, v_i3],
                )
                T.writes(rotary[v_i0, v_i1, v_i2, v_i3])
                rotary[v_i0, v_i1, v_i2, v_i3] = B[256 + v_i1 - 256, v_i3] * A[
                    v_i0, v_i1, v_i2, v_i3
                ] + C[256 + v_i1 - 256, v_i3] * T.Select(
                    T.int64(64) <= v_i3,
                    A[v_i0, v_i1, v_i2, v_i3 - T.int64(64)],
                    A[v_i0, v_i1, v_i2, v_i3 + T.int64(64)] * T.float16(-1),
                )

    @R.function(pure=False)
    def foo(
        input_tokens: R.Tensor((1, 256, 4096), dtype="float16"),
        mask: R.Tensor((1, 1, 256, 256), dtype="float16"),
        div_const: R.Tensor((1, 32, 256, 256), dtype="float16"),
        maximum_const: R.Tensor((1, 32, 256, 256), dtype="float16"),
        kv_cache: R.Tuple(R.Object, R.Object),
        linear_weight: R.Tensor((4096, 4096), dtype="float16"),
        linear_weight1: R.Tensor((4096, 4096), dtype="float16"),
        linear_weight2: R.Tensor((4096, 4096), dtype="float16"),
        linear_weight3: R.Tensor((4096, 4096), dtype="float16"),
        rms_norm_weight: R.Tensor((4096,), dtype="float16"),
        cos_cached: R.Tensor((2048, 128), dtype="float16"),
        sin_cached: R.Tensor((2048, 128), dtype="float16"),
    ):
        cls = LlamaAttentionLayer
        lv6 = R.call_tir(
            cls.rms_norm,
            (input_tokens, rms_norm_weight),
            out_sinfo=R.Tensor((1, 256, 4096), dtype="float16"),
        )
        lv7: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(linear_weight, axes=None)
        lv7_copy: R.Tensor((4096, 4096), dtype="float16") = R.dist.annotate_sharding(
            lv7, "mesh[0]", "S[1]"
        )
        lv8: R.Tensor((1, 256, 4096), dtype="float16") = R.matmul(lv6, lv7_copy, out_dtype="void")
        lv9: R.Tensor((1, 256, 32, 128), dtype="float16") = R.reshape(
            lv8, R.shape([1, 256, 32, 128])
        )
        lv10: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(linear_weight1, axes=None)
        lv10_copy: R.Tensor((4096, 4096), dtype="float16") = R.dist.annotate_sharding(
            lv10, "mesh[0]", "S[1]"
        )
        lv11: R.Tensor((1, 256, 4096), dtype="float16") = R.matmul(lv6, lv10_copy, out_dtype="void")
        lv12: R.Tensor((1, 256, 32, 128), dtype="float16") = R.reshape(
            lv11, R.shape([1, 256, 32, 128])
        )
        lv13: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(linear_weight2, axes=None)
        lv13_copy: R.Tensor((4096, 4096), dtype="float16") = R.dist.annotate_sharding(
            lv13, "mesh[0]", "S[1]"
        )
        lv14: R.Tensor((1, 256, 4096), dtype="float16") = R.matmul(lv6, lv13_copy, out_dtype="void")
        lv15: R.Tensor((1, 256, 32, 128), dtype="float16") = R.reshape(
            lv14, R.shape([1, 256, 32, 128])
        )
        lv16 = R.call_tir(
            cls.rotary_embedding,
            (lv9, cos_cached, sin_cached),
            out_sinfo=R.Tensor((1, 256, 32, 128), dtype="float16"),
            tir_vars=R.shape([256]),
        )
        lv17 = R.call_tir(
            cls.rotary_embedding,
            (lv12, cos_cached, sin_cached),
            out_sinfo=R.Tensor((1, 256, 32, 128), dtype="float16"),
            tir_vars=R.shape([256]),
        )
        lv18: R.Tensor((256, 32, 128), dtype="float16") = R.reshape(lv17, R.shape([256, 32, 128]))
        lv19: R.Tensor((256, 32, 128), dtype="float16") = R.reshape(lv15, R.shape([256, 32, 128]))
        lv20: R.Object = kv_cache[0]
        lv21: R.Object = R.call_packed(
            "vm.builtin.attention_kv_cache_append", lv20, lv18, sinfo_args=(R.Object,)
        )
        lv22: R.Object = kv_cache[1]
        lv23: R.Object = R.call_packed(
            "vm.builtin.attention_kv_cache_append", lv22, lv19, sinfo_args=(R.Object,)
        )
        lv24: R.Tensor((256, 32, 128), dtype="float16") = R.call_packed(
            "vm.builtin.attention_kv_cache_view",
            lv21,
            R.shape([256, 32, 128]),
            sinfo_args=(R.Tensor((256, 32, 128), dtype="float16"),),
        )
        lv25: R.Tensor((256, 32, 128), dtype="float16") = R.call_packed(
            "vm.builtin.attention_kv_cache_view",
            lv23,
            R.shape([256, 32, 128]),
            sinfo_args=(R.Tensor((256, 32, 128), dtype="float16"),),
        )
        lv26: R.Tensor((1, 256, 32, 128), dtype="float16") = R.reshape(
            lv24, R.shape([1, 256, 32, 128])
        )
        lv27: R.Tensor((1, 256, 32, 128), dtype="float16") = R.reshape(
            lv25, R.shape([1, 256, 32, 128])
        )
        lv28: R.Tensor((1, 32, 256, 128), dtype="float16") = R.permute_dims(lv16, axes=[0, 2, 1, 3])
        lv29: R.Tensor((1, 32, 256, 128), dtype="float16") = R.permute_dims(lv26, axes=[0, 2, 1, 3])
        lv30: R.Tensor((1, 32, 256, 128), dtype="float16") = R.permute_dims(lv27, axes=[0, 2, 1, 3])
        lv31: R.Tensor((1, 32, 128, 256), dtype="float16") = R.permute_dims(lv29, axes=[0, 1, 3, 2])
        lv32: R.Tensor((1, 32, 256, 256), dtype="float16") = R.matmul(lv28, lv31, out_dtype="void")
        lv33: R.Tensor((1, 32, 256, 256), dtype="float16") = R.divide(lv32, div_const)
        lv34: R.Tensor((1, 32, 256, 256), dtype="float16") = R.maximum(lv33, maximum_const)
        lv35: R.Tensor((1, 32, 256, 256), dtype="float16") = R.minimum(lv34, mask)
        # lv36: R.Tensor((1, 32, 256, 256), dtype="float32") = R.astype(lv35, dtype="float32")
        lv37: R.Tensor((1, 32, 256, 256), dtype="float16") = R.nn.softmax(lv35, axis=-1)
        # lv38: R.Tensor((1, 32, 256, 256), dtype="float16") = R.astype(lv37, dtype="float16")
        lv39: R.Tensor((1, 32, 256, 128), dtype="float16") = R.matmul(lv37, lv30, out_dtype="void")
        lv40: R.Tensor((1, 256, 32, 128), dtype="float16") = R.permute_dims(lv39, axes=[0, 2, 1, 3])
        lv41: R.Tensor((1, 256, 4096), dtype="float16") = R.reshape(lv40, R.shape([1, 256, 4096]))
        lv42: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(linear_weight3, axes=None)
        lv43: R.Tensor((1, 256, 4096), dtype="float16") = R.matmul(lv41, lv42, out_dtype="void")
        lv44: R.Tensor((1, 256, 4096), dtype="float16") = R.add(input_tokens, lv43)
        gv = lv44

        return gv


@I.ir_module
class ShardedLlamaAttentionLayer:
    I.module_attrs({"device_num": 10})
    # I.module_global_infos({"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]})
    I.module_global_infos(
        {
            "mesh": [
                R.device_mesh((2,), I.Range(0, 2)),  # mesh[0]
                R.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
            ]
        }
    )

    @T.prim_func
    def rms_norm(
        A: T.Buffer((T.int64(1), 256, T.int64(4096)), "float16"),
        B: T.Buffer((T.int64(4096),), "float16"),
        rms_norm_1: T.Buffer((T.int64(1), 256, T.int64(4096)), "float16"),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        Ared_temp = T.alloc_buffer((T.int64(1), 256))
        for bsz, i, k in T.grid(T.int64(1), 256, T.int64(4096)):
            with T.block("Ared_temp"):
                v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
                T.reads(A[v_bsz, v_i, v_k])
                T.writes(Ared_temp[v_bsz, v_i])
                with T.init():
                    Ared_temp[v_bsz, v_i] = T.float32(0)
                Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast(
                    "float32", A[v_bsz, v_i, v_k]
                ) * T.Cast("float32", A[v_bsz, v_i, v_k])
        for bsz, i, k in T.grid(T.int64(1), 256, T.int64(4096)):
            with T.block("rms_norm"):
                v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
                T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
                T.writes(rms_norm_1[v_bsz, v_i, v_k])
                rms_norm_1[v_bsz, v_i, v_k] = T.Cast(
                    "float16",
                    T.Cast("float32", B[v_k])
                    * (
                        T.Cast("float32", A[v_bsz, v_i, v_k])
                        / T.sqrt(
                            Ared_temp[v_bsz, v_i] * T.float32(0.000244140625)
                            + T.float32(9.9999999999999995e-07)
                        )
                    ),
                )

    @T.prim_func
    def rotary_embedding(
        A: T.Buffer((T.int64(1), 256, T.int64(32), T.int64(128)), "float16"),
        B: T.Buffer((T.int64(2048), T.int64(128)), "float16"),
        C: T.Buffer((T.int64(2048), T.int64(128)), "float16"),
        rotary: T.Buffer((T.int64(1), 256, T.int64(32), T.int64(128)), "float16"),
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), 256, T.int64(32), T.int64(128)):
            with T.block("rotary"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(
                    B[256 + v_i1 - 256, v_i3],
                    A[v_i0, v_i1, v_i2, v_i3 - T.int64(64) : v_i3 - T.int64(64) + T.int64(129)],
                    C[256 + v_i1 - 256, v_i3],
                )
                T.writes(rotary[v_i0, v_i1, v_i2, v_i3])
                rotary[v_i0, v_i1, v_i2, v_i3] = B[256 + v_i1 - 256, v_i3] * A[
                    v_i0, v_i1, v_i2, v_i3
                ] + C[256 + v_i1 - 256, v_i3] * T.Select(
                    T.int64(64) <= v_i3,
                    A[v_i0, v_i1, v_i2, v_i3 - T.int64(64)],
                    A[v_i0, v_i1, v_i2, v_i3 + T.int64(64)] * T.float16(-1),
                )

    @R.function(pure=False)
    def foo(
        input_tokens: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R"),
        mask: R.DTensor((1, 1, 256, 256), "float16", "mesh[0]", "R"),
        div_const: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]"),
        maximum_const: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]"),
        kv_cache: R.Tuple(R.Object, R.Object),
        linear_weight: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]"),
        linear_weight1: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]"),
        linear_weight2: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]"),
        linear_weight3: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]"),
        rms_norm_weight: R.DTensor((4096,), "float16", "mesh[0]", "R"),
        cos_cached: R.DTensor((2048, 128), "float16", "mesh[0]", "R"),
        sin_cached: R.DTensor((2048, 128), "float16", "mesh[0]", "R"),
    ) -> R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R"):
        cls = ShardedLlamaAttentionLayer
        lv6 = R.dist.call_tir(
            cls.rms_norm,
            (input_tokens, rms_norm_weight),
            out_sinfo=R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R"),
        )
        lv7: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = R.permute_dims(
            linear_weight, axes=None
        )
        lv7_copy: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = lv7
        lv8: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]") = R.matmul(
            lv6, lv7_copy, out_dtype="void"
        )
        lv9: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
            lv8, R.shape([1, 256, 32, 128])
        )
        lv10: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = R.permute_dims(
            linear_weight1, axes=None
        )
        lv10_copy: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = lv10
        lv11: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]") = R.matmul(
            lv6, lv10_copy, out_dtype="void"
        )
        lv12: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
            lv11, R.shape([1, 256, 32, 128])
        )
        lv13: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = R.permute_dims(
            linear_weight2, axes=None
        )
        lv13_copy: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = lv13
        lv14: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]") = R.matmul(
            lv6, lv13_copy, out_dtype="void"
        )
        lv15: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
            lv14, R.shape([1, 256, 32, 128])
        )
        lv16 = R.dist.call_tir(
            cls.rotary_embedding,
            (lv9, cos_cached, sin_cached),
            out_sinfo=R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]"),
            tir_vars=R.shape([256]),
        )
        lv17 = R.dist.call_tir(
            cls.rotary_embedding,
            (lv12, cos_cached, sin_cached),
            out_sinfo=R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]"),
            tir_vars=R.shape([256]),
        )
        lv18: R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]") = R.reshape(
            lv17, R.shape([256, 32, 128])
        )
        lv19: R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]") = R.reshape(
            lv15, R.shape([256, 32, 128])
        )
        lv20: R.Object = kv_cache[0]
        lv21: R.Object = R.call_packed(
            "vm.builtin.distributed.attention_kv_cache_append",
            lv20,
            lv18,
            sinfo_args=(R.Object,),
        )
        lv22: R.Object = kv_cache[1]
        lv23: R.Object = R.call_packed(
            "vm.builtin.distributed.attention_kv_cache_append",
            lv22,
            lv19,
            sinfo_args=(R.Object,),
        )
        lv24: R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]") = R.call_packed(
            "vm.builtin.distributed.attention_kv_cache_view",
            lv21,
            R.shape([256, 32, 128]),
            sinfo_args=(R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]"),),
        )
        lv25: R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]") = R.call_packed(
            "vm.builtin.distributed.attention_kv_cache_view",
            lv23,
            R.shape([256, 32, 128]),
            sinfo_args=(R.DTensor((256, 32, 128), "float16", "mesh[0]", "S[1]"),),
        )
        lv26: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
            lv24, R.shape([1, 256, 32, 128])
        )
        lv27: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
            lv25, R.shape([1, 256, 32, 128])
        )
        lv28: R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]") = R.permute_dims(
            lv16, axes=[0, 2, 1, 3]
        )
        lv29: R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]") = R.permute_dims(
            lv26, axes=[0, 2, 1, 3]
        )
        lv30: R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]") = R.permute_dims(
            lv27, axes=[0, 2, 1, 3]
        )
        lv31: R.DTensor((1, 32, 128, 256), "float16", "mesh[0]", "S[1]") = R.permute_dims(
            lv29, axes=[0, 1, 3, 2]
        )
        lv32: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]") = R.matmul(
            lv28, lv31, out_dtype="void"
        )
        lv33: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]") = R.divide(lv32, div_const)
        lv34: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]") = R.maximum(
            lv33, maximum_const
        )
        lv35: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]") = R.minimum(lv34, mask)
        lv37: R.DTensor((1, 32, 256, 256), "float16", "mesh[0]", "S[1]") = R.nn.softmax(
            lv35, axis=-1
        )
        lv39: R.DTensor((1, 32, 256, 128), "float16", "mesh[0]", "S[1]") = R.matmul(
            lv37, lv30, out_dtype="void"
        )
        lv40: R.DTensor((1, 256, 32, 128), "float16", "mesh[0]", "S[2]") = R.permute_dims(
            lv39, axes=[0, 2, 1, 3]
        )
        lv41: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "S[2]") = R.reshape(
            lv40, R.shape([1, 256, 4096])
        )
        lv42: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]") = R.permute_dims(
            linear_weight3, axes=None
        )
        lv43: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R") = R.matmul(
            lv41, lv42, out_dtype="void"
        )
        lv44: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R") = R.add(input_tokens, lv43)
        gv: R.DTensor((1, 256, 4096), "float16", "mesh[0]", "R") = lv44
        return gv


@I.ir_module
class LlamaAttentionLayerDynamicShape:
    I.module_attrs({"device_num": 10})
    I.module_global_infos(
        {
            "mesh": [
                R.device_mesh((2,), I.Range(0, 2)),  # mesh[0]
                R.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
            ]
        }
    )

    @T.prim_func
    def rms_norm(var_A: T.handle, B: T.Buffer((T.int64(4096),), "float16"), var_rms_norm: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
        rms_norm_1 = T.match_buffer(var_rms_norm, (T.int64(1), n, T.int64(4096)), "float16")
        # with T.block("root"):
        Ared_temp = T.alloc_buffer((T.int64(1), n))
        for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("Ared_temp"):
                v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
                T.reads(A[v_bsz, v_i, v_k])
                T.writes(Ared_temp[v_bsz, v_i])
                with T.init():
                    Ared_temp[v_bsz, v_i] = T.float32(0)
                Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast(
                    "float32", A[v_bsz, v_i, v_k]
                ) * T.Cast("float32", A[v_bsz, v_i, v_k])
        for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("rms_norm"):
                v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
                T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
                T.writes(rms_norm_1[v_bsz, v_i, v_k])
                rms_norm_1[v_bsz, v_i, v_k] = T.Cast(
                    "float16",
                    T.Cast("float32", B[v_k])
                    * (
                        T.Cast("float32", A[v_bsz, v_i, v_k])
                        / T.sqrt(
                            Ared_temp[v_bsz, v_i] * T.float32(0.000244140625)
                            + T.float32(9.9999999999999995e-07)
                        )
                    ),
                )

    @T.prim_func
    def rotary_embedding(
        var_A: T.handle,
        B: T.Buffer((T.int64(2048), T.int64(128)), "float16"),
        C: T.Buffer((T.int64(2048), T.int64(128)), "float16"),
        var_rotary: T.handle,
        m: T.int64,
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
        rotary = T.match_buffer(var_rotary, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), n, T.int64(32), T.int64(128)):
            with T.block("rotary"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(
                    B[m + v_i1 - n, v_i3],
                    A[v_i0, v_i1, v_i2, v_i3 - T.int64(64) : v_i3 - T.int64(64) + T.int64(129)],
                    C[m + v_i1 - n, v_i3],
                )
                T.writes(rotary[v_i0, v_i1, v_i2, v_i3])
                rotary[v_i0, v_i1, v_i2, v_i3] = B[m + v_i1 - n, v_i3] * A[
                    v_i0, v_i1, v_i2, v_i3
                ] + C[m + v_i1 - n, v_i3] * T.Select(
                    T.int64(64) <= v_i3,
                    A[v_i0, v_i1, v_i2, v_i3 - T.int64(64)],
                    A[v_i0, v_i1, v_i2, v_i3 + T.int64(64)] * T.float16(-1),
                )

    @R.function(pure=False)
    def foo(
        input_tokens: R.Tensor((1, "n", 4096), dtype="float16"),
        mask: R.Tensor((1, 1, "n", "m"), dtype="float16"),
        kv_cache: R.Tuple(R.Object, R.Object),
        linear_weight: R.Tensor((4096, 4096), dtype="float16"),
        linear_weight1: R.Tensor((4096, 4096), dtype="float16"),
        linear_weight2: R.Tensor((4096, 4096), dtype="float16"),
        linear_weight3: R.Tensor((4096, 4096), dtype="float16"),
        rms_norm_weight: R.Tensor((4096,), dtype="float16"),
        cos_cached: R.Tensor((2048, 128), dtype="float16"),
        sin_cached: R.Tensor((2048, 128), dtype="float16"),
    ):
        n = T.int64()
        m = T.int64()
        cls = LlamaAttentionLayerDynamicShape
        lv6 = R.call_tir(
            cls.rms_norm,
            (input_tokens, rms_norm_weight),
            out_sinfo=R.Tensor((1, n, 4096), dtype="float16"),
        )
        lv7: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(linear_weight, axes=None)
        lv7_copy: R.Tensor((4096, 4096), dtype="float16") = R.dist.annotate_sharding(
            lv7, "mesh[0]", "S[1]"
        )
        lv8: R.Tensor((1, n, 4096), dtype="float16") = R.matmul(lv6, lv7_copy, out_dtype="void")
        lv9: R.Tensor((1, n, 32, 128), dtype="float16") = R.reshape(lv8, R.shape([1, n, 32, 128]))
        lv10: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(linear_weight1, axes=None)
        lv10_copy: R.Tensor((4096, 4096), dtype="float16") = R.dist.annotate_sharding(
            lv10, "mesh[0]", "S[1]"
        )
        lv11: R.Tensor((1, n, 4096), dtype="float16") = R.matmul(lv6, lv10_copy, out_dtype="void")
        lv12: R.Tensor((1, n, 32, 128), dtype="float16") = R.reshape(lv11, R.shape([1, n, 32, 128]))
        lv13: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(linear_weight2, axes=None)
        lv13_copy: R.Tensor((4096, 4096), dtype="float16") = R.dist.annotate_sharding(
            lv13, "mesh[0]", "S[1]"
        )
        lv14: R.Tensor((1, n, 4096), dtype="float16") = R.matmul(lv6, lv13_copy, out_dtype="void")
        lv15: R.Tensor((1, n, 32, 128), dtype="float16") = R.reshape(lv14, R.shape([1, n, 32, 128]))
        lv16 = R.call_tir(
            cls.rotary_embedding,
            (lv9, cos_cached, sin_cached),
            out_sinfo=R.Tensor((1, n, 32, 128), dtype="float16"),
            tir_vars=R.shape([m]),
        )
        lv17 = R.call_tir(
            cls.rotary_embedding,
            (lv12, cos_cached, sin_cached),
            out_sinfo=R.Tensor((1, n, 32, 128), dtype="float16"),
            tir_vars=R.shape([m]),
        )
        lv18: R.Tensor((n, 32, 128), dtype="float16") = R.reshape(lv17, R.shape([n, 32, 128]))
        lv19: R.Tensor((n, 32, 128), dtype="float16") = R.reshape(lv15, R.shape([n, 32, 128]))
        lv20: R.Object = kv_cache[0]
        lv21: R.Object = R.call_packed(
            "vm.builtin.attention_kv_cache_append", lv20, lv18, sinfo_args=(R.Object,)
        )
        lv22: R.Object = kv_cache[1]
        lv23: R.Object = R.call_packed(
            "vm.builtin.attention_kv_cache_append", lv22, lv19, sinfo_args=(R.Object,)
        )
        lv24: R.Tensor((m, 32, 128), dtype="float16") = R.call_packed(
            "vm.builtin.attention_kv_cache_view",
            lv21,
            R.shape([m, 32, 128]),
            sinfo_args=(R.Tensor((m, 32, 128), dtype="float16"),),
        )
        lv25: R.Tensor((m, 32, 128), dtype="float16") = R.call_packed(
            "vm.builtin.attention_kv_cache_view",
            lv23,
            R.shape([m, 32, 128]),
            sinfo_args=(R.Tensor((m, 32, 128), dtype="float16"),),
        )
        lv26: R.Tensor((1, m, 32, 128), dtype="float16") = R.reshape(lv24, R.shape([1, m, 32, 128]))
        lv27: R.Tensor((1, m, 32, 128), dtype="float16") = R.reshape(lv25, R.shape([1, m, 32, 128]))
        lv28: R.Tensor((1, 32, n, 128), dtype="float16") = R.permute_dims(lv16, axes=[0, 2, 1, 3])
        lv29: R.Tensor((1, 32, m, 128), dtype="float16") = R.permute_dims(lv26, axes=[0, 2, 1, 3])
        lv30: R.Tensor((1, 32, m, 128), dtype="float16") = R.permute_dims(lv27, axes=[0, 2, 1, 3])
        lv31: R.Tensor((1, 32, 128, m), dtype="float16") = R.permute_dims(lv29, axes=[0, 1, 3, 2])
        lv32: R.Tensor((1, 32, n, m), dtype="float16") = R.matmul(lv28, lv31, out_dtype="void")
        lv33: R.Tensor((1, 32, n, m), dtype="float16") = R.divide(
            lv32, R.const(8, dtype="float16")
        )  # just choose some random value
        lv34: R.Tensor((1, 32, n, m), dtype="float16") = R.maximum(
            lv33, R.const(1, dtype="float16")
        )  # just choose some random value
        lv35: R.Tensor((1, 32, n, m), dtype="float16") = R.minimum(lv34, mask)
        # lv36: R.Tensor((1, 32, n, m), dtype="float32") = R.astype(lv35, dtype="float32")
        lv37: R.Tensor((1, 32, n, m), dtype="float16") = R.nn.softmax(lv35, axis=-1)
        # lv38: R.Tensor((1, 32, n, m), dtype="float16") = R.astype(lv37, dtype="float16")
        lv39: R.Tensor((1, 32, n, 128), dtype="float16") = R.matmul(lv37, lv30, out_dtype="void")
        lv40: R.Tensor((1, n, 32, 128), dtype="float16") = R.permute_dims(lv39, axes=[0, 2, 1, 3])
        lv41: R.Tensor((1, n, 4096), dtype="float16") = R.reshape(lv40, R.shape([1, n, 4096]))
        lv42: R.Tensor((4096, 4096), dtype="float16") = R.permute_dims(linear_weight3, axes=None)
        lv43: R.Tensor((1, n, 4096), dtype="float16") = R.matmul(lv41, lv42, out_dtype="void")
        lv44: R.Tensor((1, n, 4096), dtype="float16") = R.add(input_tokens, lv43)
        gv = lv44

        return gv


@I.ir_module
class ShardedLlamaAttentionLayerDynamicShape:
    I.module_attrs({"device_num": 10})
    I.module_global_infos(
        {"mesh": [R.device_mesh((2,), I.Range(0, 2)), R.device_mesh((1,), I.Range(4, 5))]}
    )

    @T.prim_func
    def rms_norm(var_A: T.handle, B: T.Buffer((T.int64(4096),), "float16"), var_rms_norm: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
        rms_norm_1 = T.match_buffer(var_rms_norm, (T.int64(1), n, T.int64(4096)), "float16")
        # with T.block("root"):
        Ared_temp = T.alloc_buffer((T.int64(1), n))
        for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("Ared_temp"):
                v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
                T.reads(A[v_bsz, v_i, v_k])
                T.writes(Ared_temp[v_bsz, v_i])
                with T.init():
                    Ared_temp[v_bsz, v_i] = T.float32(0)
                Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast(
                    "float32", A[v_bsz, v_i, v_k]
                ) * T.Cast("float32", A[v_bsz, v_i, v_k])
        for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
            with T.block("rms_norm"):
                v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
                T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
                T.writes(rms_norm_1[v_bsz, v_i, v_k])
                rms_norm_1[v_bsz, v_i, v_k] = T.Cast(
                    "float16",
                    T.Cast("float32", B[v_k])
                    * (
                        T.Cast("float32", A[v_bsz, v_i, v_k])
                        / T.sqrt(
                            Ared_temp[v_bsz, v_i] * T.float32(0.000244140625)
                            + T.float32(9.9999999999999995e-07)
                        )
                    ),
                )

    @T.prim_func
    def rotary_embedding(
        var_A: T.handle,
        B: T.Buffer((T.int64(2048), T.int64(128)), "float16"),
        C: T.Buffer((T.int64(2048), T.int64(128)), "float16"),
        var_rotary: T.handle,
        m: T.int64,
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
        rotary = T.match_buffer(var_rotary, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(1), n, T.int64(32), T.int64(128)):
            with T.block("rotary"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(
                    B[m + v_i1 - n, v_i3],
                    A[v_i0, v_i1, v_i2, v_i3 - T.int64(64) : v_i3 - T.int64(64) + T.int64(129)],
                    C[m + v_i1 - n, v_i3],
                )
                T.writes(rotary[v_i0, v_i1, v_i2, v_i3])
                rotary[v_i0, v_i1, v_i2, v_i3] = B[m + v_i1 - n, v_i3] * A[
                    v_i0, v_i1, v_i2, v_i3
                ] + C[m + v_i1 - n, v_i3] * T.Select(
                    T.int64(64) <= v_i3,
                    A[v_i0, v_i1, v_i2, v_i3 - T.int64(64)],
                    A[v_i0, v_i1, v_i2, v_i3 + T.int64(64)] * T.float16(-1),
                )

    @R.function(pure=False)
    def foo(
        input_tokens: R.DTensor((1, "n", 4096), "float16", "mesh[0]", "R"),
        mask: R.DTensor((1, 1, "n", "m"), "float16", "mesh[0]", "R"),
        kv_cache: R.Tuple(R.Object, R.Object),
        linear_weight: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]"),
        linear_weight1: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]"),
        linear_weight2: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]"),
        linear_weight3: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]"),
        rms_norm_weight: R.DTensor((4096,), "float16", "mesh[0]", "R"),
        cos_cached: R.DTensor((2048, 128), "float16", "mesh[0]", "R"),
        sin_cached: R.DTensor((2048, 128), "float16", "mesh[0]", "R"),
    ) -> R.DTensor((1, "n", 4096), "float16", "mesh[0]", "R"):
        n = T.int64()
        m = T.int64()
        cls = ShardedLlamaAttentionLayerDynamicShape
        lv6 = R.dist.call_tir(
            cls.rms_norm,
            (input_tokens, rms_norm_weight),
            out_sinfo=R.DTensor((1, n, 4096), "float16", "mesh[0]", "R"),
        )
        lv7: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = R.permute_dims(
            linear_weight, axes=None
        )
        lv7_copy: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = lv7
        lv8: R.DTensor((1, n, 4096), "float16", "mesh[0]", "S[2]") = R.matmul(
            lv6, lv7_copy, out_dtype="void"
        )
        lv9: R.DTensor((1, n, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
            lv8, R.shape([1, n, 32, 128])
        )
        lv10: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = R.permute_dims(
            linear_weight1, axes=None
        )
        lv10_copy: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = lv10
        lv11: R.DTensor((1, n, 4096), "float16", "mesh[0]", "S[2]") = R.matmul(
            lv6, lv10_copy, out_dtype="void"
        )
        lv12: R.DTensor((1, n, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
            lv11, R.shape([1, n, 32, 128])
        )
        lv13: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = R.permute_dims(
            linear_weight2, axes=None
        )
        lv13_copy: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[1]") = lv13
        lv14: R.DTensor((1, n, 4096), "float16", "mesh[0]", "S[2]") = R.matmul(
            lv6, lv13_copy, out_dtype="void"
        )
        lv15: R.DTensor((1, n, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
            lv14, R.shape([1, n, 32, 128])
        )
        lv16 = R.dist.call_tir(
            cls.rotary_embedding,
            (lv9, cos_cached, sin_cached),
            out_sinfo=R.DTensor((1, n, 32, 128), "float16", "mesh[0]", "S[2]"),
            tir_vars=R.shape([m]),
        )
        lv17 = R.dist.call_tir(
            cls.rotary_embedding,
            (lv12, cos_cached, sin_cached),
            out_sinfo=R.DTensor((1, n, 32, 128), "float16", "mesh[0]", "S[2]"),
            tir_vars=R.shape([m]),
        )
        lv18: R.DTensor((n, 32, 128), "float16", "mesh[0]", "S[1]") = R.reshape(
            lv17, R.shape([n, 32, 128])
        )
        lv19: R.DTensor((n, 32, 128), "float16", "mesh[0]", "S[1]") = R.reshape(
            lv15, R.shape([n, 32, 128])
        )
        lv20: R.Object = kv_cache[0]
        lv21: R.Object = R.call_packed(
            "vm.builtin.distributed.attention_kv_cache_append", lv20, lv18, sinfo_args=(R.Object,)
        )
        lv22: R.Object = kv_cache[1]
        lv23: R.Object = R.call_packed(
            "vm.builtin.distributed.attention_kv_cache_append", lv22, lv19, sinfo_args=(R.Object,)
        )
        lv24: R.DTensor((m, 32, 128), "float16", "mesh[0]", "S[1]") = R.call_packed(
            "vm.builtin.distributed.attention_kv_cache_view",
            lv21,
            R.shape([m, 32, 128]),
            sinfo_args=(R.DTensor((m, 32, 128), "float16", "mesh[0]", "S[1]"),),
        )
        lv25: R.DTensor((m, 32, 128), "float16", "mesh[0]", "S[1]") = R.call_packed(
            "vm.builtin.distributed.attention_kv_cache_view",
            lv23,
            R.shape([m, 32, 128]),
            sinfo_args=(R.DTensor((m, 32, 128), "float16", "mesh[0]", "S[1]"),),
        )
        lv26: R.DTensor((1, m, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
            lv24, R.shape([1, m, 32, 128])
        )
        lv27: R.DTensor((1, m, 32, 128), "float16", "mesh[0]", "S[2]") = R.reshape(
            lv25, R.shape([1, m, 32, 128])
        )
        lv28: R.DTensor((1, 32, n, 128), "float16", "mesh[0]", "S[1]") = R.permute_dims(
            lv16, axes=[0, 2, 1, 3]
        )
        lv29: R.DTensor((1, 32, m, 128), "float16", "mesh[0]", "S[1]") = R.permute_dims(
            lv26, axes=[0, 2, 1, 3]
        )
        lv30: R.DTensor((1, 32, m, 128), "float16", "mesh[0]", "S[1]") = R.permute_dims(
            lv27, axes=[0, 2, 1, 3]
        )
        lv31: R.DTensor((1, 32, 128, m), "float16", "mesh[0]", "S[1]") = R.permute_dims(
            lv29, axes=[0, 1, 3, 2]
        )
        lv32: R.DTensor((1, 32, n, m), "float16", "mesh[0]", "S[1]") = R.matmul(
            lv28, lv31, out_dtype="void"
        )
        lv33: R.DTensor((1, 32, n, m), "float16", "mesh[0]", "S[1]") = R.divide(
            lv32, R.dist.const(8, R.DTensor((), "float16", "mesh[0]", "R"))
        )
        lv34: R.DTensor((1, 32, n, m), "float16", "mesh[0]", "S[1]") = R.maximum(
            lv33, R.dist.const(1, R.DTensor((), "float16", "mesh[0]", "R"))
        )
        lv35: R.DTensor((1, 32, n, m), "float16", "mesh[0]", "S[1]") = R.minimum(lv34, mask)
        lv37: R.DTensor((1, 32, n, m), "float16", "mesh[0]", "S[1]") = R.nn.softmax(lv35, axis=-1)
        lv39: R.DTensor((1, 32, n, 128), "float16", "mesh[0]", "S[1]") = R.matmul(
            lv37, lv30, out_dtype="void"
        )
        lv40: R.DTensor((1, n, 32, 128), "float16", "mesh[0]", "S[2]") = R.permute_dims(
            lv39, axes=[0, 2, 1, 3]
        )
        lv41: R.DTensor((1, n, 4096), "float16", "mesh[0]", "S[2]") = R.reshape(
            lv40, R.shape([1, n, 4096])
        )
        lv42: R.DTensor((4096, 4096), "float16", "mesh[0]", "S[0]") = R.permute_dims(
            linear_weight3, axes=None
        )
        lv43: R.DTensor((1, n, 4096), "float16", "mesh[0]", "R") = R.matmul(
            lv41, lv42, out_dtype="void"
        )
        lv44: R.DTensor((1, n, 4096), "float16", "mesh[0]", "R") = R.add(input_tokens, lv43)
        gv: R.DTensor((1, n, 4096), "float16", "mesh[0]", "R") = lv44
        return gv


def test_mlp():
    after = relax.distributed.transform.PropagateSharding()(MLP)
    assert_structural_equal(after, ShardedMLP)


def test_mlp_const():
    after = relax.distributed.transform.PropagateSharding()(MLPWithConst)
    assert_structural_equal(after, ShardedMLPWithConst)


def test_mlp_dynamic_shape():
    after = relax.distributed.transform.PropagateSharding()(MLPDynamicShape)
    assert_structural_equal(after, ShardedMLPDynamicShape)


def test_mlp_pipeline_parallelism():
    after = relax.distributed.transform.PropagateSharding()(PipelineMLP)
    assert_structural_equal(after, ShardedPipelineMLP)


def test_decoder_layer():
    # mod = relax.transform.LegalizeOps({"relax.reshape": lambda bb, call: bb.normalize(call)})(LlamaAttentionLayer)
    mod = LlamaAttentionLayer
    after = relax.distributed.transform.PropagateSharding()(mod)
    assert_structural_equal(after, ShardedLlamaAttentionLayer)


def test_decoder_layer_dynamic_shape():
    # mod = relax.transform.LegalizeOps({"relax.reshape": lambda bb, call: bb.normalize(call)})(LlamaAttentionLayer)
    mod = LlamaAttentionLayerDynamicShape
    after = relax.distributed.transform.PropagateSharding()(mod)
    assert_structural_equal(after, ShardedLlamaAttentionLayerDynamicShape)


if __name__ == "__main__":
    tvm.testing.main()
