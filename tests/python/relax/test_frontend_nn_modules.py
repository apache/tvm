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
from typing import List, Tuple

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.ir import assert_structural_equal
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import core, modules, spec
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R


def test_relu():
    @R.function
    def forward(
        x: R.Tensor((3, 3), dtype="float32"),
        _io: R.Object,
    ) -> R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tuple(R.Object)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            relu: R.Tensor((3, 3), dtype="float32") = R.nn.relu(x)
            gv1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tuple(R.Object)) = relu, (_io,)
            R.output(gv1)
        return gv1

    mod = modules.ReLU()
    tvm_mod, _ = mod.export_tvm(spec={"forward": {"x": spec.Tensor((3, 3), "float32")}}, debug=True)
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_silu():
    @R.function
    def forward(
        x: R.Tensor((3, 3), dtype="float32"),
        _io: R.Object,
    ) -> R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tuple(R.Object)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            silu: R.Tensor((3, 3), dtype="float32") = R.nn.silu(x)
            gv1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tuple(R.Object)) = silu, (_io,)
            R.output(gv1)
        return gv1

    mod = modules.SiLU()
    tvm_mod, _ = mod.export_tvm(spec={"forward": {"x": spec.Tensor((3, 3), "float32")}}, debug=True)
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_gelu():
    @R.function
    def forward(
        x: R.Tensor((3, 3), dtype="float32"),
        _io: R.Object,
    ) -> R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tuple(R.Object)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            gelu: R.Tensor((3, 3), dtype="float32") = R.nn.gelu(x)
            gv1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tuple(R.Object)) = gelu, (_io,)
            R.output(gv1)
        return gv1

    mod = modules.GELU()
    tvm_mod, _ = mod.export_tvm(spec={"forward": {"x": spec.Tensor((3, 3), "float32")}}, debug=True)
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_identity():
    @R.function
    def forward(
        x: R.Tensor((3, 3), dtype="float32"),
        _io: R.Object,
    ) -> R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tuple(R.Object)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            gv1: R.Tuple(R.Tensor((3, 3), dtype="float32"), R.Tuple(R.Object)) = x, (_io,)
            R.output(gv1)
        return gv1

    mod = modules.Identity()
    tvm_mod, _ = mod.export_tvm(spec={"forward": {"x": spec.Tensor((3, 3), "float32")}}, debug=True)
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_linear():
    @R.function
    def forward(
        x: R.Tensor((1, 4), dtype="float32"),
        _io: R.Object,
        weight: R.Tensor((8, 4), dtype="float32"),
        bias: R.Tensor((8,), dtype="float32"),
    ) -> R.Tuple(R.Tensor((1, 8), dtype="float32"), R.Tuple(R.Object)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            permute_dims: R.Tensor((4, 8), dtype="float32") = R.permute_dims(weight, axes=None)
            matmul: R.Tensor((1, 8), dtype="float32") = R.matmul(x, permute_dims, out_dtype="void")
            add: R.Tensor((1, 8), dtype="float32") = R.add(matmul, bias)
            gv1: R.Tuple(R.Tensor((1, 8), dtype="float32"), R.Tuple(R.Object)) = add, (_io,)
            R.output(gv1)
        return gv1

    mod = modules.Linear(4, 8)
    tvm_mod, _ = mod.export_tvm(spec={"forward": {"x": spec.Tensor((1, 4), "float32")}}, debug=True)
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_conv1d():
    @R.function
    def forward(
        x: R.Tensor((1, 3, 32), dtype="float32"),
        _io: R.Object,
        weight: R.Tensor((32, 3, 3), dtype="float32"),
        bias: R.Tensor((32,), dtype="float32"),
    ) -> R.Tuple(R.Tensor((1, 32, 30), dtype="float32"), R.Tuple(R.Object)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            lv1: R.Tensor((1, 32, 30), dtype="float32") = R.nn.conv1d(
                x,
                weight,
                strides=[1],
                padding=[0, 0],
                dilation=[1],
                groups=1,
                data_layout="NCW",
                kernel_layout="OIW",
                out_layout="NCW",
                out_dtype="void",
            )
            lv2: R.Tensor((1, 32, 1), dtype="float32") = R.reshape(bias, R.shape([1, 32, 1]))
            conv1d: R.Tensor((1, 32, 30), dtype="float32") = R.add(lv1, lv2)
            gv1: R.Tuple(R.Tensor((1, 32, 30), dtype="float32"), R.Tuple(R.Object)) = conv1d, (_io,)
            R.output(gv1)
        return gv1

    mod = modules.Conv1D(3, 32, 3, bias=True)
    tvm_mod, _ = mod.export_tvm(
        spec={
            "forward": {
                "x": spec.Tensor([1, 3, 32], "float32"),
            }
        },
        debug=True,
    )
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_conv1d_transpose():
    # fmt: off
    @R.function
    def forward(x: R.Tensor((1, 3, 30), dtype="float32"), _io: R.Object, weight: R.Tensor((3, 32, 3), dtype="float32"), bias: R.Tensor((32,), dtype="float32")) -> R.Tuple(R.Tensor((1, 32, 32), dtype="float32"), R.Tuple(R.Object)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            lv1: R.Tensor((1, 32, 32), dtype="float32") = R.nn.conv1d_transpose(x, weight, strides=[1], padding=[0, 0], output_padding=[0], dilation=[1], groups=1, data_layout="NCW", kernel_layout="IOW", out_layout="NCW", out_dtype="void")
            lv2: R.Tensor((1, 32, 1), dtype="float32") = R.reshape(bias, R.shape([1, 32, 1]))
            conv1d_transpose: R.Tensor((1, 32, 32), dtype="float32") = R.add(lv1, lv2)
            gv1: R.Tuple(R.Tensor((1, 32, 32), dtype="float32"), R.Tuple(R.Object)) = conv1d_transpose, (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    mod = modules.ConvTranspose1D(3, 32, 3, bias=True)
    tvm_mod, _ = mod.export_tvm(
        spec={
            "forward": {
                "x": spec.Tensor([1, 3, 30], "float32"),
            }
        },
        debug=True,
    )
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_layer_norm():
    @R.function
    def forward(
        x: R.Tensor((2, 4, 8), dtype="float32"),
        _io: R.Object,
        weight: R.Tensor((8,), dtype="float32"),
        bias: R.Tensor((8,), dtype="float32"),
    ) -> R.Tuple(R.Tensor((2, 4, 8), dtype="float32"), R.Tuple(R.Object)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            layer_norm: R.Tensor((2, 4, 8), dtype="float32") = R.nn.layer_norm(
                x, weight, bias, axes=[-1], epsilon=1.0000000000000001e-05, center=True, scale=True
            )
            gv1: R.Tuple(R.Tensor((2, 4, 8), dtype="float32"), R.Tuple(R.Object)) = layer_norm, (
                _io,
            )
            R.output(gv1)
        return gv1

    mod = modules.LayerNorm(8)
    tvm_mod, _ = mod.export_tvm(
        spec={"forward": {"x": spec.Tensor((2, 4, 8), "float32")}}, debug=True
    )
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_conv2d():
    @R.function
    def forward(
        x: R.Tensor((1, 3, 32, 32), dtype="float32"),
        _io: R.Object,
        weight: R.Tensor((32, 3, 3, 3), dtype="float32"),
        bias: R.Tensor((32,), dtype="float32"),
    ) -> R.Tuple(R.Tensor((1, 32, 30, 30), dtype="float32"), R.Tuple(R.Object)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            lv1: R.Tensor((1, 32, 30, 30), dtype="float32") = R.nn.conv2d(x, weight)
            lv2: R.Tensor((1, 32, 1, 1), dtype="float32") = R.reshape(bias, R.shape([1, 32, 1, 1]))
            conv2d: R.Tensor((1, 32, 30, 30), dtype="float32") = R.add(lv1, lv2)
            gv1: R.Tuple(R.Tensor((1, 32, 30, 30), dtype="float32"), R.Tuple(R.Object)) = conv2d, (
                _io,
            )
            R.output(gv1)
        return gv1

    mod = modules.Conv2D(3, 32, 3, bias=True)
    tvm_mod, _ = mod.export_tvm(
        spec={
            "forward": {
                "x": spec.Tensor([1, 3, 32, 32], "float32"),
            }
        },
        debug=True,
    )
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_conv3d():
    @R.function
    def forward(
        x: R.Tensor((1, 3, 32, 32, 32), dtype="float32"),
        _io: R.Object,
        weight: R.Tensor((32, 3, 3, 3, 3), dtype="float32"),
        bias: R.Tensor((32,), dtype="float32"),
    ) -> R.Tuple(R.Tensor((1, 32, 30, 30, 30), dtype="float32"), R.Tuple(R.Object)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            lv1: R.Tensor((1, 32, 30, 30, 30), dtype="float32") = R.nn.conv3d(x, weight)
            lv2: R.Tensor((1, 32, 1, 1, 1), dtype="float32") = R.reshape(
                bias, R.shape([1, 32, 1, 1, 1])
            )
            conv3d: R.Tensor((1, 32, 30, 30, 30), dtype="float32") = R.add(lv1, lv2)
            gv1: R.Tuple(
                R.Tensor((1, 32, 30, 30, 30), dtype="float32"), R.Tuple(R.Object)
            ) = conv3d, (_io,)
            R.output(gv1)
        return gv1

    mod = modules.Conv3D(3, 32, 3, bias=True)
    tvm_mod, _ = mod.export_tvm(
        spec={
            "forward": {
                "x": spec.Tensor([1, 3, 32, 32, 32], "float32"),
            }
        },
        debug=True,
    )
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_conv2d_dynamic():
    @R.function
    def forward(
        x: R.Tensor(("n", "c", "h", "w"), dtype="float32"),
        _io: R.Object,
        weight: R.Tensor((32, "in_channels", 3, 3), dtype="float32"),
        bias: R.Tensor((32,), dtype="float32"),
    ) -> R.Tuple(R.Tensor(("n", 32, "h - 2", "w - 2"), dtype="float32"), R.Tuple(R.Object)):
        n = T.int64()
        h = T.int64()
        w = T.int64()
        c = T.int64()
        in_channels = T.int64()
        R.func_attr({"num_input": 2})
        with R.dataflow():
            lv1: R.Tensor((n, 32, h - 2, w - 2), dtype="float32") = R.nn.conv2d(x, weight)
            lv2: R.Tensor((1, 32, 1, 1), dtype="float32") = R.reshape(bias, R.shape([1, 32, 1, 1]))
            conv2d: R.Tensor((n, 32, h - 2, w - 2), dtype="float32") = R.add(lv1, lv2)
            gv1: R.Tuple(R.Tensor((n, 32, h - 2, w - 2), dtype="float32"), R.Tuple(R.Object)) = (
                conv2d,
                (_io,),
            )
            R.output(gv1)
        return gv1

    mod = modules.Conv2D(tvm.tir.Var("in_channels", "int64"), 32, 3, bias=True)
    tvm_mod, _ = mod.export_tvm(
        spec={
            "forward": {
                "x": spec.Tensor(["n", "c", "h", "w"], "float32"),
            }
        },
        debug=True,
    )
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_rms_norm():
    @R.function
    def forward(
        x: R.Tensor((2, 4, 8), dtype="float32"),
        _io: R.Object,
        weight: R.Tensor((8,), dtype="float32"),
    ) -> R.Tuple(R.Tensor((2, 4, 8), dtype="float32"), R.Tuple(R.Object)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            rms_norm: R.Tensor((2, 4, 8), dtype="float32") = R.nn.rms_norm(
                x, weight, axes=[2], epsilon=1.0000000000000001e-05
            )
            gv1: R.Tuple(R.Tensor((2, 4, 8), dtype="float32"), R.Tuple(R.Object)) = rms_norm, (_io,)
            R.output(gv1)
        return gv1

    mod = modules.RMSNorm(8, [2], bias=False)
    tvm_mod, _ = mod.export_tvm(
        spec={"forward": {"x": spec.Tensor((2, 4, 8), "float32")}}, debug=True
    )
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_group_norm():
    @R.function
    def forward(
        x: R.Tensor((2, 4, 8), dtype="float32"),
        _io: R.Object,
        weight: R.Tensor((4,), dtype="float32"),
        bias: R.Tensor((4,), dtype="float32"),
    ) -> R.Tuple(R.Tensor((2, 4, 8), dtype="float32"), R.Tuple(R.Object)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            group_norm: R.Tensor((2, 4, 8), dtype="float32") = R.nn.group_norm(
                x, weight, bias, num_groups=2, channel_axis=1, axes=[2]
            )
            gv1: R.Tuple(R.Tensor((2, 4, 8), dtype="float32"), R.Tuple(R.Object)) = group_norm, (
                _io,
            )
            R.output(gv1)
        return gv1

    mod = modules.GroupNorm(num_groups=2, num_channels=4)
    tvm_mod, _ = mod.export_tvm(
        spec={"forward": {"x": spec.Tensor((2, 4, 8), "float32")}}, debug=True
    )
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_embedding():
    @R.function
    def forward(
        x: R.Tensor((1, 4), dtype="int32"),
        _io: R.Object,
        weight: R.Tensor((4, 8), dtype="float32"),
    ) -> R.Tuple(R.Tensor((1, 4, 8), dtype="float32"), R.Tuple(R.Object)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            reshape: R.Tensor((4,), dtype="int32") = R.reshape(x, R.shape([4]))
            take: R.Tensor((4, 8), dtype="float32") = R.take(weight, reshape, axis=0)
            reshape1: R.Tensor((1, 4, 8), dtype="float32") = R.reshape(take, R.shape([1, 4, 8]))
            gv1: R.Tuple(R.Tensor((1, 4, 8), dtype="float32"), R.Tuple(R.Object)) = reshape1, (_io,)
            R.output(gv1)
        return gv1

    mod = modules.Embedding(4, 8, "float32")
    tvm_mod, _ = mod.export_tvm(spec={"forward": {"x": spec.Tensor((1, 4), "int32")}}, debug=True)
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_timestep_embedding():
    @R.function
    def forward(
        sample: R.Tensor((32, 32), dtype="float32"),
        condition: R.Tensor((32, 16), dtype="float32"),
        _io: R.Object,
        linear_1_weight: R.Tensor((32, 32), dtype="float32"),
        linear_1_bias: R.Tensor((32,), dtype="float32"),
        cond_proj_weight: R.Tensor((32, 16), dtype="float32"),
        linear_2_weight: R.Tensor((32, 32), dtype="float32"),
        linear_2_bias: R.Tensor((32,), dtype="float32"),
    ) -> R.Tuple(R.Tensor((32, 32), dtype="float32"), R.Tuple(R.Object)):
        R.func_attr({"num_input": 3})
        with R.dataflow():
            permute_dims: R.Tensor((16, 32), dtype="float32") = R.permute_dims(
                cond_proj_weight, axes=None
            )
            matmul: R.Tensor((32, 32), dtype="float32") = R.matmul(
                condition, permute_dims, out_dtype="void"
            )
            add: R.Tensor((32, 32), dtype="float32") = R.add(sample, matmul)
            permute_dims1: R.Tensor((32, 32), dtype="float32") = R.permute_dims(
                linear_1_weight, axes=None
            )
            matmul1: R.Tensor((32, 32), dtype="float32") = R.matmul(
                add, permute_dims1, out_dtype="void"
            )
            add1: R.Tensor((32, 32), dtype="float32") = R.add(matmul1, linear_1_bias)
            silu: R.Tensor((32, 32), dtype="float32") = R.nn.silu(add1)
            permute_dims2: R.Tensor((32, 32), dtype="float32") = R.permute_dims(
                linear_2_weight, axes=None
            )
            matmul2: R.Tensor((32, 32), dtype="float32") = R.matmul(
                silu, permute_dims2, out_dtype="void"
            )
            add2: R.Tensor((32, 32), dtype="float32") = R.add(matmul2, linear_2_bias)
            gv1: R.Tuple(R.Tensor((32, 32), dtype="float32"), R.Tuple(R.Object)) = add2, (_io,)
            R.output(gv1)
        return gv1

    mod = modules.TimestepEmbedding(32, 32, cond_proj_dim=16)
    tvm_mod, _ = mod.export_tvm(
        spec={
            "forward": {
                "sample": spec.Tensor((32, 32), "float32"),
                "condition": spec.Tensor((32, 16), "float32"),
            }
        },
        debug=True,
    )
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_timesteps():
    @R.function
    def forward(
        x: R.Tensor((3,), dtype="float32"), _io: R.Object
    ) -> R.Tuple(R.Tensor((3, 10), dtype="float32"), R.Tuple(R.Object)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            lv1: R.Tensor((3,), dtype="float32") = R.astype(x, dtype="float32")
            lv2: R.Tensor((3, 1), dtype="float32") = R.expand_dims(lv1, axis=[1])
            lv3: R.Tensor((5,), dtype="float32") = R.arange(
                R.prim_value(0), R.prim_value(5), R.prim_value(1), dtype="float32"
            )
            lv4: R.Tensor((5,), dtype="float32") = R.multiply(
                R.const(-9.2103404998779297, "float32"), lv3
            )
            lv5: R.Tensor((5,), dtype="float32") = R.divide(lv4, R.const(4, "float32"))
            lv6: R.Tensor((5,), dtype="float32") = R.exp(lv5)
            lv7: R.Tensor((1, 5), dtype="float32") = R.expand_dims(lv6, axis=[0])
            lv8: R.Tensor((3, 5), dtype="float32") = R.multiply(lv2, lv7)
            lv9: R.Tensor((3, 5), dtype="float32") = R.sin(lv8)
            lv10: R.Tensor((3, 5), dtype="float32") = R.cos(lv8)
            lv11: R.Tensor((3, 10), dtype="float32") = R.concat((lv9, lv10), axis=-1)
            get_timestep_embedding: R.Tensor((3, 10), dtype="float32") = R.astype(
                lv11, dtype="float32"
            )
            gv1: R.Tuple(R.Tensor((3, 10), dtype="float32"), R.Tuple(R.Object)) = (
                get_timestep_embedding,
                (_io,),
            )
            R.output(gv1)
        return gv1

    mod = modules.Timesteps(10)
    tvm_mod, _ = mod.export_tvm(spec={"forward": {"x": spec.Tensor((3,), "float32")}}, debug=True)
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_kv_cache():
    @I.ir_module
    class Module:
        @R.function
        def _initialize_effect() -> R.Tuple(R.Object, R.Object):
            with R.dataflow():
                _io: R.Object = R.null_value()
                lv: R.Tensor((8, 2, 4), dtype="float32") = R.zeros(
                    R.shape([8, 2, 4]), dtype="float32"
                )
                cache: R.Object = R.call_pure_packed(
                    "vm.builtin.attention_kv_cache_create",
                    lv,
                    R.shape([8, 2, 4]),
                    R.prim_value(0),
                    sinfo_args=[R.Object()],
                )
                lv1 = _io, cache
                gv = lv1
                R.output(gv)
            return gv

        @R.function
        def forward(
            x: R.Tensor((2, 4), dtype="float32"), _io: R.Object, cache: R.Object
        ) -> R.Tuple(R.Tensor((4, 2, 4), dtype="float32"), R.Tuple(R.Object, R.Object)):
            R.func_attr({"num_input": 3})
            with R.dataflow():
                lv2: R.Object = R.call_inplace_packed(
                    "vm.builtin.attention_kv_cache_append",
                    cache,
                    x,
                    inplace_indices=[0],
                    sinfo_args=[R.Object()],
                )
                lv3: R.Tensor((4, 2, 4), dtype="float32") = R.call_pure_packed(
                    "vm.builtin.attention_kv_cache_view",
                    lv2,
                    R.shape([4, 2, 4]),
                    sinfo_args=(R.Tensor((4, 2, 4), dtype="float32"),),
                )
                gv1: R.Tuple(R.Tensor((4, 2, 4), dtype="float32"), R.Tuple(R.Object, R.Object)) = (
                    lv3,
                    (_io, lv2),
                )
                R.output(gv1)
            return gv1

    class KVCacheTest(modules.Module):
        def __init__(self) -> None:
            self.cache = modules.KVCache(8, [2, 4])

        def forward(self, x: core.Tensor) -> core.Tensor:
            self.cache.append(x)
            return self.cache.view(4)

    tvm_mod, _ = KVCacheTest().export_tvm(
        spec={"forward": {"x": spec.Tensor((2, 4), "float32")}}, debug=True
    )
    assert_structural_equal(tvm_mod, Module, True)


def test_attention():
    @R.function
    def forward(
        hidden_states: R.Tensor((2, 4096, 640), dtype="float32"),
        encoder_hidden_states: R.Tensor((2, 77, 2048), dtype="float32"),
        _io: R.Object,
        to_q_weight: R.Tensor((640, 640), dtype="float32"),
        to_k_weight: R.Tensor((640, 2048), dtype="float32"),
        to_v_weight: R.Tensor((640, 2048), dtype="float32"),
        group_norm_weight: R.Tensor((640,), dtype="float32"),
        group_norm_bias: R.Tensor((640,), dtype="float32"),
        to_out_0_weight: R.Tensor((640, 640), dtype="float32"),
        to_out_0_bias: R.Tensor((640,), dtype="float32"),
    ) -> R.Tuple(R.Tensor((2, 4096, 640), dtype="float32"), R.Tuple(R.Object)):
        R.func_attr({"num_input": 3})
        with R.dataflow():
            group_norm: R.Tensor((2, 4096, 640), dtype="float32") = R.nn.group_norm(
                hidden_states,
                group_norm_weight,
                group_norm_bias,
                num_groups=8,
                channel_axis=2,
                axes=[1],
                epsilon=1.0000000000000001e-05,
                center=True,
                scale=True,
            )
            permute_dims: R.Tensor((640, 640), dtype="float32") = R.permute_dims(
                to_q_weight, axes=None
            )
            matmul: R.Tensor((2, 4096, 640), dtype="float32") = R.matmul(
                group_norm, permute_dims, out_dtype="void"
            )
            permute_dims1: R.Tensor((2048, 640), dtype="float32") = R.permute_dims(
                to_k_weight, axes=None
            )
            matmul1: R.Tensor((2, 77, 640), dtype="float32") = R.matmul(
                encoder_hidden_states, permute_dims1, out_dtype="void"
            )
            permute_dims2: R.Tensor((2048, 640), dtype="float32") = R.permute_dims(
                to_v_weight, axes=None
            )
            matmul2: R.Tensor((2, 77, 640), dtype="float32") = R.matmul(
                encoder_hidden_states, permute_dims2, out_dtype="void"
            )
            reshape: R.Tensor((2, 4096, 10, 64), dtype="float32") = R.reshape(
                matmul, R.shape([2, 4096, 10, 64])
            )
            reshape1: R.Tensor((2, 77, 10, 64), dtype="float32") = R.reshape(
                matmul1, R.shape([2, 77, 10, 64])
            )
            reshape2: R.Tensor((2, 77, 10, 64), dtype="float32") = R.reshape(
                matmul2, R.shape([2, 77, 10, 64])
            )
            scaled_dot_product_attention: R.Tensor(
                (2, 4096, 10, 64), dtype="float32"
            ) = R.nn.attention(reshape, reshape1, reshape2, scale=None, causal_mask=None)
            reshape3: R.Tensor((2, 4096, 640), dtype="float32") = R.reshape(
                scaled_dot_product_attention, R.shape([2, 4096, 640])
            )
            permute_dims3: R.Tensor((640, 640), dtype="float32") = R.permute_dims(
                to_out_0_weight, axes=None
            )
            matmul3: R.Tensor((2, 4096, 640), dtype="float32") = R.matmul(
                reshape3, permute_dims3, out_dtype="void"
            )
            add: R.Tensor((2, 4096, 640), dtype="float32") = R.add(matmul3, to_out_0_bias)
            gv1: R.Tuple(R.Tensor((2, 4096, 640), dtype="float32"), R.Tuple(R.Object)) = add, (_io,)
            R.output(gv1)
        return gv1

    mod = modules.Attention(query_dim=640, cross_attention_dim=2048, heads=10, norm_num_groups=8)
    tvm_mod, _ = mod.export_tvm(
        spec={
            "forward": {
                "hidden_states": spec.Tensor((2, 4096, 640), "float32"),
                "encoder_hidden_states": spec.Tensor((2, 77, 2048), "float32"),
            }
        },
        debug=True,
    )
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_nn_module_tuple_input():
    class Layer(nn.Module):
        def __init__(self):
            pass

        def forward(self, x: Tuple[nn.Tensor, nn.Tensor]):
            x0 = x[0]
            x1 = x[1]
            y0 = nn.add(x0, x1)
            y1 = nn.subtract(x0, x1)
            return (y0, y1)

    # fmt: off
    @R.function
    def forward(x: R.Tuple(R.Tensor((10, 5), dtype="float32"), R.Tensor((10, 5), dtype="float32")), _io: R.Object) -> R.Tuple(R.Tuple(R.Tensor((10, 5), dtype="float32"), R.Tensor((10, 5), dtype="float32")), R.Tuple(R.Object)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            lv1: R.Tensor((10, 5), dtype="float32") = x[0]
            lv2: R.Tensor((10, 5), dtype="float32") = x[1]
            add: R.Tensor((10, 5), dtype="float32") = R.add(lv1, lv2)
            subtract: R.Tensor((10, 5), dtype="float32") = R.subtract(lv1, lv2)
            gv1: R.Tuple(R.Tuple(R.Tensor((10, 5), dtype="float32"), R.Tensor((10, 5), dtype="float32")), R.Tuple(R.Object)) = (add, subtract), (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    mod = Layer()
    tvm_mod, _ = mod.export_tvm(
        spec={
            "forward": {
                "x": (spec.Tensor([10, 5], dtype="float32"), spec.Tensor([10, 5], dtype="float32"))
            }
        },
        debug=True,
    )

    assert_structural_equal(tvm_mod["forward"], forward)


def test_nn_module_list_input():
    class Layer(nn.Module):
        def __init__(self):
            pass

        def forward(self, x: List[nn.Tensor]):
            x0 = x[0]
            x1 = x[1]
            y0 = nn.add(x0, x1)
            y1 = nn.subtract(x0, x1)
            return [y0, y1]

    # fmt: off
    @R.function
    def forward(x: R.Tuple(R.Tensor((10, 5), dtype="float32"), R.Tensor((10, 5), dtype="float32")), _io: R.Object) -> R.Tuple(R.Tuple(R.Tensor((10, 5), dtype="float32"), R.Tensor((10, 5), dtype="float32")), R.Tuple(R.Object)):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            lv1: R.Tensor((10, 5), dtype="float32") = x[0]
            lv2: R.Tensor((10, 5), dtype="float32") = x[1]
            add: R.Tensor((10, 5), dtype="float32") = R.add(lv1, lv2)
            subtract: R.Tensor((10, 5), dtype="float32") = R.subtract(lv1, lv2)
            gv1: R.Tuple(R.Tuple(R.Tensor((10, 5), dtype="float32"), R.Tensor((10, 5), dtype="float32")), R.Tuple(R.Object)) = (add, subtract), (_io,)
            R.output(gv1)
        return gv1
    # fmt: on

    mod = Layer()
    tvm_mod, _ = mod.export_tvm(
        spec={
            "forward": {
                "x": [spec.Tensor([10, 5], dtype="float32"), spec.Tensor([10, 5], dtype="float32")]
            }
        },
        debug=True,
    )

    assert_structural_equal(tvm_mod["forward"], forward)


def test_module_list():
    class Module(nn.Module):
        def __init__(self):
            self.layers = nn.ModuleList(
                [nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)]) for _ in range(1)]
            )

        def forward(self, x: nn.Tensor):
            return self.layers(x)

    mod = Module()
    named_params = dict(mod.named_parameters())
    assert ["layers.0.0.weight", "layers.0.1.weight"] == sorted(list(named_params.keys()))


if __name__ == "__main__":
    tvm.testing.main()
