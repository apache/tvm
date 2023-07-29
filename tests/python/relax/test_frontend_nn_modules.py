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
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm.ir import assert_structural_equal
from tvm.relax.frontend.nn import core, modules, spec
from tvm.script import ir as I
from tvm.script import relax as R


def test_linear():
    @R.function
    def forward(
        x: R.Tensor((1, 4), dtype="float32"),
        weight: R.Tensor((8, 4), dtype="float32"),
        bias: R.Tensor((8,), dtype="float32"),
        _io: R.Object,
    ) -> R.Tuple(R.Tensor((1, 8), dtype="float32"), R.Tuple(R.Object)):
        with R.dataflow():
            permute_dims: R.Tensor((4, 8), dtype="float32") = R.permute_dims(weight, axes=None)
            matmul: R.Tensor((1, 8), dtype="float32") = R.matmul(x, permute_dims, out_dtype="void")
            add: R.Tensor((1, 8), dtype="float32") = R.add(matmul, bias)
            gv1: R.Tuple(R.Tensor((1, 8), dtype="float32"), R.Tuple(R.Object)) = add, (_io,)
            R.output(gv1)
        return gv1

    mod = modules.Linear(4, 8)
    tvm_mod, _ = mod.export_tvm(spec={"forward": {"x": spec.Tensor((1, 4), "float32")}})
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_rms_norm():
    @R.function
    def forward(
        x: R.Tensor((2, 4, 8), dtype="float32"),
        weight: R.Tensor((8,), dtype="float32"),
        _io: R.Object,
    ) -> R.Tuple(R.Tensor((2, 4, 8), dtype="float32"), R.Tuple(R.Object)):
        with R.dataflow():
            rms_norm: R.Tensor((2, 4, 8), dtype="float32") = R.nn.rms_norm(
                x, weight, axes=[2], epsilon=1.0000000000000001e-05
            )
            gv1: R.Tuple(R.Tensor((2, 4, 8), dtype="float32"), R.Tuple(R.Object)) = rms_norm, (_io,)
            R.output(gv1)
        return gv1

    mod = modules.RMSNorm(8, [2], bias=False)
    tvm_mod, _ = mod.export_tvm(spec={"forward": {"x": spec.Tensor((2, 4, 8), "float32")}})
    tvm_mod.show()
    assert_structural_equal(tvm_mod["forward"], forward, True)


def test_embedding():
    @R.function
    def forward(
        x: R.Tensor((1, 4), dtype="int32"),
        weight: R.Tensor((4, 8), dtype="float32"),
        _io: R.Object,
    ) -> R.Tuple(R.Tensor((1, 4, 8), dtype="float32"), R.Tuple(R.Object)):
        with R.dataflow():
            reshape: R.Tensor((4,), dtype="int32") = R.reshape(x, R.shape([4]))
            take: R.Tensor((4, 8), dtype="float32") = R.take(weight, reshape, axis=0)
            reshape1: R.Tensor((1, 4, 8), dtype="float32") = R.reshape(take, R.shape([1, 4, 8]))
            gv1: R.Tuple(R.Tensor((1, 4, 8), dtype="float32"), R.Tuple(R.Object)) = reshape1, (_io,)
            R.output(gv1)
        return gv1

    mod = modules.Embedding(4, 8, "float32")
    tvm_mod, _ = mod.export_tvm(spec={"forward": {"x": spec.Tensor((1, 4), "int32")}})
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
                cache: R.Object = R.call_packed(
                    "vm.builtin.attention_kv_cache_create",
                    lv,
                    R.shape([8, 2, 4]),
                    R.prim_value(0),
                    sinfo_args=(R.Object,),
                )
                lv1: R.Tuple(R.Object, R.Object) = _io, cache
                gv: R.Tuple(R.Object, R.Object) = lv1
                R.output(gv)
            return gv

        @R.function
        def forward(
            x: R.Tensor((2, 4), dtype="float32"), _io: R.Object, cache: R.Object
        ) -> R.Tuple(R.Tensor((4, 2, 4), dtype="float32"), R.Tuple(R.Object, R.Object)):
            with R.dataflow():
                lv2: R.Object = R.call_packed(
                    "vm.builtin.attention_kv_cache_append", cache, x, sinfo_args=(R.Object,)
                )
                lv3: R.Tensor((4, 2, 4), dtype="float32") = R.call_packed(
                    "vm.builtin.attention_kv_cache_view",
                    lv2,
                    R.shape([4, 2, 4]),
                    sinfo_args=(R.Tensor((4, 2, 4), dtype="float32"),),
                )
                gv1: R.Tuple(
                    R.Tensor((4, 2, 4), dtype="float32"), R.Tuple(R.Object, R.Object)
                ) = lv3, (_io, lv2)
                R.output(gv1)
            return gv1

    class KVCacheTest(modules.Module):
        def __init__(self) -> None:
            self.cache = modules.KVCache(8, [2, 4])

        def forward(self, x: core.Tensor) -> core.Tensor:
            self.cache.append(x)
            return self.cache.view(4)

    tvm_mod, _ = KVCacheTest().export_tvm(spec={"forward": {"x": spec.Tensor((2, 4), "float32")}})
    assert_structural_equal(tvm_mod, Module, True)


if __name__ == "__main__":
    tvm.testing.main()
