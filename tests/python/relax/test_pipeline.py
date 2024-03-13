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
import tvm
import tvm.testing
from tvm import relax
from tvm.script import relax as R, tir as T


def test_pipeline_compile():
    pipeline = relax.get_pipeline()

    @tvm.script.ir_module
    class Mod:
        @R.function
        def main(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
            lv0 = R.add(x, y)
            return lv0

    mod = Mod
    mod = pipeline(mod)
    target = tvm.target.Target("llvm", host="llvm")

    ex = relax.build(mod, target)
    x_np = np.random.rand(3, 4).astype(np.float32)
    y_np = np.random.rand(3, 4).astype(np.float32)
    x = tvm.nd.array(x_np)
    y = tvm.nd.array(y_np)

    vm = relax.VirtualMachine(ex, tvm.cpu())
    z = vm["main"](x, y)
    tvm.testing.assert_allclose(z.numpy(), x_np + y_np, rtol=1e-7, atol=1e-7)


def test_pipeline_with_kv_cache():
    """A dummy pipline that simulates KV update."""
    pipeline = relax.get_pipeline()

    @tvm.script.ir_module
    class Mod:
        @R.function
        def create_kv_cache(reserve_slots: R.Shape(["m"])):
            # just allocate minimum slot since it is only used to signal dtype
            m = T.int64()
            init_data = R.ones((1, 4), "float32")
            kv_cache = R.call_pure_packed(
                "vm.builtin.attention_kv_cache_create",
                init_data,
                R.shape([m, 4]),
                0,
                sinfo_args=[R.Object()],
            )
            return kv_cache

        @R.function(pure=False)
        def main(
            x: R.Tensor((1, 4), "float32"),
            y: R.Tensor((1, 4), "float32"),
            shape: R.Shape(["L", 4]),
            kv_cache: R.Object,
        ):
            L = T.int64()
            # computation of the current value
            curr_value = R.add(x, y)
            # update cache
            kv_cache = R.call_packed(
                "vm.builtin.attention_kv_cache_append", kv_cache, curr_value, sinfo_args=[R.Object]
            )
            # return the updated cache view
            kv = R.call_packed(
                "vm.builtin.attention_kv_cache_view",
                kv_cache,
                shape,
                sinfo_args=[R.Tensor((L, 4), "float32")],
            )
            return (kv, kv_cache)

    mod = Mod
    mod = pipeline(mod)

    target = tvm.target.Target("llvm", host="llvm")

    ex = relax.build(mod, target)

    num_steps = 8
    cache_np = np.empty((num_steps, 4), dtype="float32")
    vm = relax.VirtualMachine(ex, tvm.cpu())

    kv_cache = vm["create_kv_cache"](tvm.runtime.ShapeTuple([1]))

    for i in range(num_steps):
        x_np = np.random.rand(1, 4).astype(np.float32)
        y_np = np.random.rand(1, 4).astype(np.float32)
        x = tvm.nd.array(x_np)
        y = tvm.nd.array(y_np)
        np_shape = (i + 1, 4)
        kv, kv_cache = vm["main"](x, y, tvm.runtime.ShapeTuple(np_shape), kv_cache)

        cache_np[i, :] = x_np + y_np
        tvm.testing.assert_allclose(kv.numpy(), cache_np[: np_shape[0], :], rtol=1e-7, atol=1e-7)
