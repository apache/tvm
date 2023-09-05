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
"""Test sharded loader"""
# pylint: disable=missing-docstring
import json
import tempfile

import numpy as np

from tvm import dlight as dl
from tvm import relax as rx
from tvm._ffi import register_func
from tvm.contrib import tvmjs
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.target import Target


@register_func("tests.disco.shard_with_numpy", override=True)
def _shard_with_numpy(src, num_shards, tgt):
    s_0, s_1, s_2 = src.shape
    tgt.copyfrom(src.numpy().reshape(s_0, num_shards, s_1 // num_shards, s_2).transpose(1, 0, 2, 3))


def _create_loader(sess, path, param_dict, shard_info):
    path_ndarray_cache = path + "/ndarray-cache.json"
    tvmjs.dump_ndarray_cache(param_dict, path, encode_format="raw")
    with open(path_ndarray_cache, "r", encoding="utf-8") as i_f:
        ndarray_cache = i_f.read()
    loader_create = sess.get_global_func("runtime.disco.ShardLoader")
    shard_with_numpy = sess.get_global_func("tests.disco.shard_with_numpy")
    loader = loader_create(path_ndarray_cache, ndarray_cache, shard_info, shard_with_numpy)
    return loader


def test_load_shard():
    devices = [0, 1]
    param_dict = {
        "x_0": np.random.uniform(size=[64, 128]).astype("float16"),
        "x_1": np.random.uniform(size=[32, 128]).astype("float32"),
    }
    shard_info = json.dumps(
        {
            "x_0": 1,
            "x_1": 0,
        }
    )
    with tempfile.TemporaryDirectory() as path:
        sess = di.ThreadedSession(num_workers=len(devices))
        sess.init_ccl("nccl", *devices)
        loader = _create_loader(sess, path, param_dict, shard_info)
        loader_load = sess.get_global_func("runtime.disco.ShardLoaderLoad")
        d_0 = loader_load(loader, ShapeTuple([0]))
        d_1 = loader_load(loader, ShapeTuple([1]))
        np.testing.assert_equal(
            param_dict["x_0"][:, 0:64],
            d_0.debug_get_from_remote(0).numpy(),
        )
        np.testing.assert_equal(
            param_dict["x_0"][:, 64:128],
            d_0.debug_get_from_remote(1).numpy(),
        )
        np.testing.assert_equal(
            param_dict["x_1"][0:16, :],
            d_1.debug_get_from_remote(0).numpy(),
        )
        np.testing.assert_equal(
            param_dict["x_1"][16:32, :],
            d_1.debug_get_from_remote(1).numpy(),
        )


def test_load_shard_in_relax():
    devices = [0, 1]
    param_dict = {
        "x_0": np.random.uniform(size=[64, 128]).astype("float16"),
        "x_1": np.random.uniform(size=[32, 128]).astype("float32"),
    }
    shard_info = json.dumps(
        {
            "x_0": 1,
            "x_1": 0,
        }
    )

    # pylint: disable=invalid-name
    @I.ir_module
    class Module:  # pylint: disable=too-few-public-methods
        @R.function
        def main(
            loader: R.Object,
        ) -> R.Tuple(R.Tensor((64, 64), "float32"), R.Tensor((16, 128), "float32"),):
            R.func_attr({"global_symbol": "main"})
            with R.dataflow():
                lv0: R.Tensor((64, 64), "float32") = R.call_pure_packed(
                    "runtime.disco.ShardLoaderLoad",
                    loader,
                    R.shape([0]),
                    sinfo_args=R.Tensor((64, 64), "float32"),
                )
                lv1: R.Tensor((16, 128), "float32") = R.call_pure_packed(
                    "runtime.disco.ShardLoaderLoad",
                    loader,
                    R.shape([1]),
                    sinfo_args=R.Tensor((16, 128), "float32"),
                )
                lv2 = R.tuple(lv0, lv1)
                R.output(lv2)
            return lv2

    # pylint: enable=invalid-name
    def relax_build(mod, target):
        with target:
            mod = rx.get_pipeline("zero")(mod)  # pylint: disable=no-value-for-parameter
            mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            )(mod)
            return rx.build(mod, target="cuda")

    target = Target(
        {
            "kind": "cuda",
            "max_shared_memory_per_block": 49152,
            "max_threads_per_block": 1024,
            "thread_warp_size": 32,
            "registers_per_block": 65536,
            "arch": "sm_80",
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        dso_path = tmpdir + "/test.so"
        relax_build(Module, target).export_library(dso_path)
        sess = di.ThreadedSession(num_workers=len(devices))
        sess.init_ccl("nccl", *devices)
        mod = sess.load_vm_module(dso_path)
        loader = _create_loader(sess, tmpdir, param_dict, shard_info)
        result = mod["main"](loader)
        np.testing.assert_equal(
            param_dict["x_0"][:, 0:64],
            result.debug_get_from_remote(0)[0].numpy(),
        )
        np.testing.assert_equal(
            param_dict["x_0"][:, 64:128],
            result.debug_get_from_remote(1)[0].numpy(),
        )
        np.testing.assert_equal(
            param_dict["x_1"][0:16, :],
            result.debug_get_from_remote(0)[1].numpy(),
        )
        np.testing.assert_equal(
            param_dict["x_1"][16:32, :],
            result.debug_get_from_remote(1)[1].numpy(),
        )


def test_load_shard_all():
    devices = [0, 1]
    param_dict = {
        "param_0": np.random.uniform(size=[64, 128]).astype("float16"),
        "param_1": np.random.uniform(size=[32, 128]).astype("float32"),
    }
    shard_info = json.dumps(
        {
            "param_0": 1,
            "param_1": 0,
        }
    )
    with tempfile.TemporaryDirectory() as path:
        sess = di.ThreadedSession(num_workers=len(devices))
        sess.init_ccl("nccl", *devices)
        loader = _create_loader(sess, path, param_dict, shard_info)
        loader_load = sess.get_global_func("runtime.disco.ShardLoaderLoadAll")
        params = loader_load(loader)
        p_0 = params.debug_get_from_remote(0)
        p_1 = params.debug_get_from_remote(1)
        np.testing.assert_equal(param_dict["param_0"][:, 0:64], p_0[0].numpy())
        np.testing.assert_equal(param_dict["param_0"][:, 64:128], p_1[0].numpy())
        np.testing.assert_equal(param_dict["param_1"][0:16, :], p_0[1].numpy())
        np.testing.assert_equal(param_dict["param_1"][16:32, :], p_1[1].numpy())


def test_load_shard_broadcast():
    devices = [0, 1]
    param_dict = {
        "param_0": np.random.uniform(size=[64, 128]).astype("float16"),
        "param_1": np.random.uniform(size=[32, 128]).astype("float32"),
    }
    shard_info = "{}"
    with tempfile.TemporaryDirectory() as path:
        sess = di.ThreadedSession(num_workers=len(devices))
        sess.init_ccl("nccl", *devices)
        loader = _create_loader(sess, path, param_dict, shard_info)
        loader_load = sess.get_global_func("runtime.disco.ShardLoaderLoadAll")
        params = loader_load(loader)
        p_0 = params.debug_get_from_remote(0)
        p_1 = params.debug_get_from_remote(1)
        np.testing.assert_equal(param_dict["param_0"], p_0[0].numpy())
        np.testing.assert_equal(param_dict["param_0"], p_1[0].numpy())
        np.testing.assert_equal(param_dict["param_1"], p_0[1].numpy())
        np.testing.assert_equal(param_dict["param_1"], p_1[1].numpy())


if __name__ == "__main__":
    test_load_shard()
    test_load_shard_in_relax()
    test_load_shard_all()
    test_load_shard_broadcast()
