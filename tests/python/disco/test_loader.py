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
import tempfile

import numpy as np
import json
import tvm
from tvm.contrib import tvmjs, utils
from tvm import relax as rx
from tvm._ffi import register_func
from tvm.runtime import ShapeTuple, String
from tvm.runtime import disco as di
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


def test_load_shard():
    devices = [1, 2]
    param_dict = {
        "x_0": np.random.uniform(size=[64, 128]).astype("float16"),
        "x_1": np.random.uniform(size=[32, 128]).astype("float32"),
    }
    shard_info = [
        {
            "name": "x_0",
            "shard_dim": 1,
        },
        {
            "name": "x_1",
            "shard_dim": 0,
        },
    ]

    @register_func("tests.disco.shard_with_numpy", override=True)
    def shard_with_numpy(src, start, length, tgt_ndarray):
        tgt_ndarray.copyfrom(src.numpy()[ : , start : start + length, : ])

    with tempfile.TemporaryDirectory() as path:
        path = "/tmp/tmp_junru/"
        path_ndarray_cache = path + "/ndarray-cache.json"
        path_shard_info =  path + "/shard-info.json"
        tvmjs.dump_ndarray_cache(param_dict, path, encode_format="raw")
        with open(path_shard_info, 'w') as o_f:
            json.dump(shard_info, o_f)

        sess = di.ThreadedSession(num_workers=len(devices))
        sess.init_ccl("nccl", *devices)

        loader_create = sess.get_global_func("runtime.disco.ShardLoader")
        loader_load = sess.get_global_func("runtime.disco.ShardLoaderLoad")
        shard_with_numpy = sess.get_global_func("tests.disco.shard_with_numpy")

        loader = loader_create(path_ndarray_cache, path_shard_info, shard_with_numpy)
        d_0 = loader_load(loader, 0)
        d_1 = loader_load(loader, 1)

        np.testing.assert_equal(
            param_dict["x_0"][ : , 0 : 64],
            d_0.debug_get_from_remote(0).numpy(),
        )
        np.testing.assert_equal(
            param_dict["x_0"][ : , 64 : 128],
            d_0.debug_get_from_remote(1).numpy(),
        )
        np.testing.assert_equal(
            param_dict["x_1"][0 : 16, : ],
            d_1.debug_get_from_remote(0).numpy(),
        )
        np.testing.assert_equal(
            param_dict["x_1"][16 : 32, : ],
            d_1.debug_get_from_remote(1).numpy(),
        )
        



if __name__ == "__main__":
    test_load_shard()
