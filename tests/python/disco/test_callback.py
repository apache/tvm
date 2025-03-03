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

import pathlib
import tempfile

import numpy as np

import tvm
import tvm.testing

from tvm.script import relax as R, tir as T


@tvm.testing.requires_nccl
def test_callback():
    """Simulate lazy loading of parameters in a callback

    The output of a lazy parameter loading, which would accept a
    callback to load the parameters.
    """

    @R.function
    def transform_params(
        rank_arg: R.Prim(value="rank"),
        fget_item: R.Callable([R.Object, R.Prim("int64")], R.Object),
    ):
        rank = T.int64()

        A = fget_item(R.str("A"), R.prim_value(0))
        A = R.match_cast(A, R.Tensor([4, 4], "int32"))
        A = R.strided_slice(A, axes=[0], begin=[rank * 2], end=[(rank + 1) * 2])

        B = fget_item(R.str("B"), R.prim_value(1))
        B = R.match_cast(B, R.Tensor([2, 2], "float32"))
        B = R.strided_slice(B, axes=[1], begin=[rank * 1], end=[(rank + 1) * 1])

        return (A, B)

    pipeline = tvm.ir.transform.Sequential(
        [
            tvm.relax.transform.LegalizeOps(),
            tvm.dlight.ApplyDefaultSchedule(tvm.dlight.gpu.Fallback()),
        ],
        name="pipeline",
    )

    with tvm.target.Target("cuda"):
        mod = tvm.IRModule.from_expr(transform_params)
        mod = pipeline(mod)
        built = tvm.relax.build(mod, "cuda")

    num_shards = 2

    session = tvm.runtime.disco.ProcessSession(num_workers=num_shards)
    session.import_python_module("tvm.exec.disco_worker")
    session.init_ccl("nccl", *range(num_shards))

    worker_device = session.get_global_func("runtime.disco.device")()
    worker_id = session.get_global_func("runtime.disco.worker_rank")()
    callback_maker = session.get_global_func("tests.disco.test_callback")
    fget_item = callback_maker(worker_device)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = pathlib.Path(temp_dir)

        # TODO(Lunderberg): Update `disco.Session.load_vm_module` to
        # allow a `tvm.runtime.Module` argument.  This would avoid the
        # need for a temporary file.
        shlib_path = temp_dir.joinpath("libtemp.so")
        built.export_library(shlib_path)
        vm = session.load_vm_module(shlib_path.as_posix())
        transform_params = vm["transform_params"]

        params = transform_params(worker_id, fget_item)

        # Worker 0 is the same PID as the controlling scope, so
        # `debug_get_from_remote(0)` returns the NDArray containing
        # the output.
        params_gpu0 = params.debug_get_from_remote(0)
        assert params_gpu0[0].device == tvm.cuda(0)
        assert params_gpu0[1].device == tvm.cuda(0)
        np.testing.assert_array_equal(
            params_gpu0[0].numpy(),
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ],
        )
        np.testing.assert_array_equal(
            params_gpu0[1].numpy(),
            [[0], [2]],
        )

        # Worker 1 is a different PID altogether, so
        # `debug_get_from_remote(1)` returns a new NDArray within the
        # calling scope's PID.
        params_gpu1 = params.debug_get_from_remote(1)
        assert params_gpu1[0].device == tvm.cpu()
        assert params_gpu1[1].device == tvm.cpu()
        np.testing.assert_array_equal(
            params_gpu1[0].numpy(),
            [
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ],
        )
        np.testing.assert_array_equal(
            params_gpu1[1].numpy(),
            [[1], [3]],
        )


if __name__ == "__main__":
    tvm.testing.main()
