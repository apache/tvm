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
import pytest

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T
from tvm.testing import env


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_nccl(), reason="need nccl")
def test_callback():
    """Simulate lazy loading of parameters in a callback

    The output of a lazy parameter loading, which would accept a
    callback to load the parameters.
    """

    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(private=True, s_tir=True)
        def slice_A(
            A: T.Buffer((4, 4), "int32"),
            rank: T.int64,
            A_sharded: T.Buffer((2, 4), "int32"),
        ):
            for i, j in T.grid(2, 4):
                with T.sblock("slice_A"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    A_sharded[vi, vj] = A[rank * 2 + vi, vj]

        @T.prim_func(private=True, s_tir=True)
        def slice_B(
            B: T.Buffer((2, 2), "float32"),
            rank: T.int64,
            B_sharded: T.Buffer((2, 1), "float32"),
        ):
            for i in range(2):
                with T.sblock("slice_B"):
                    vi = T.axis.spatial(2, i)
                    B_sharded[vi, 0] = B[vi, rank]

        @R.function
        def transform_params(
            rank_arg: R.Prim("int64"),
            fget_item: R.Callable([R.Any, R.Prim("int64")], R.Any),
        ):
            cls = Module

            A = fget_item(R.str("A"), R.prim_value(0))
            A = R.match_cast(A, R.Tensor([4, 4], "int32"))
            A = R.call_tir(
                cls.slice_A,
                (A, rank_arg),
                out_ty=R.Tensor([2, 4], "int32"),
            )

            B = fget_item(R.str("B"), R.prim_value(1))
            B = R.match_cast(B, R.Tensor([2, 2], "float32"))
            B = R.call_tir(
                cls.slice_B,
                (B, rank_arg),
                out_ty=R.Tensor([2, 1], "float32"),
            )

            return (A, B)

    pipeline = tvm.ir.transform.Sequential(
        [
            tvm.relax.transform.LegalizeOps(),
            tvm.s_tir.dlight.ApplyDefaultSchedule(tvm.s_tir.dlight.gpu.Fallback()),
        ],
        name="pipeline",
    )

    with tvm.target.Target("cuda"):
        mod = Module
        mod = pipeline(mod)
        built = tvm.compile(mod, "cuda")

    num_shards = 2

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = pathlib.Path(temp_dir)

        # TODO(Lunderberg): Update `disco.Session.load_vm_module` to
        # allow a `tvm.runtime.Module` argument.  This would avoid the
        # need for a temporary file.
        shlib_path = temp_dir.joinpath("libtemp.so")
        built.export_library(shlib_path)

        def run_and_check():
            session = tvm.runtime.disco.ProcessSession(num_workers=num_shards)
            try:
                session.import_python_module("tvm.exec.disco_worker")
                session.init_ccl("nccl", *range(num_shards))

                worker_device = session.get_global_func("runtime.disco.device")()
                worker_id = session.get_global_func("runtime.disco.worker_rank")()
                callback_maker = session.get_global_func("tests.disco.test_callback")
                fget_item = callback_maker(worker_device)
                vm = session.load_vm_module(shlib_path.as_posix())
                transform_params = vm["transform_params"]

                params = transform_params(worker_id, fget_item)

                # Worker 0 is the same PID as the controlling scope, so
                # `debug_get_from_remote(0)` returns the Tensor containing
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
                # `debug_get_from_remote(1)` returns a new Tensor within the
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
            finally:
                session.shutdown()

        tvm.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm.testing.main()
