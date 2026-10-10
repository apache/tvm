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
import pytest

from tvm.script import tirx as T
from tvm.tirx.analysis import verify_well_formed as verify


def test_root_scope():
    # fmt: off
    @T.function(check_well_formed=False)
    def test1() -> None:
        T.device_entry()

    @T.function(check_well_formed=False)
    def test2() -> None:
        pass

    @T.function(check_well_formed=False)
    def test3() -> None:
        pass

    @T.function(check_well_formed=False)
    def test4() -> None:
        T.device_entry()

        # fmt: on

    verify(test1)
    verify(test2)
    verify(test3)
    verify(test4)


def test_nested_scope():
    # fmt: off
    @T.function(check_well_formed=False)
    def test1() -> None:
        T.device_entry()

    @T.function(check_well_formed=False)
    def test2() -> None:
        T.device_entry()

    @T.function(check_well_formed=False)
    def test3() -> None:
        T.device_entry()
    @T.function(check_well_formed=False)
    def test4() -> None:
        T.device_entry()

        # fmt: on

    verify(test1)
    verify(test2)
    verify(test3)
    verify(test4)


def test_cuda_index_binds_are_independent():
    @T.function
    def kernel():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(8, 10, 12), block=128, cluster=(2, 2)))
        bx, by, bz = T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z")
        same_bx = T.cuda.block_idx("x")
        alias = bx
        T.evaluate(alias + same_bx + by + bz)

    verify(kernel)


def test_layout():
    ### TileLayout
    # fmt: off
    @T.function(check_well_formed=False)
    def test1():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(32,), block=4 * 32))
        _lane = T.cuda.lane_id()
        A = T.alloc_tensor((2,), layout=T.TileLayout(T.S[2, 1]))

        A[0] = 0
        # fmt: on
    verify(test1)

    ### ComposeLayout (bare swizzle)
    # fmt: off
    @T.function(check_well_formed=False)
    def test2():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(32,), block=4 * 32))
        _lane = T.cuda.lane_id()
        A = T.alloc_tensor(
            (512,), scope="shared", layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(512,)]))
        )

        A[0] = 0
        # fmt: on
    verify(test2)


def test_host():
    # fmt: off
    @T.function(check_well_formed=False)
    def test1(A: T.Tensor((16, 16), dtype='float32', align=16)):

        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float32", 2, A.data, 16, 16, 64, 16, 16, 1, 1, 0, 0, 0, 0)  # noqa: E501

        T.device_entry()
        for blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                bar = T.alloc_tensor((1,), "uint64", scope="shared", align=8)
                phase = T.alloc_tensor((1,), "int32", scope="local")
                A_smem = T.alloc_tensor((16, 16), "float32", scope="shared", align=128)

                phase[0] = 0
                if threadIdx == 0:
                    T.ptx.mbarrier.init.shared.b64(bar.data, T.uint32(1))
                    T.ptx.fence.proxy.async_.shared__cta()
                    T.ptx["cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](A_smem.data, T.address_of(A_map), 0, 0, bar.data)  # noqa: E501
                    T.ptx.mbarrier.arrive.expect_tx.shared.b64(bar.data, T.uint32(16*16*4))
                T.cuda.mbarrier_wait(bar.data, phase[0])
                phase[0] = phase[0] ^ 1
                T.evaluate(A_smem[0, 0])
        # fmt: on
    verify(test1)


def test_device_func():
    # Per-call exec-scope migration: scope is now attached per op via the
    # ``T.op[scope](...)`` subscription surface instead of a ``with T.cta():``
    # region. ``test1`` exercises a per-call-scoped op; ``test2`` the plain
    # (unscoped) op. The old multi-root-scope negative case asserted the removed
    # "only one root scope" verifier rule and no longer has an equivalent, so it
    # is dropped.
    # fmt: off
    @T.function(check_well_formed=False)
    def test1(A: T.Tensor((128,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(128,)))
        T.cuda.tile.mov(A, 0., scope='cta')

    @T.function(check_well_formed=False)
    def test2(A: T.Tensor((128,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(128,), block=(128,)))
        T.cuda.tile.mov(A, 0.)
    # fmt: on
    verify(test1)
    verify(test2)


def test_preferred_cluster_validation():
    T.cuda.LaunchConfig(grid=(8, 4), block=128, cluster=(2, 1), preferred_cluster=(2, 2))
    with pytest.raises(ValueError, match="requires an explicit cluster"):
        T.cuda.LaunchConfig(grid=8, block=128, preferred_cluster=2)
    with pytest.raises(ValueError, match="divisible"):
        T.cuda.LaunchConfig(grid=(8, 3), block=128, cluster=(2, 1), preferred_cluster=(2, 2))
    with pytest.raises(ValueError, match="cluster.z"):
        T.cuda.LaunchConfig(grid=(8, 4, 2), block=128, cluster=(2, 1), preferred_cluster=(2, 2, 2))
        # fmt: on


def test_index_calls_do_not_define_launch_extents():
    @T.function
    def kernel():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=8, block=128))
        bx = T.cuda.block_idx("x")
        lane = T.cuda.lane_id()
        T.evaluate(bx + lane)

    verify(kernel)


@pytest.mark.parametrize("grid,cluster", [(1, 2), (7, 2), ((8, 1), (2, 2))])
def test_invalid_launch_geometry(grid, cluster):
    with pytest.raises(ValueError, match="divisible"):
        T.cuda.LaunchConfig(grid=grid, block=128, cluster=cluster)


def test_tuple_indices_use_separate_binds():
    import tvm_ffi

    import tvm

    @T.function
    def kernel():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(2, 3, 4), block=32))
        bx, by, bz = T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z")
        T.evaluate(bx + by + bz)

    binds = []
    tvm_ffi.structural_walk(
        kernel.body, lambda node: binds.append(node) if isinstance(node, tvm.ir.Bind) else None
    )
    assert [bind.var.name for bind in binds] == ["bx", "by", "bz"]
    assert [bind.value.args[0].value for bind in binds] == ["x", "y", "z"]
    verify(kernel)


if __name__ == "__main__":
    import tvm.testing

    tvm.testing.main()
