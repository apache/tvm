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
"""Basic tests for a Disco nvshmem support"""

# pylint: disable=missing-docstring
import tempfile

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.support.popen_pool import PopenWorker
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di
from tvm.script import tirx as Tx

NUM_WORKERS = 4


def run_prim_func(sess, prim_func, *args):
    """Compile, export, load, and run a PrimFunc in the shared disco session."""
    target = tvm.target.Target("cuda")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test.so"
        mod = tvm.compile(prim_func, target=target, tir_pipeline="tirx")
        print(mod.mod.imports[0].inspect_source())
        mod.export_library(path)
        rt_mod = sess.load_vm_module(path)
        rt_mod["main"](*args)
        sess._sync_all()


def create_nvshmem_array(sess, shape, dtype, init_data_fn=None, zero_out=True):
    """Create and optionally initialize an nvshmem-accessible DNDArray."""
    nvshmem_empty = sess.get_global_func("runtime.disco.nvshmem.empty")
    arr = nvshmem_empty(ShapeTuple(shape), dtype, None)

    if init_data_fn:
        for i in range(NUM_WORKERS):
            arr.debug_copy_from(i, init_data_fn(i, shape, dtype))
    elif zero_out:
        zero_data = np.zeros(shape, dtype=dtype)
        for i in range(NUM_WORKERS):
            arr.debug_copy_from(i, zero_data)

    return arr


@pytest.mark.skip(reason="nvshmem doesn't work with pytest")
def test_codegen_nvshmem():
    def _test_func():
        ############ setup ############
        sess = di.ProcessSession(num_workers=NUM_WORKERS)
        f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
        uid = f_init_nvshmem_uid()
        init_dfunc = sess.get_global_func("runtime.disco.nvshmem.init_nvshmem")
        init_dfunc(uid, NUM_WORKERS, 0)
        sess.sync_worker_0()

        def test_thread_info(sess):
            @Tx.prim_func
            def main(res: Tx.Buffer((2,), "int32")):
                with Tx.kernel():
                    cta_id = Tx.cta_id([1])
                    tid = Tx.thread_id([nwarps * 32])
                    with Tx.thread():
                        res[0] = Tx.nvshmem.my_pe()
                        res[1] = Tx.nvshmem.n_pes()

            res_array = sess.empty((2,), "int32")
            run_prim_func(sess, main, res_array)

        def test_transfer(sess, scope, shape, nwarps, nelems, op_name):
            """Tests data transfer operations (get/put) at thread, warp, and block scopes."""
            dtype = "float32"
            is_get = "get" in op_name
            op_func = getattr(Tx.nvshmem, op_name)
            if scope != "thread":
                op_func = getattr(op_func, scope)

            # fmt: off
            @Tx.prim_func
            def main(A: Tx.Buffer(shape, dtype), B: Tx.Buffer(shape, dtype)):
                with Tx.kernel():
                    cta_id = Tx.cta_id([1])
                    warp_id = Tx.warp_id([nwarps])
                    lane_id = Tx.lane_id([32])
                    tid = Tx.thread_id([nwarps * 32])

                    with Tx.thread():
                        my_pe = Tx.nvshmem.my_pe()
                        n_pes = Tx.nvshmem.n_pes()
                        offset = Tx.if_then_else(
                            scope == "block", 0, Tx.if_then_else(scope == "thread", tid, warp_id * 32)  # noqa: E501
                        )
                        op_func(dst=B.ptr_to([offset]), src=A.ptr_to([offset]), nelems=nelems, pe=(my_pe + 1) % n_pes)  # noqa: E501
                        Tx.nvshmem.quiet()
            # fmt: on

            def init_fn(i, s, d):
                return np.arange(s[0], dtype=d) + i * 100

            A_array = create_nvshmem_array(sess, shape, dtype, init_fn)
            B_array = create_nvshmem_array(sess, shape, dtype)
            sess.sync_worker_0()
            run_prim_func(sess, main, A_array, B_array)

            for i in range(NUM_WORKERS):
                if is_get:
                    expected_B = A_array.debug_get_from_remote((i + 1) % NUM_WORKERS).numpy()
                    actual_B = B_array.debug_get_from_remote(i).numpy()
                else:  # put
                    expected_B = A_array.debug_get_from_remote(i).numpy()
                    actual_B = B_array.debug_get_from_remote((i + 1) % NUM_WORKERS).numpy()
                np.testing.assert_equal(actual_B, expected_B)

        def test_signal_op(sess, sig_op):
            """Tests signal_op and wait_until to implement a barrier-like pattern."""
            cmp_value = 1 if sig_op == "set" else 2

            # fmt: off
            @Tx.prim_func
            def main(res: Tx.Buffer((1,), "uint64")):
                with Tx.kernel():
                    cta_id = Tx.cta_id([1])
                    tid = Tx.thread_id([nwarps * 32])
                    with Tx.thread():
                        my_pe = Tx.nvshmem.my_pe()
                        n_pes = Tx.nvshmem.n_pes()
                        dst_pe = (my_pe + 1) % n_pes
                        if sig_op == "add":
                            res[0] = 1
                        Tx.nvshmem.barrier_all()
                        Tx.nvshmem.signal_op(sig_addr=res.ptr_to([0]), signal=1, sig_op=sig_op, pe=dst_pe)  # noqa: E501
                        Tx.nvshmem.wait_until(ivar=res.ptr_to([0]), cmp="eq", cmp_value=cmp_value)
            # fmt: on

            res_array = create_nvshmem_array(sess, (1,), "uint64")
            sess.sync_worker_0()
            run_prim_func(sess, main, res_array)

            for i in range(NUM_WORKERS):
                res = res_array.debug_get_from_remote(i).numpy()
                if sig_op == "set":
                    np.testing.assert_equal(res[0], 1)
                elif sig_op == "add":
                    np.testing.assert_equal(res[0], 2)

        def test_put_signal(sess, scope, shape, nwarps, nelems, cmp_value):
            """Tests combined data transfer and signal operations at thread/warp/block scopes."""
            dtype = "float32"
            op_func = getattr(Tx.nvshmem, "putmem_signal_nbi")
            if scope != "thread":
                op_func = getattr(op_func, scope)

            @Tx.prim_func
            def main(
                A: Tx.Buffer(shape, dtype),
                B: Tx.Buffer(shape, dtype),
                signal_array: Tx.Buffer((1,), "uint64"),
            ):
                with Tx.kernel():
                    cta_id = Tx.cta_id([1])
                    warp_id = Tx.warp_id([nwarps])
                    lane_id = Tx.lane_id([32])
                    tid = Tx.thread_id([nwarps * 32])

                    with Tx.thread():
                        my_pe = Tx.nvshmem.my_pe()
                        n_pes = Tx.nvshmem.n_pes()
                        dst_pe = (my_pe + 1) % n_pes
                        offset = Tx.if_then_else(
                            scope == "block",
                            0,
                            Tx.if_then_else(scope == "thread", tid, warp_id * 32),
                        )
                        op_func(
                            dst=B.access_ptr("w", offset=offset),
                            src=A.access_ptr("r", offset=offset),
                            nelems=nelems,
                            sig_addr=signal_array.access_ptr("w", offset=0),
                            signal=1,
                            sig_op="set",
                            pe=dst_pe,
                        )
                        Tx.nvshmem.wait_until(
                            ivar=signal_array.access_ptr("r", offset=0),
                            cmp="eq",
                            cmp_value=cmp_value,
                        )

            def init_A(i, s, d):
                return np.arange(s[0], dtype=d) + i * 100

            A_array = create_nvshmem_array(sess, shape, dtype, init_A)
            B_array = create_nvshmem_array(sess, shape, dtype)
            signal_array = create_nvshmem_array(sess, (1,), "uint64")

            sess.sync_worker_0()
            run_prim_func(sess, main, A_array, B_array, signal_array)

            for i in range(NUM_WORKERS):
                expected = A_array.debug_get_from_remote(i).numpy()
                actual = B_array.debug_get_from_remote((i + 1) % NUM_WORKERS).numpy()
                signal_np = signal_array.debug_get_from_remote(i).numpy()
                np.testing.assert_equal(actual, expected)
                np.testing.assert_equal(signal_np[0], cmp_value)

        def test_fence_barrier(sess):
            shape = (64,)
            dtype = "float32"

            # fmt: off
            @Tx.prim_func
            def main(A: Tx.Buffer(shape, dtype), B: Tx.Buffer(shape, dtype), res: Tx.Buffer((1,), "uint64")):  # noqa: E501
                with Tx.kernel():
                    cta_id = Tx.cta_id([1])
                    warp_id = Tx.warp_id([nwarps])
                    lane_id = Tx.lane_id([32])
                    tid = Tx.thread_id([2 * 32])

                    with Tx.thread():
                        my_pe = Tx.nvshmem.my_pe()
                        n_pes = Tx.nvshmem.n_pes()
                        dst_pe = (my_pe + 1) % n_pes
                        Tx.nvshmem.barrier_all()
                        Tx.nvshmem.putmem_nbi.block(dst=B.ptr_to([0]), src=A.ptr_to([0]), nelems=4 * 64, pe=(my_pe + 1) % n_pes)  # noqa: E501
                        Tx.nvshmem.fence()
                        if tid == 0:
                            Tx.nvshmem.signal_op(sig_addr=res.ptr_to([0]), signal=1, sig_op="set", pe=dst_pe)  # noqa: E501
                        Tx.nvshmem.wait_until(ivar=res.ptr_to([0]), cmp="eq", cmp_value=1)
            # fmt: on
            def init_fn(i, s, d):
                return np.arange(s[0], dtype=d) + i * 100

            A_array = create_nvshmem_array(sess, shape, dtype, init_fn)
            B_array = create_nvshmem_array(sess, shape, dtype)
            res_array = create_nvshmem_array(sess, (1,), "uint64")
            run_prim_func(sess, main, A_array, B_array, res_array)

            for i in range(NUM_WORKERS):
                expected_B = A_array.debug_get_from_remote(i).numpy()
                actual_B = B_array.debug_get_from_remote((i + 1) % NUM_WORKERS).numpy()
                np.testing.assert_equal(actual_B, expected_B)

        # test thread info
        test_thread_info(sess)
        print("\n\ntest_thread_info done\n\n")

        # test transfer
        for scope, shape, nwarps, nelems, op_name in [
            ("thread", (32,), 1, 4, "getmem_nbi"),
            ("thread", (32,), 1, 4, "putmem_nbi"),
            ("warp", (64,), 2, 4 * 32, "getmem_nbi"),
            ("warp", (64,), 2, 4 * 32, "putmem_nbi"),
            ("block", (64,), 2, 4 * 64, "getmem_nbi"),
            ("block", (64,), 2, 4 * 64, "putmem_nbi"),
        ]:
            test_transfer(sess, scope, shape, nwarps, nelems, op_name)
            print(f"\n\ntest_transfer done for {scope}, {shape}, {nwarps}, {nelems}, {op_name}\n\n")

        # test signal op
        for sig_op in ["set", "add"]:
            test_signal_op(sess, sig_op)
            print(f"\n\ntest_signal_op done for {sig_op}\n\n")

        # test put signal
        for scope, shape, nwarps, nelems, cmp_value in [
            ("thread", (32,), 1, 4, 32),
            ("warp", (64,), 2, 4 * 32, 2),
            ("block", (64,), 2, 4 * 64, 1),
        ]:
            test_put_signal(sess, scope, shape, nwarps, nelems, cmp_value)
            print(
                f"\n\ntest_put_signal done for {scope}, {shape}, {nwarps}, {nelems}, {cmp_value}\n\n"  # noqa: E501
            )

        # test fence barrier
        test_fence_barrier(sess)
        print("\n\ntest_fence_barrier done\n\n")

        ############ cleanup ############
        finalize_dfunc = sess.get_global_func("runtime.disco.nvshmem.finalize_nvshmem")
        finalize_dfunc()
        sess.sync_worker_0()
        return True

    p = PopenWorker()
    p.send(_test_func)
    assert p.recv()


if __name__ == "__main__":
    test_codegen_nvshmem()
