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
"""Test gpu code verifier"""
import tvm
from tvm import te
from tvm import topi
import tvm.testing
import tvm.topi.testing


def get_verify_pass(valid, **kwargs):
    def _fverify(f, *_):
        valid[0] = tvm.tir.analysis.verify_gpu_code(f, kwargs)
        return f

    return tvm.tir.transform.prim_func_pass(_fverify, opt_level=0)


@tvm.testing.requires_gpu
def test_shared_memory():
    def check_shared_memory(storage_scope, dtype):
        N = 1024
        M = 128

        tvm_type = tvm.runtime.DataType(dtype)
        type_size = tvm_type.bits // 8 * tvm_type.lanes

        A = te.placeholder((N,), name="A", dtype=dtype)
        B = te.compute((N,), lambda i: A[i], name="B")

        s = te.create_schedule([B.op])
        AA = s.cache_read(A, storage_scope, [B])
        o, i = s[B].split(s[B].op.axis[0], M)
        s[AA].compute_at(s[B], o)
        s[B].bind(o, te.thread_axis("blockIdx.x"))
        s[B].bind(i, te.thread_axis("threadIdx.x"))

        # shared memory usage: M * sizeof(dtype) Bytes
        # thread usage: M

        for target in ["opencl", "cuda"]:
            if not tvm.testing.device_enabled(target):
                continue
            valid = [None]
            with tvm.transform.PassContext(
                config={
                    "tir.add_lower_pass": [
                        (
                            2,
                            get_verify_pass(
                                valid,
                                max_shared_memory_per_block=type_size * M - 1,
                                max_threads_per_block=M,
                            ),
                        )
                    ]
                }
            ):
                tvm.build(s, [A, B], target)
            assert not valid[0]

            with tvm.transform.PassContext(
                config={
                    "tir.add_lower_pass": [
                        (
                            2,
                            get_verify_pass(
                                valid,
                                max_shared_memory_per_block=type_size * M,
                                max_threads_per_block=M,
                            ),
                        )
                    ]
                }
            ):
                tvm.build(s, [A, B], target)
            assert valid[0]

    check_shared_memory("shared", "float32")
    check_shared_memory("shared", "int8x4")
    check_shared_memory("shared.dyn", "float32")


@tvm.testing.requires_gpu
def test_local_memory():
    N = 1024
    M = 128

    A = te.placeholder((N,), name="A", dtype="float32")
    B = te.compute((N,), lambda i: A[i], name="B")

    s = te.create_schedule([B.op])
    AA = s.cache_read(A, "local", [B])
    o, i = s[B].split(s[B].op.axis[0], M)
    s[AA].compute_at(s[B], o)
    s[B].bind(o, te.thread_axis("blockIdx.x"))

    # local memory usage: M * 4B
    # thread usage: M

    for target in ["opencl", "cuda"]:
        if not tvm.testing.device_enabled(target):
            continue

        valid = [None]
        with tvm.transform.PassContext(
            config={
                "tir.add_lower_pass": [
                    (
                        2,
                        get_verify_pass(
                            valid, max_local_memory_per_block=4 * M - 1, max_threads_per_block=1
                        ),
                    )
                ]
            }
        ):
            tvm.build(s, [A, B], target)
        assert not valid[0]

        with tvm.transform.PassContext(
            config={
                "tir.add_lower_pass": [
                    (
                        2,
                        get_verify_pass(
                            valid, max_local_memory_per_block=4 * M, max_threads_per_block=1
                        ),
                    )
                ]
            }
        ):
            tvm.build(s, [A, B], target)
        assert valid[0]


@tvm.testing.requires_gpu
def test_num_thread():
    N = 1024
    M = 128

    A = te.placeholder((N,), name="A", dtype="float32")
    B = te.compute((N,), lambda i: A[i], name="B")

    s = te.create_schedule([B.op])
    o, i = s[B].split(s[B].op.axis[0], M)

    s[B].bind(o, te.thread_axis("threadIdx.x"))
    s[B].bind(i, te.thread_axis("threadIdx.y"))

    # shared memory usage: 0
    # thread usage: N

    for target in ["opencl", "cuda"]:
        if not tvm.testing.device_enabled(target):
            continue

        valid = [None]
        with tvm.transform.PassContext(
            config={
                "tir.add_lower_pass": [
                    (
                        2,
                        get_verify_pass(
                            valid, max_shared_memory_per_block=0, max_threads_per_block=N - 1
                        ),
                    )
                ]
            }
        ):
            tvm.build(s, [A, B], target)
        assert not valid[0]

        with tvm.transform.PassContext(
            config={
                "tir.add_lower_pass": [
                    (
                        2,
                        get_verify_pass(
                            valid, max_shared_memory_per_block=0, max_threads_per_block=N
                        ),
                    )
                ]
            }
        ):
            tvm.build(s, [A, B], target)
        assert valid[0]

        with tvm.transform.PassContext(
            config={
                "tir.add_lower_pass": [
                    (
                        2,
                        get_verify_pass(
                            valid,
                            max_shared_memory_per_block=0,
                            max_threads_per_block=N,
                            max_thread_y=M - 1,
                        ),
                    )
                ]
            }
        ):
            tvm.build(s, [A, B], target)
        assert not valid[0]

        with tvm.transform.PassContext(
            config={
                "tir.add_lower_pass": [
                    (
                        2,
                        get_verify_pass(
                            valid,
                            max_shared_memory_per_block=0,
                            max_threads_per_block=N,
                            max_thread_y=M,
                        ),
                    )
                ]
            }
        ):
            tvm.build(s, [A, B], target)
        assert valid[0]


@tvm.testing.requires_gpu
def test_multiple_kernels():
    N = 1024

    A = te.placeholder((N, N), name="A")
    B = te.compute((N, N), lambda i, j: A[i, j])
    C = te.compute((N, N), lambda i, j: B[i, j])

    s = te.create_schedule([C.op])

    s[C].bind(s[C].op.axis[1], te.thread_axis("threadIdx.x"))
    s[B].bind(s[B].op.axis[1], te.thread_axis("threadIdx.x"))

    # shared memory usage: 0
    # thread usage: N

    for target in ["opencl", "cuda"]:
        if not tvm.testing.device_enabled(target):
            continue

        valid = [None]
        with tvm.transform.PassContext(
            config={
                "tir.add_lower_pass": [
                    (
                        2,
                        get_verify_pass(
                            valid, max_shared_memory_per_block=0, max_threads_per_block=N - 1
                        ),
                    )
                ]
            }
        ):
            tvm.build(s, [A, C], target)
        assert not valid[0]

        with tvm.transform.PassContext(
            config={
                "tir.add_lower_pass": [
                    (
                        2,
                        get_verify_pass(
                            valid, max_shared_memory_per_block=0, max_threads_per_block=N
                        ),
                    )
                ]
            }
        ):
            tvm.build(s, [A, C], target)
        assert valid[0]


@tvm.testing.requires_gpu
def test_wrong_bind():
    N = 1024

    A = te.placeholder((N, N - 1), name="A")
    B = te.compute((N, N - 1), lambda i, j: A[i, j])

    s = te.create_schedule([B.op])

    # bind a thread axis to two loop axes with different lengths
    s[B].bind(s[B].op.axis[0], te.thread_axis("threadIdx.x"))
    s[B].bind(s[B].op.axis[1], te.thread_axis("threadIdx.x"))

    for target in ["opencl", "cuda"]:
        if not tvm.testing.device_enabled(target):
            continue

        valid = [None]
        with tvm.transform.PassContext(
            config={
                "tir.add_lower_pass": [(2, get_verify_pass(valid, max_threads_per_block=N * N))]
            }
        ):
            tvm.build(s, [A, B], target)
        assert not valid[0]


@tvm.testing.requires_gpu
def test_vectorize():
    N = 1024

    A = te.placeholder((N, N), name="A")
    B = te.compute((N, N), lambda i, j: A[i, j])

    s = te.create_schedule([B.op])

    i, j = s[B].op.axis

    s[B].bind(i, te.thread_axis("blockIdx.x"))
    jo, ji = s[B].split(j, factor=64)
    s[B].bind(jo, te.thread_axis("threadIdx.x"))
    s[B].vectorize(ji)

    for target in ["opencl", "cuda"]:
        if not tvm.testing.device_enabled(target):
            continue

        valid = [None]
        with tvm.transform.PassContext(
            config={"tir.add_lower_pass": [(2, get_verify_pass(valid, max_vector_bytes=16))]}
        ):
            tvm.lower(s, [A, B])
        assert not valid[0]


@tvm.testing.requires_gpu
def test_vectorize_half():
    N = 1024

    A = te.placeholder((N, N), name="A", dtype="float16")
    B = te.compute((N, N), lambda i, j: A[i, j])

    s = te.create_schedule([B.op])

    i, j = s[B].op.axis

    s[B].bind(i, te.thread_axis("blockIdx.x"))
    jo, ji = s[B].split(j, factor=8)
    s[B].bind(jo, te.thread_axis("threadIdx.x"))
    s[B].vectorize(ji)

    for target in ["opencl", "cuda"]:
        if not tvm.testing.device_enabled(target):
            continue

        valid = [None]
        with tvm.transform.PassContext(
            config={"tir.add_lower_pass": [(2, get_verify_pass(valid, max_vector_bytes=16))]}
        ):
            tvm.lower(s, [A, B])
        assert valid[0]


@tvm.testing.requires_gpu
def test_vectorize_strided():
    N = 1024

    A = te.placeholder((N, N), name="A", dtype="float16")
    B = te.compute((N, N), lambda i, j: A[j, i])

    s = te.create_schedule([B.op])

    i, j = s[B].op.axis

    s[B].bind(i, te.thread_axis("blockIdx.x"))
    jo, ji = s[B].split(j, factor=8)
    s[B].vectorize(ji)

    for target in ["opencl", "cuda"]:
        if not tvm.testing.device_enabled(target):
            continue

        valid = [None]
        with tvm.transform.PassContext(
            config={"tir.add_lower_pass": [(2, get_verify_pass(valid, max_vector_bytes=16))]}
        ):
            tvm.lower(s, [A, B])
        assert not valid[0]


@tvm.testing.requires_gpu
def test_vthread():
    N = 1024

    A = te.placeholder((N, 16), name="A")
    B = te.compute((N, 16), lambda i, j: A[i, j])

    s = te.create_schedule([B.op])

    s[B].bind(s[B].op.axis[0], te.thread_axis("blockIdx.x"))
    s[B].bind(s[B].op.axis[1], te.thread_axis("vthread"))

    for target in ["opencl", "cuda"]:
        if not tvm.testing.device_enabled(target):
            continue

        valid = [None]

        for phase in [1, 2]:
            with tvm.transform.PassContext(
                config={"tir.add_lower_pass": [(phase, get_verify_pass(valid, max_vthread=16))]}
            ):
                tvm.build(s, [A, B], target)
            assert valid[0]

            with tvm.transform.PassContext(
                config={"tir.add_lower_pass": [(phase, get_verify_pass(valid, max_vthread=15))]}
            ):
                tvm.build(s, [A, B], target)
            assert not valid[0]


@tvm.testing.requires_gpu
def test_redundant_kernels():
    dtype = "float32"
    A = te.placeholder(shape=(1,), name="A", dtype=dtype)
    B = te.placeholder(shape=(1,), name="B", dtype=dtype)
    C = te.placeholder(shape=(1,), name="C", dtype=dtype)
    D = topi.less(A, C)
    E = topi.less(B, C)
    F = topi.logical_or(D, E)
    G = topi.identity(F)

    for target in ["opencl", "cuda"]:
        if not tvm.testing.device_enabled(target):
            continue
        print("Running on target: %s" % target)
        valid = [None]

        with tvm.target.Target(target):
            s = tvm.topi.testing.get_reduce_schedule(target)(G)

        with tvm.transform.PassContext(
            config={"tir.add_lower_pass": [(2, get_verify_pass(valid, max_kernels=1))]}
        ):
            tvm.build(s, [A, B, C, G], target)
        assert valid[0]


if __name__ == "__main__":
    tvm.testing.main()
