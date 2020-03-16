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

def get_verify_pass(valid, **kwargs):
    def verify_pass(stmt):
        valid[0] = tvm.tir.ir_pass.VerifyGPUCode(stmt, kwargs)
        return stmt
    return verify_pass

def test_shared_memory():
    def check_shared_memory(dtype):
        N = 1024
        M = 128

        tvm_type = tvm.runtime.DataType(dtype)
        type_size = tvm_type.bits // 8 * tvm_type.lanes

        A = te.placeholder((N,), name='A', dtype=dtype)
        B = te.compute((N, ), lambda i: A[i], name='B')

        s = te.create_schedule([B.op])
        AA = s.cache_read(A, "shared", [B])
        o, i = s[B].split(s[B].op.axis[0], M)
        s[AA].compute_at(s[B], o)
        s[B].bind(o, te.thread_axis("blockIdx.x"))
        s[B].bind(i, te.thread_axis("threadIdx.x"))

        # shared memory usage: M * sizeof(dtype) Bytes
        # thread usage: M

        for target in ['opencl', 'cuda']:
            if not tvm.context(target).exist:
                continue
            valid = [None]
            with tvm.target.build_config(**{"add_lower_pass": [
                (2, get_verify_pass(valid,
                                    max_shared_memory_per_block=type_size * M - 1,
                                    max_threads_per_block=M))]}):
                tvm.build(s, [A, B], target)
            assert not valid[0]

            with tvm.target.build_config(**{"add_lower_pass": [
                (2, get_verify_pass(valid,
                                    max_shared_memory_per_block=type_size * M,
                                    max_threads_per_block=M))]}):
                tvm.build(s, [A, B], target)
            assert valid[0]
    check_shared_memory('float32')
    check_shared_memory('int8x4')

def test_local_memory():
    N = 1024
    M = 128

    A = te.placeholder((N,), name='A', dtype='float32')
    B = te.compute((N, ), lambda i: A[i], name='B')

    s = te.create_schedule([B.op])
    AA = s.cache_read(A, "local", [B])
    o, i = s[B].split(s[B].op.axis[0], M)
    s[AA].compute_at(s[B], o)
    s[B].bind(o, te.thread_axis("blockIdx.x"))

    # local memory usage: M * 4B
    # thread usage: M

    for target in ['opencl', 'cuda']:
        if not tvm.context(target).exist:
            continue

        valid = [None]
        with tvm.target.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_local_memory_per_block=4 * M - 1,
                                max_threads_per_block=1))]}):
            tvm.build(s, [A, B], target)
        assert not valid[0]

        with tvm.target.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_local_memory_per_block=4 * M,
                                max_threads_per_block=1))]}):
            tvm.build(s, [A, B], target)
        assert valid[0]

def test_num_thread():
    N = 1024
    M = 128

    A = te.placeholder((N,), name='A', dtype='float32')
    B = te.compute((N, ), lambda i: A[i], name='B')

    s = te.create_schedule([B.op])
    o, i = s[B].split(s[B].op.axis[0], M)

    s[B].bind(o, te.thread_axis('threadIdx.x'))
    s[B].bind(i, te.thread_axis("threadIdx.y"))

    # shared memory usage: 0
    # thread usage: N

    for target in ['opencl', 'cuda']:
        if not tvm.context(target).exist:
            continue

        valid = [None]
        with tvm.target.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_shared_memory_per_block=0,
                                max_threads_per_block=N - 1))]}):
            tvm.build(s, [A, B], target)
        assert not valid[0]

        with tvm.target.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_shared_memory_per_block=0,
                                max_threads_per_block=N))]}):
            tvm.build(s, [A, B], target)
        assert valid[0]

        with tvm.target.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_shared_memory_per_block=0,
                                max_threads_per_block=N,
                                max_thread_y=M-1))]}):
            tvm.build(s, [A, B], target)
        assert not valid[0]

        with tvm.target.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_shared_memory_per_block=0,
                                max_threads_per_block=N,
                                max_thread_y=M))]}):
            tvm.build(s, [A, B], target)
        assert valid[0]

def test_multiple_kernels():
    N = 1024

    A = te.placeholder((N, N), name='A')
    B = te.compute((N, N), lambda i, j: A[i, j])
    C = te.compute((N, N), lambda i, j: B[i, j])

    s = te.create_schedule([C.op])

    s[C].bind(s[C].op.axis[1], te.thread_axis("threadIdx.x"))
    s[B].bind(s[B].op.axis[1], te.thread_axis("threadIdx.x"))

    # shared memory usage: 0
    # thread usage: N

    for target in ['opencl', 'cuda']:
        if not tvm.context(target).exist:
            continue

        valid = [None]
        with tvm.target.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_shared_memory_per_block=0,
                                max_threads_per_block=N - 1))]}):
            tvm.build(s, [A, C], target)
        assert not valid[0]

        with tvm.target.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_shared_memory_per_block=0,
                                max_threads_per_block=N))]}):
            tvm.build(s, [A, C], target)
        assert valid[0]

def test_wrong_bind():
    N = 1024

    A = te.placeholder((N, N-1), name='A')
    B = te.compute((N, N-1), lambda i, j: A[i, j])

    s = te.create_schedule([B.op])

    # bind a thread axis to two loop axes with different lengths
    s[B].bind(s[B].op.axis[0], te.thread_axis("threadIdx.x"))
    s[B].bind(s[B].op.axis[1], te.thread_axis("threadIdx.x"))

    for target in ['opencl', 'cuda']:
        if not tvm.context(target).exist:
            continue

        valid = [None]
        with tvm.target.build_config(**{"add_lower_pass": [
                (2, get_verify_pass(valid, max_threads_per_block=N*N))]}):
            tvm.build(s, [A, B], target)
        assert not valid[0]


if __name__ == "__main__":
    test_local_memory()
    test_shared_memory()
    test_num_thread()
    test_multiple_kernels()
    test_wrong_bind()
