"""Test gpu code verifier"""
import tvm

def get_verify_pass(valid, **kwargs):
    def verify_pass(stmt):
        valid[0] = tvm.ir_pass.VerifyGPUCode(stmt, kwargs)
        return stmt
    return verify_pass

def test_shared_memory():
    N = 1024
    M = 128

    A = tvm.placeholder((N,), name='A', dtype='float32')
    B = tvm.compute((N, ), lambda i: A[i], name='B')

    s = tvm.create_schedule([B.op])
    AA = s.cache_read(A, "shared", [B])
    o, i = s[B].split(s[B].op.axis[0], M)
    s[AA].compute_at(s[B], o)
    s[B].bind(o, tvm.thread_axis("blockIdx.x"))
    s[B].bind(i, tvm.thread_axis("threadIdx.x"))

    # shared memory usage: M * 4B
    # thread usage: M

    for target in ['opencl', 'cuda']:
        if not tvm.context(target).exist:
            continue
        valid = [None]
        with tvm.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_shared_memory_per_block=4 * M - 1,
                                max_threads_per_block=M))]}):
            tvm.build(s, [A, B], target)
        assert not valid[0]

        with tvm.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_shared_memory_per_block=4 * M,
                                max_threads_per_block=M))]}):
            tvm.build(s, [A, B], target)
        assert valid[0]

def test_local_memory():
    N = 1024
    M = 128

    A = tvm.placeholder((N,), name='A', dtype='float32')
    B = tvm.compute((N, ), lambda i: A[i], name='B')

    s = tvm.create_schedule([B.op])
    AA = s.cache_read(A, "local", [B])
    o, i = s[B].split(s[B].op.axis[0], M)
    s[AA].compute_at(s[B], o)
    s[B].bind(o, tvm.thread_axis("blockIdx.x"))

    # local memory usage: M * 4B
    # thread usage: M

    for target in ['opencl', 'cuda']:
        if not tvm.context(target).exist:
            continue

        valid = [None]
        with tvm.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_local_memory_per_block=4 * M - 1,
                                max_threads_per_block=1))]}):
            tvm.build(s, [A, B], target)
        assert not valid[0]

        with tvm.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_local_memory_per_block=4 * M,
                                max_threads_per_block=1))]}):
            tvm.build(s, [A, B], target)
        assert valid[0]

def test_num_thread():
    N = 1024
    M = 128

    A = tvm.placeholder((N,), name='A', dtype='float32')
    B = tvm.compute((N, ), lambda i: A[i], name='B')

    s = tvm.create_schedule([B.op])
    o, i = s[B].split(s[B].op.axis[0], M)

    s[B].bind(o, tvm.thread_axis('threadIdx.x'))
    s[B].bind(i, tvm.thread_axis("threadIdx.y"))

    # shared memory usage: 0
    # thread usage: N

    for target in ['opencl', 'cuda']:
        if not tvm.context(target).exist:
            continue

        valid = [None]
        with tvm.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_shared_memory_per_block=0,
                                max_threads_per_block=N - 1))]}):
            tvm.build(s, [A, B], target)
        assert not valid[0]

        with tvm.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_shared_memory_per_block=0,
                                max_threads_per_block=N))]}):
            tvm.build(s, [A, B], target)
        assert valid[0]

        with tvm.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_shared_memory_per_block=0,
                                max_threads_per_block=N,
                                max_thread_y=M-1))]}):
            tvm.build(s, [A, B], target)
        assert not valid[0]

        with tvm.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_shared_memory_per_block=0,
                                max_threads_per_block=N,
                                max_thread_y=M))]}):
            tvm.build(s, [A, B], target)
        assert valid[0]

def test_multiple_kernels():
    N = 1024

    A = tvm.placeholder((N, N), name='A')
    B = tvm.compute((N, N), lambda i, j: A[i, j])
    C = tvm.compute((N, N), lambda i, j: B[i, j])

    s = tvm.create_schedule([C.op])

    s[C].bind(s[C].op.axis[1], tvm.thread_axis("threadIdx.x"))
    s[B].bind(s[B].op.axis[1], tvm.thread_axis("threadIdx.x"))

    # shared memory usage: 0
    # thread usage: N

    for target in ['opencl', 'cuda']:
        if not tvm.context(target).exist:
            continue

        valid = [None]
        with tvm.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_shared_memory_per_block=0,
                                max_threads_per_block=N - 1))]}):
            tvm.build(s, [A, C], target)
        assert not valid[0]

        with tvm.build_config(**{"add_lower_pass": [
            (2, get_verify_pass(valid,
                                max_shared_memory_per_block=0,
                                max_threads_per_block=N))]}):
            tvm.build(s, [A, C], target)
        assert valid[0]

def test_wrong_bind():
    N = 1024

    A = tvm.placeholder((N, N-1), name='A')
    B = tvm.compute((N, N-1), lambda i, j: A[i, j])

    s = tvm.create_schedule([B.op])

    # bind a thread axis to two loop axes with different lengths
    s[B].bind(s[B].op.axis[0], tvm.thread_axis("threadIdx.x"))
    s[B].bind(s[B].op.axis[1], tvm.thread_axis("threadIdx.x"))

    for target in ['opencl', 'cuda']:
        if not tvm.context(target).exist:
            continue

        valid = [None]
        with tvm.build_config(**{"add_lower_pass": [
                (2, get_verify_pass(valid, max_threads_per_block=N*N))]}):
            tvm.build(s, [A, B], target)
        assert not valid[0]


if __name__ == "__main__":
    test_local_memory()
    test_shared_memory()
    test_num_thread()
    test_multiple_kernels()
    test_wrong_bind()
