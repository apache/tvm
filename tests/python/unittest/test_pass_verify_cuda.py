"""Test cuda verifier"""
import tvm

global valid

def cuda_verify_pass(max_shared_memory, max_num_thread):
    def verify_pass(stmt):
        global valid
        valid = tvm.ir_pass.VerifyCuda(stmt, max_shared_memory, max_num_thread)
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

    global valid
    with tvm.build_config(**{"add_lower_pass": [(2, cuda_verify_pass(4 * M - 1, M))]}):
        tvm.build(s, [A, B], 'cuda')
    assert not valid

    with tvm.build_config(**{"add_lower_pass": [(2, cuda_verify_pass(4 * M, M))]}):
        tvm.build(s, [A, B], 'cuda')
    assert valid


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

    global valid
    with tvm.build_config(**{"add_lower_pass": [(2, cuda_verify_pass(0, N - 1))]}):
        tvm.build(s, [A, B], 'cuda')
    assert not valid

    with tvm.build_config(**{"add_lower_pass": [(2, cuda_verify_pass(0, N))]}):
        tvm.build(s, [A, B], 'cuda')
    assert valid


if __name__ == "__main__":
    test_shared_memory()
    test_num_thread()
