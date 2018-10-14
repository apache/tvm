from nose.tools import raises
import tvm

@raises(Exception)
def test_loop_dependent_allocate():
    N = tvm.var("N")
    A = tvm.placeholder((2*N,), "float32", "A")
    C = tvm.compute((N, ), lambda i: A[2*i] + A[i+1], name='C')
    s = tvm.create_schedule(C.op)
    AA = s.cache_read(A, "local", [C])
    s[AA].compute_at(s[C], s[C].op.axis[0])
    # this line should fail due to IRUseDefAnalysis sees an allocate statement
    # referencing undefined variable
    tvm.lower(s, [A,C])

if __name__ == "__main__":
    test_loop_dependent_allocate()
