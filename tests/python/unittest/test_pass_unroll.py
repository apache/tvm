import tvm
import os

def test_unroll_loop():
    dtype = 'int64'
    n = tvm.var('n')
    Ab = tvm.decl_buffer((n, ), dtype)
    i = tvm.var('i')
    j = tvm.var('j')
    # for i in 0 to n-1:
    stmt = tvm.make.For(
        i, n, 2, 0, 0,
        tvm.make.For(j, 0, 8, 3, 0,
                     tvm.make.Store(Ab.data,
                                    tvm.make.Load(dtype, Ab.data, i) + 1,
                                    j + 1)))
    assert isinstance(stmt, tvm.stmt.For)
    ret = tvm.ir_pass.UnrollLoop(stmt, 16, 8, 0, True)
    assert not isinstance(ret, tvm.stmt.For)
    ret = tvm.ir_pass.UnrollLoop(stmt, 15, 8, 0, True)
    assert isinstance(ret, tvm.stmt.For)
    ret = tvm.ir_pass.UnrollLoop(stmt, 16, 8, 0, False)
    assert isinstance(ret, tvm.stmt.For)
    assert ret.for_type == tvm.stmt.For.Unrolled


if __name__ == "__main__":
    with tvm.build_config(dump_pass_ir=True):
        test_unroll_loop()

    def end_with(*suffix):
        ends = suffix
        def run(s):
            f = map(s.endswith, ends)
            if True in f: return s
        return run

    file_list = os.listdir('./')
    cc_file = end_with('.cc')
    cc_file = filter(cc_file, file_list)
    assert len(cc_file) == 3
    for i in cc_file:
        os.remove(i)
    
