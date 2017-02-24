import tvm
from tvm.addon import cc_compiler as cc
import os
import tempfile
import numpy as np

def test_dso_module_load():
    if not tvm.codegen.target_enabled("llvm"):
        return
    dtype = 'int64'
    temp_dir = tempfile.mkdtemp()

    def save_object(names):
        n = tvm.Var('n')
        Ab = tvm.Buffer((n, ), dtype)
        i = tvm.Var('i')
        # for i in 0 to n-1:
        stmt = tvm.make.For(
            i, 0, n - 1, 0, 0,
            tvm.make.Store(Ab.data,
                           tvm.make.Load(dtype, Ab.data, i) + 1,
                           i + 1))
        fapi = tvm.ir_pass.MakeAPI(stmt, "ramp", [Ab], 0)
        m = tvm.codegen.build(fapi, "llvm")
        for name in names:
            m.save(name)

    path_obj = "%s/test.o" % temp_dir
    path_ll = "%s/test.ll" % temp_dir
    path_bc = "%s/test.bc" % temp_dir
    path_dso = "%s/test.so" % temp_dir
    save_object([path_obj, path_ll, path_bc])
    cc.create_shared(path_dso, [path_obj])

    f1 = tvm.module.load(path_dso)
    f2 = tvm.module.load(path_dso)

    a = tvm.nd.array(np.zeros(10, dtype=dtype))
    f1(a)
    np.testing.assert_equal(a.asnumpy(), np.arange(a.shape[0]))
    a = tvm.nd.array(np.zeros(10, dtype=dtype))
    f2(a)
    np.testing.assert_equal(a.asnumpy(), np.arange(a.shape[0]))
    files = [path_obj, path_ll, path_bc, path_dso]
    for f in files:
        os.remove(f)
    os.rmdir(temp_dir)


def test_cuda_module_load():
    pass

if __name__ == "__main__":
    test_dso_module_load()
