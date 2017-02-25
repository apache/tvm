import tvm
from tvm.addon import cc_compiler as cc, testing
import os
import numpy as np
import subprocess

runtime_py = """
import os
import sys
os.environ["TVM_USE_RUNTIME_LIB"] = "1"
import tvm
import numpy as np
path_dso = sys.argv[1]
dtype = sys.argv[2]
ff = tvm.module.load(path_dso)
a = tvm.nd.array(np.zeros(10, dtype=dtype))
ff(a)
np.testing.assert_equal(a.asnumpy(), np.arange(a.shape[0]))
print("Finish runtime checking...")
"""

def test_dso_module_load():
    if not tvm.codegen.enabled("llvm"):
        return
    dtype = 'int64'
    temp = testing.tempdir()

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

    path_obj = temp.relpath("test.o")
    path_ll = temp.relpath("test.ll")
    path_bc = temp.relpath("test.bc")
    path_dso = temp.relpath("test.so")
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

    path_runtime_py = temp.relpath("runtime.py")
    with open(path_runtime_py, "w") as fo:
        fo.write(runtime_py)

    subprocess.check_call(
        "python %s %s %s" % (path_runtime_py, path_dso, dtype),
        shell=True)

if __name__ == "__main__":
    test_dso_module_load()
