"""
The following instruction is based on web/README.md.

Setup an RPC server:
$ python -m tvm.exec.rpc_proxy --example-rpc=1

Go to http://localhost:9190 in browser.

Click "Connect To Proxy".

Run this test script:
$ python tests/webgl/test_remote_save_load.py
"""

import numpy as np
import tvm
from tvm import rpc
from tvm.contrib import util, emscripten

proxy_host = "localhost"
proxy_port = 9090

def try_remote_save_load():
    if not tvm.module.enabled("rpc"):
        return
    if not tvm.module.enabled("opengl"):
        return
    if not tvm.module.enabled("llvm"):
        return

    # Build the module.
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = tvm.create_schedule(C.op)
    s[C].opengl()
    target_host = "llvm -target=asmjs-unknown-emscripten -system-lib"
    f = tvm.build(s, [A, B, C], "opengl", target_host=target_host, name="myadd")

    remote = rpc.connect(proxy_host, proxy_port, key="js")

    temp = util.tempdir()
    ctx = remote.opengl(0)
    path_obj = temp.relpath("myadd.bc")
    path_dso = temp.relpath("myadd.js")
    path_gl = temp.relpath("myadd.gl")
    path_json = temp.relpath("myadd.tvm_meta.json")

    f.save(path_obj)
    emscripten.create_js(path_dso, path_obj, side_module=True)
    f.imported_modules[0].save(path_gl)

    remote.upload(path_dso, "myadd.dso")
    remote.upload(path_gl)
    remote.upload(path_json)

    remote.download("myadd.dso")
    remote.download("myadd.gl")
    remote.download("myadd.tvm_meta.json")

    print('Loading myadd.dso')
    fhost = remote.load_module("myadd.dso")

    print('Loading myadd.gl')
    fdev = remote.load_module("myadd.gl")

    print('import_module')
    fhost.import_module(fdev)

    print('running...')
    a = tvm.nd.array(np.random.uniform(size=16).astype(A.dtype), ctx)
    b = tvm.nd.array(np.zeros(16, dtype=A.dtype), ctx)
    c = tvm.nd.array(np.zeros(16, dtype=C.dtype), ctx)
    fhost(a, b, c)
    np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

if __name__ == "__main__":
    try_remote_save_load()
