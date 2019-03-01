import numpy as np

import tvm
from tvm import relay

from tvm._ffi.function import _init_api
_init_api("tvm.relay.build_module")

class BuildModule(object):
    def __init__(self):
        self.mod = relay.build_module._BuildModule()
        self._get_graph_json = self.mod["get_graph_json"]
        self._get_module = self.mod["get_module"]
        self._build = self.mod["build"]

    def build(self, func, target, target_host):
        self._build(func, target, target_host)

    def get_json(self):
        return self._get_graph_json()

    def get_module(self):
        return self._get_module()

def test_build():
    m_bld = BuildModule()
    # func
    a = relay.var("a", dtype="float32", shape=(2, 3))
    b = relay.var("b", dtype="float32", shape=(2, 3))
    x = relay.add(a, b)
    c = relay.var("c", dtype="float32", shape=(2, 3))
    y = relay.add(x, c)
    func = relay.Function([a, b, c], y)
    # build
    m_bld.build(func, "llvm", "llvm")
    g_json = m_bld.get_json()
    mmod = m_bld.get_module()
    # test
    A = tvm.nd.array(np.random.uniform(-1, 1, (2, 3)).astype("float32"))
    B = tvm.nd.array(np.random.uniform(-1, 1, (2, 3)).astype("float32"))
    C = tvm.nd.array(np.random.uniform(-1, 1, (2, 3)).astype("float32"))

    rt = tvm.contrib.graph_runtime.create(g_json, mmod, tvm.cpu())
    rt.set_input("a", A)
    rt.set_input("b", B)
    rt.set_input("c", C)
    rt.run()
    out = rt.get_output(0)

    np.testing.assert_allclose(out.asnumpy(), A.asnumpy() + B.asnumpy() + C.asnumpy())


