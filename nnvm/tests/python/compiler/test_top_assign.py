import numpy as np

import tvm
from tvm.contrib import graph_runtime

import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list


def test_update():
    w = sym.Variable("w")
    w2 = sym.Variable("w2")
    w = sym._assign(w, w + 1)
    w2 = sym._assign(w2, w + 1)

    dshape = (5, 3, 18, 18)
    shape_dict = {"w": dshape, "w2":dshape}
    dtype = "float32"

    def check(target, ctx):
        graph, lib, _ = nnvm.compiler.build(w2, target, shape_dict)

        m = graph_runtime.create(graph, lib, ctx)

        data = tvm.nd.array(np.random.uniform(size=dshape).astype(dtype))
        m.set_input("w", data)
        m.run()
        out = m.get_input("w2", tvm.nd.empty(dshape, dtype))
        np.testing.assert_allclose(out.asnumpy(), data.asnumpy() + 2, rtol=1e-5)

        m.run()
        out = m.get_input("w2", tvm.nd.empty(dshape, dtype))
        np.testing.assert_allclose(out.asnumpy(), data.asnumpy() + 3, rtol=1e-5)

    for target, ctx in ctx_list():
        check(target, ctx)


if __name__ == "__main__":
    test_update()
