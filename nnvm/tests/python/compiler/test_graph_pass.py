"""Unittest cases for graph pass"""
import nnvm
import nnvm.compiler
from nnvm.compiler import graph_pass

def test_infer_attr():
    x = nnvm.symbol.Variable("x")
    y = x * 2
    g = nnvm.graph.create(y)
    ishape, oshape = graph_pass.infer_shape(g, x=(10,20))
    assert tuple(oshape[0]) == (10, 20)

    itype, otype = graph_pass.infer_dtype(g, x="float32")
    assert otype[0] == "float32"


if __name__ == "__main__":
    test_infer_attr()
