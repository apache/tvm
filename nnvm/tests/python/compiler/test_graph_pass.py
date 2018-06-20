"""Unittest cases for graph pass"""
import nnvm
import nnvm.compiler
from nnvm import symbol as sym
from nnvm.compiler import graph_util, graph_attr

def test_infer_attr():
    x = sym.Variable("x")
    y = x * 2
    g = nnvm.graph.create(y)
    ishape, oshape = graph_util.infer_shape(g, x=(10,20))
    assert tuple(oshape[0]) == (10, 20)

    itype, otype = graph_util.infer_dtype(g, x="float32")
    assert otype[0] == "float32"

if __name__ == "__main__":
    test_infer_attr()
