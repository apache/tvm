"""Test graph equality of onnx models."""
import nnvm
import onnx
from nnvm.compiler import graph_util, graph_attr
from model_zoo import super_resolution, super_resolution_sym

def compare_graph(onnx_file, nnvm_sym, ishape):
    onnx_model = onnx.load(onnx_file)
    onnx_sym, params = nnvm.frontend.from_onnx(onnx_model)
    g1 = nnvm.graph.create(onnx_sym)
    g2 = nnvm.graph.create(nnvm_sym)
    ishapes = {'input_0': ishape}
    graph_attr.set_shape_inputs(g1, ishapes)
    graph_attr.set_shape_inputs(g2, ishapes)
    g1 = g1.apply("InferShape").apply("SimplifyInference")
    g2 = g2.apply("InferShape").apply("SimplifyInference")
    graph_util.check_graph_equal(g1, g2)

def test_super_resolution_example():
    fname, symbol = super_resolution, super_resolution_sym
    compare_graph(fname, symbol, ishape=(1, 1, 224, 224))

if __name__ == '__main__':
    test_super_resolution_example()
