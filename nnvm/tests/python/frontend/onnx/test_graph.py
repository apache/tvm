"""Test graph equality of onnx models."""
import nnvm
import onnx
from nnvm.compiler import graph_util, graph_attr
from model_zoo import super_resolution

def compare_graph(onnx_file, nnvm_sym, ishape):
    onnx_vars = [int(n) for n in onnx.__version__.split('.')] if hasattr(onnx, "__version__") else []
    if len(onnx_vars) >= 2 and (onnx_vars[0] > 0 or onnx_vars[1] >= 2):  # version >= 0.2
        onnx_model = onnx.load(onnx_file)
        onnx_sym, params = nnvm.frontend.from_onnx(onnx_model.graph)
    else:
        onnx_graph = onnx.load(onnx_file)
        onnx_sym, params = nnvm.frontend.from_onnx(onnx_graph)
    g1 = nnvm.graph.create(onnx_sym)
    g2 = nnvm.graph.create(nnvm_sym)
    ishapes = {'input_0': ishape}
    graph_attr.set_shape_inputs(g1, ishapes)
    graph_attr.set_shape_inputs(g2, ishapes)
    g1 = g1.apply("InferShape").apply("SimplifyInference")
    g2 = g2.apply("InferShape").apply("SimplifyInference")
    graph_util.check_graph_equal(g1, g2)

def test_super_resolution_example():
    fname, symbol = super_resolution
    compare_graph(fname, symbol, ishape=(1, 1, 224, 224))

if __name__ == '__main__':
    test_super_resolution_example()
