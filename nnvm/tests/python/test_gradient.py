import json
import nnvm.symbol as sym
import nnvm.graph as graph

def grad(ys, xs, ys_grads):
    g = graph.create(ys)
    g._set_symbol_list_attr('grad_ys', ys)
    g._set_symbol_list_attr('grad_xs', xs)
    g._set_symbol_list_attr('grad_ys_out_grad', ys_grads)
    return g.apply('Gradient')

def test_graph_gradient():
    x0 = sym.Variable('x0')
    x1 = sym.Variable('x1')
    yg = sym.Variable('yg')
    y = sym.exp(sym.mul(x0, x1))
    grad_graph = grad(y, [x0], yg)
    print("Original graph")
    print(y.debug_str())
    print("Gradient  graph")
    print(grad_graph.symbol.debug_str())

if __name__ == "__main__":
    test_graph_gradient()
