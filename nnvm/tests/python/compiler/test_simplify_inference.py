"""Unittest cases for simplify batch_norm"""
import nnvm
from nnvm import symbol as sym
from nnvm.compiler import graph_util, graph_attr

def test_simplify_batchnorm():
    def simple_bn(x, gamma, beta, moving_mean, moving_var,
                  axis=1, epsilon=1e-5, shape=None):
        # expect = (x - moving_mean) / sym.sqrt(moving_var + eps) * gamma + beta
        scale = sym.elemwise_mul(1 / sym.sqrt(moving_var + epsilon), gamma)
        shift = sym.elemwise_add(
            sym.elemwise_mul(sym.negative(moving_mean), scale), beta)
        shape = [-1 if i == axis else 1 for i in range(len(shape))]
        # for 2D
        scale = sym.reshape(scale, shape=shape)
        shift = sym.reshape(shift, shape=shape)
        return x * scale + shift


    # Before simplify
    def check(dim, axis, nstep):
        eps = 0.01
        x = sym.Variable("x") + 1
        beta = sym.Variable("beta")
        gamma = sym.Variable("gamma")
        moving_var = sym.Variable("moving_var")
        moving_mean = sym.Variable("moving_mean")
        y1, y2 = x, x
        ishape = {"x": tuple(10 for i in range(dim))}
        for i in range(nstep):
            y1 = sym.batch_norm(
                y1 + 1, gamma, beta, moving_mean, moving_var, epsilon=eps, axis=axis)
            y1 = sym.dropout(y1)
            y2 = simple_bn(y2 + 1, gamma, beta, moving_mean, moving_var,
                           epsilon=eps, axis=axis, shape=ishape["x"])
        g = nnvm.graph.create(y1)
        g2 = nnvm.graph.create(y2)
        graph_attr.set_shape_inputs(g, ishape)
        g1 = g.apply("InferShape").apply("SimplifyInference")
        # Some prints for debug
        # print(g1.ir())
        # assert graph equals as expected
        graph_util.check_graph_equal(g1, g2)

    check(2, 1, 1)
    check(4, 0, 3)

if __name__ == "__main__":
    test_simplify_batchnorm()
