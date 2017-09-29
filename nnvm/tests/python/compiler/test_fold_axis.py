"""Unittest cases for fold_axis"""
import nnvm
from nnvm import symbol as sym
from nnvm.compiler import graph_util, graph_attr

def test_fold_axis_conv():
    def before(x, conv_weight, conv_bias, scale, channels):
        y = sym.conv2d(x, conv_weight, conv_bias,
                       channels=channels,
                       kernel_size=(3, 3),
                       padding=(1, 1),
                       name="conv")
        y = sym.relu(y)
        y = y * sym.expand_dims(scale, axis=1, num_newaxis=2)
        return y

    def expected(x, conv_weight, conv_bias, scale, channels):
        conv_weight = conv_weight * sym.expand_dims(scale, axis=1, num_newaxis=3)
        conv_bias = conv_bias * scale
        y = sym.conv2d(x,
                       conv_weight,
                       conv_bias,
                       channels=channels,
                       kernel_size=(3, 3),
                       padding=(1, 1),
                       name="conv")
        y = sym.relu(y)
        return y

    # Before simplify
    def check(shape, channels):
        x = sym.Variable("x") + 1
        weight = sym.Variable("weight")
        bias = sym.Variable("bias")
        scale = sym.Variable("scale")
        y1 = before(x, weight, bias, scale, channels)
        y2 = expected(x, weight, bias, scale, channels)
        ishape = {"x": shape, "scale": (channels,)}
        g1 = nnvm.graph.create(y1)
        g2 = nnvm.graph.create(y2)
        graph_attr.set_shape_inputs(g1, ishape)
        g1 = g1.apply("InferShape").apply("FoldScaleAxis")
        # assert graph equals as expected
        graph_util.check_graph_equal(g1, g2)

    check((2, 4, 10, 10), 2)

if __name__ == "__main__":
    test_fold_axis_conv()
