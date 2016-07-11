import nnvm.symbol as sym
import nnvm.graph as graph

def test_json_pass():
    x = sym.Variable('x')
    y = sym.conv2d(data=x, name='conv', stride=(2,2))
    g = graph.create(y)
    ret = g.apply('SaveJSON')
    g2 = ret.apply('LoadJSON')
    assert g2.apply('SaveJSON').attr('json') == ret.attr('json')


if __name__ == "__main__":
    test_json_pass()
