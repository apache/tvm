import nnvm.symbol as sym
import nnvm.graph as graph

def test_json_pass():
    x = sym.Variable('x')
    y = sym.conv2d(data=x, name='conv', stride=(2,2))
    g = graph.create(y)
    ret = g.apply('SaveJSON')
    ret._set_json_attr('json', ret.json_attr('json'))
    g2 = ret.apply('LoadJSON')
    assert g2.apply('SaveJSON').json_attr('json') == ret.json_attr('json')

def test_graph_json_attr():
    x = sym.Variable('x')
    y = sym.conv2d(data=x, name='conv', stride=(2,2))
    g = graph.create(y)
    g._set_json_attr('ilist', [1,2,3], 'list_int')
    assert g.json_attr('ilist') == [1,2,3]


if __name__ == "__main__":
    test_graph_json_attr()
    test_json_pass()
