import json
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

def test_order_mutation_pass():
    x = sym.Variable('x')
    y = sym.conv2d(data=x, name='conv', dev='gpu')
    y = sym.add(y, x, name='add1')
    # write after read
    z = sym.assign(x, y, name='assign')
    # read after write
    t = sym.add(y, x, name='add2')
    g = graph.create(sym.Group([t, z]))
    jgraph = json.loads(g.apply(['OrderMutation', 'SaveJSON']).json_attr('json'))
    jnodes = jgraph['nodes']
    nindex = {n['name']: i for i, n in enumerate(jnodes)}
    assert nindex['assign'] in jnodes[nindex['add2']]['control_deps']
    assert nindex['conv'] in jnodes[nindex['assign']]['control_deps']
    assert nindex['add1'] in jnodes[nindex['assign']]['control_deps']
    assert jnodes[nindex['assign']]['inputs'][0][2] == 1

if __name__ == "__main__":
    test_order_mutation_pass()
    test_graph_json_attr()
    test_json_pass()
