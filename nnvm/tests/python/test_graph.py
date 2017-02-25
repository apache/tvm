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


def test_json_pass_with_attr():
    x = sym.Variable('x')
    y = sym.conv2d(data=x, name='conv', stride=(2,2))
    g = graph.create(y)
    g._set_json_attr('version', '0.1.0')
    ret = g.apply('SaveJSON')
    json_str = ret.json_attr('json')
    print(json_str)
    ret._set_json_attr('json', json_str)
    g2 = ret.apply('LoadJSON')
    assert g2.json_attr('version') == '0.1.0'


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

def test_list_args():
    x = sym.Variable('x')
    z = sym.Variable('z')
    y = sym.conv2d(data=x, name='conv', dev='gpu')
    y = sym.add(y, z, name='add1')
    # write after read
    z = sym.assign(x, y, name='assign')
    assert z.list_input_names('read_only') == ['conv_weight', 'z']
    assert z.list_input_names('aux_state') == ['x']

def test_infer_shape():
    x = sym.Variable('x', shape=(4, 2))
    y = sym.add(x, x, name='add1')
    y = sym.reshape(y, target=(2, 4), name="reshape1")
    g = graph.create(y)
    g._set_json_attr("shape_attr_key", "shape")
    g = g.apply('InferShape')
    jgraph = json.loads(g.apply('SaveJSON').json_attr('json'))
    jnodes = jgraph['nodes']
    jnode_row_ptr = jgraph['node_row_ptr']
    nindex = {n['name']: i for i, n in enumerate(jnodes)}
    assert g.json_attr('shape')[jnode_row_ptr[nindex["reshape1"]]] == [2, 4]
    assert g.json_attr('shape')[jnode_row_ptr[nindex["add1"]]] == [4, 2]

def test_infer_shape_known_partial():
    x = sym.Variable('x', shape=(4, 2))
    y = sym.add(x, x, name='add1')
    y = sym.reshape(y, target=(2, 4), name="reshape1")
    g = graph.create(y)
    jgraph = json.loads(g.apply('SaveJSON').json_attr('json'))
    shape = [[4, 2], [] , []]
    g._set_json_attr("shape", shape, 'list_shape')
    g = g.apply("InferShape")
    jnodes = jgraph['nodes']
    jnode_row_ptr = jgraph['node_row_ptr']
    nindex = {n['name']: i for i, n in enumerate(jnodes)}
    assert g.json_attr('shape')[jnode_row_ptr[nindex["reshape1"]]] == [2, 4]
    assert g.json_attr('shape')[jnode_row_ptr[nindex["add1"]]] == [4, 2]

def test_infer_type():
    x = sym.Variable('x', dtype=0)
    y = sym.add(x, x, name='add1')
    y = sym.cast(y, dtype=1, name="cast1")
    g = graph.create(y)
    g._set_json_attr("dtype_attr_key", "dtype")
    g = g.apply('InferType')
    jgraph = json.loads(g.apply('SaveJSON').json_attr('json'))
    jnodes = jgraph['nodes']
    jnode_row_ptr = jgraph['node_row_ptr']
    nindex = {n['name']: i for i, n in enumerate(jnodes)}
    assert g.json_attr('dtype')[jnode_row_ptr[nindex["cast1"]]] == 1
    assert g.json_attr('dtype')[jnode_row_ptr[nindex["add1"]]] == 0

def test_place_device():
    x = sym.Variable('x', device_group="stage1")
    y = sym.add(x, x, name='add1')
    y = sym.cast(y, dtype=1, name="cast1")
    z = sym.add(y, y, device_group="stage2", name="add2")
    z = sym.add(z, sym.exp(y, device_group="stage2"),  name="add3")
    g = graph.create(z)
    g._set_json_attr("device_group_attr_key", "device_group")
    g._set_json_attr("device_assign_map", {"stage1": 0, "stage2" : 1}, "dict_str_int")
    g._set_json_attr("device_copy_op", "cross_device_copy")
    g = g.apply("PlaceDevice")
    jgraph = json.loads(g.apply('SaveJSON').json_attr('json'))
    jnodes = jgraph['nodes']
    jnode_row_ptr = jgraph['node_row_ptr']
    nindex = {n['name']: i for i, n in enumerate(jnodes)}
    assert g.json_attr('device')[jnode_row_ptr[nindex["add2"]]] == 1
    assert g.json_attr('device')[jnode_row_ptr[nindex["add3"]]] == 1
    assert g.json_attr('device')[jnode_row_ptr[nindex["cast1"]]] == 0

def test_plan_memory():
    x = sym.Variable('x', shape=(4, 2))
    x2 = sym.add(x, x, name='addk')
    y = sym.reshape(x2, target=(2, 4), name="reshapek")
    y = sym.add(y, x2, name="add2")
    y = sym.add(y, y)
    g = graph.create(y)
    g._set_json_attr("shape_attr_key", "shape")
    g = g.apply(["InferShape", "InferType", "PlanMemory"])
    jgraph = json.loads(g.apply('SaveJSON').json_attr('json'))
    jnodes = jgraph['nodes']
    jnode_row_ptr = jgraph['node_row_ptr']
    storage_id = g.json_attr('storage_id')
    nindex = {n['name']: i for i, n in enumerate(jnodes)}
    assert (storage_id[jnode_row_ptr[nindex["addk"]]] !=
            storage_id[jnode_row_ptr[nindex["reshapek"]]])
    assert (storage_id[jnode_row_ptr[nindex["add2"]]] ==
            storage_id[jnode_row_ptr[nindex["reshapek"]]])


if __name__ == "__main__":
    test_json_pass_with_attr()
    test_order_mutation_pass()
    test_graph_json_attr()
    test_json_pass()
    test_infer_shape()
    test_infer_shape_known_partial()
    test_infer_type()
    test_place_device()
    test_plan_memory()
    test_list_args()
