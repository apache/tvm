import json
import nnvm.symbol as sym
import nnvm.graph as graph
import nnvm.compiler.graph_util as graph_util

def test_json_pass():
    x = sym.Variable('x')
    y = sym.dense(data=x, name='conv', units=30)
    g = graph.create(y)
    ret = g.apply('SaveJSON')
    ret._set_json_attr('json', ret.json_attr('json'))
    g2 = ret.apply('LoadJSON')
    assert g2.apply('SaveJSON').json_attr('json') == ret.json_attr('json')
    json = g.json()
    g2 = graph.load_json(json)
    assert json == g2.json()


def test_json_pass_with_attr():
    x = sym.Variable('x')
    y = sym.dense(data=x, name='fc', units=30)
    g = graph.create(y)
    g._set_json_attr('version', '0.1.0')
    ret = g.apply('SaveJSON')
    json_str = ret.json_attr('json')
    ret._set_json_attr('json', json_str)
    g2 = ret.apply('LoadJSON')
    assert g2.json_attr('version') == '0.1.0'


def test_graph_json_attr():
    x = sym.Variable('x')
    y = sym.dense(data=x, name='fc', units=30)
    g = graph.create(y)
    g._set_json_attr('ilist', [1,2,3], 'list_int')
    assert g.json_attr('ilist') == [1,2,3]

def test_list_args():
    x = sym.Variable('x')
    z = sym.Variable('z')
    y = sym.dense(data=x, name='fc', units=30)
    y = sym.elemwise_add(y, z, name='add1')

def test_infer_shape():
    x = sym.Variable('x', shape=(2, 4, 2))
    y = sym.elemwise_add(x, x, name='add1')
    y = sym.flatten(y, name="flatten")
    g = graph.create(y)
    g._set_json_attr("shape_attr_key", "shape")
    g = g.apply('InferShape')
    jgraph = json.loads(g.apply('SaveJSON').json_attr('json'))
    jnodes = jgraph['nodes']
    jnode_row_ptr = jgraph['node_row_ptr']
    nindex = {n['name']: i for i, n in enumerate(jnodes)}
    assert g.json_attr('shape')[jnode_row_ptr[nindex["flatten"]]] == [2, 8]
    assert g.json_attr('shape')[jnode_row_ptr[nindex["add1"]]] == [2, 4, 2]

def test_infer_shape_known_partial():
    x = sym.Variable('x')
    y = sym.elemwise_add(x, x, name='add1')
    y = sym.flatten(y, name="flatten1")
    g = graph.create(y)
    jgraph = json.loads(g.apply('SaveJSON').json_attr('json'))
    shape = [[2, 4, 2], [] , []]
    g._set_json_attr("shape", shape, 'list_shape')
    g = g.apply("InferShape")
    jnodes = jgraph['nodes']
    jnode_row_ptr = jgraph['node_row_ptr']
    nindex = {n['name']: i for i, n in enumerate(jnodes)}
    assert g.json_attr('shape')[jnode_row_ptr[nindex["flatten1"]]] == [2, 8]
    assert g.json_attr('shape')[jnode_row_ptr[nindex["add1"]]] == [2, 4, 2]

def test_infer_type():
    x = sym.Variable('x', dtype=0)
    y = sym.elemwise_add(x, x, name='add1')
    y = sym.cast(y, dtype="float64", name="cast1")
    g = graph.create(y)
    g._set_json_attr("dtype_attr_key", "dtype")
    g = g.apply('InferType')
    jgraph = json.loads(g.apply('SaveJSON').json_attr('json'))
    jnodes = jgraph['nodes']
    jnode_row_ptr = jgraph['node_row_ptr']
    nindex = {n['name']: i for i, n in enumerate(jnodes)}
    assert g.json_attr('dtype')[jnode_row_ptr[nindex["cast1"]]] == 1
    assert g.json_attr('dtype')[jnode_row_ptr[nindex["add1"]]] == 0

def test_plan_memory():
    x = sym.Variable('x', shape=(4, 2))
    x2 = sym.elemwise_add(x, x, name='addk')
    y = sym.flatten(x2, name="reshapek")
    y = sym.elemwise_add(y, x2, name="add2")
    y = sym.elemwise_add(y, y)
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

def test_print_graph_ir():
    x = sym.Variable("x", shape=(1, 1, 10, 20))
    y = sym.conv2d(x + 1, name="y", channels=10, kernel_size=(3,3))
    g = graph.create(y)
    g = g.apply("InferShape")
    ir1 = g.ir()
    ir2 = g.ir(join_entry_attrs=["shape"])
    assert("y_bias" in ir1)
    assert("shape=" in ir2)

def test_gradient():
    x = sym.Variable("x")
    y = sym.Variable("y")
    z1 = sym.elemwise_add(x, sym.sqrt(y))
    z2 = sym.log(x)
    gradient = graph_util.gradients([z1, z2], [x, y])
    assert len(gradient) == 2

    g1 = sym.Variable("g1")
    g2 = sym.Variable("g2")
    grad_ys = [g1, g2]
    gradient = graph_util.gradients(sym.Group([z1, z2]),
                               sym.Group([x, y]), grad_ys=grad_ys)
    g_graph = graph.create(sym.Group(gradient)).ir()
    assert len(gradient) == 2
    assert "g1" in g_graph
    assert "g2" in g_graph

if __name__ == "__main__":
    test_print_graph_ir()
    test_json_pass_with_attr()
    test_graph_json_attr()
    test_json_pass()
    test_infer_shape()
    test_infer_shape_known_partial()
    test_infer_type()
    test_plan_memory()
    test_list_args()
    test_gradient()
