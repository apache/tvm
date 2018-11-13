import json
import nnvm
import tvm

from nnvm import testing
from nnvm import symbol as sym
from tvm import autotvm
from tvm.autotvm.task import ConfigEntity
from tvm.autotvm.graph_tuner.utils import get_conv2d_NCHWc_AVX_workload, \
    is_elemlike_op, get_direct_ancestor, get_in_nodes, get_out_nodes, \
    shape2layout, get_wkl_map, get_real_node, infer_layout_shape_avx
from topi.nn.conv2d import conv2d_NCHWc


def create_workload(dshape, kshape, strides,
                    padding, layout, out_layout,
                    dtype, out_dtype):
    data = tvm.placeholder(dshape, dtype=dtype)
    kernel = tvm.placeholder(kshape, dtype=dtype)
    return autotvm.task.args_to_workload([data, kernel, strides, padding, layout,
                                          out_layout, out_dtype], conv2d_NCHWc)


def test_get_conv2d_workload():
    dshape = (1, 3, 224, 224)
    dtype = "float32"
    target = "llvm"
    net = testing.resnet.get_symbol(1000, num_layers=18)
    g = nnvm.graph.create(net)
    tasks = autotvm.task.extract_from_graph(net, target=target,
                                            shape={'data': dshape}, dtype=dtype,
                                            symbols=(sym.conv2d,))
    expected_wkl_list = []
    for task in tasks:
        data, kernel, strides, padding, _, layout, dtype = task.args
        data_plc = tvm.placeholder(data[1], name="data")
        kernel_plc = tvm.placeholder(kernel[1], name="kernel")
        workload = autotvm.task.args_to_workload([data_plc, kernel_plc, strides,
                                                  padding, layout, layout, dtype], conv2d_NCHWc)
        expected_wkl_list.append(workload)

    wkl_list = get_conv2d_NCHWc_AVX_workload(g, {"data": dshape})
    if len(wkl_list) != len(expected_wkl_list):
        raise RuntimeError("List length mismatch: expecting %d but got %d." %
                           (len(expected_wkl_list), len(wkl_list)))

    for wkl in wkl_list:
        if wkl not in expected_wkl_list:
            raise RuntimeError("%s falsely appear in resnet18." % (str(wkl)))


def test_get_wkl_map():
    dtype = "float32"
    wkl_list = [
        create_workload((1, 3, 224, 224), (64, 3, 7, 7), (2, 2), (3, 3), "NCHW", "NCHW", dtype, dtype),
        create_workload((1, 3, 56, 56), (64, 3, 3, 3), (1, 1), (1, 1), "NCHW", "NCHW", dtype, dtype),
        create_workload((1, 3, 256, 256), (64, 3, 1, 1), (1, 1), (0, 0), "NCHW", "NCHW", dtype, dtype),
        create_workload((1, 64, 256, 256), (32, 64, 3, 3), (1, 1), (1, 1), "NCHW", "NCHW", dtype, dtype),
        create_workload((1, 3, 28, 28), (64, 3, 3, 3), (1, 1), (1, 1), "NCHW", "NCHW", dtype, dtype),
        create_workload((1, 32, 256, 256), (16, 32, 3, 3), (1, 1), (2, 2), "NCHW", "NCHW", dtype, dtype)
    ]

    data = sym.Variable("data")
    conv0 = sym.conv2d(data, channels=64, kernel_size=(1, 1))
    conv1 = sym.conv2d(conv0, channels=32, kernel_size=(3, 3), padding=(1, 1))
    out = sym.conv2d(conv1, channels=16, kernel_size=(3, 3), padding=(2, 2))

    g = nnvm.graph.create(out)
    dshape = (1, 3, 256, 256)
    graph_wkl_list = get_conv2d_NCHWc_AVX_workload(g, {"data": dshape}, unique_wkl=False)
    node_map = get_wkl_map(g, wkl_list, "conv2d", graph_wkl_list)
    expected_out = {3: 2, 6: 3, 9: 5}
    diff_set = set(node_map) ^ set(expected_out)
    if len(diff_set) != 0:
        raise RuntimeError("Output mismatch: expecting %s but got %s." % (expected_out, node_map))


def test_get_real_node():
    data = sym.Variable("data")
    out1 = data * 2
    out2 = out1 + 4
    out3 = sym.elemwise_add(out2, data)
    out = sym.elemwise_add(out3, data)
    g = nnvm.graph.create(out)
    op_name = "__mul_scalar__"
    in_node_dict = get_in_nodes(g, op_name, "data")
    g_dict = json.loads(g.json())
    real_node = get_real_node(in_node_dict, g_dict["nodes"], 4, op_name)
    expected_out = 1
    if real_node != expected_out:
        raise RuntimeError("Output mismatch: expecting %d but got %d." % (expected_out, real_node))


def verify_shape2layout(shape, layout_template, expected_layout):
    layout = shape2layout(shape, layout_template)
    if layout != expected_layout:
        raise RuntimeError("Output mismatch: expecting %s but got %s." % (expected_layout, layout))


def test_shape2layout():
    verify_shape2layout((1, 3, 4, 4, 8), "NCHWc", "NCHW8c")
    verify_shape2layout((1, 3, 4, 4, 8), "NcHWC", "N3cHWC")


def verify_infer_layout_shape_avx(wkl, in_sch, out_sch, elemlike_shape,
                                  expected_in_shape, expected_out_shape):
    in_shape, out_shape, _ = infer_layout_shape_avx(wkl, in_sch, out_sch,
                                                    elemlike_shape)
    if in_shape != expected_in_shape:
        raise RuntimeError("Input shape mismatch: expecting %s but got %s."
                           % (str(expected_in_shape), str(in_shape)))
    if out_shape != expected_out_shape:
        raise RuntimeError("Output shape mismatch: expecting %s but got %s."
                           % (str(expected_out_shape), str(out_shape)))


def test_infer_layout_shape_avx():
    dtype = "float32"
    wkl = create_workload((1, 32, 224, 224), (64, 32, 7, 7), (2, 2), (3, 3), "NCHW", "NCHW", dtype, dtype)
    cfg_dict = {"i": -1,
                "c": None,
                "e": [["tile_ic", "sp", [2, 16]],
                      ["tile_oc", "sp", [2, 32]],
                      ["tile_ow", "sp", [56, 2]],
                      ["unroll_kw", "ot", True]],
                "t": ""}
    in_sch = ConfigEntity.from_json_dict(cfg_dict)
    out_sch = ConfigEntity.from_json_dict(cfg_dict)
    expected_in_shape = (1, 1, 224, 224, 32)
    expected_out_shape = (1, 2, 224, 224, 16)
    verify_infer_layout_shape_avx(wkl, in_sch, out_sch, None,
                                  expected_in_shape, expected_out_shape)


def verify_is_elemlike_op(node, expected_result):
    out = is_elemlike_op(node)
    if out != expected_result:
        raise RuntimeError("Output mismatch: expecting checking %s to be %s but got %s."
                           % (node["op"], str(expected_result), str(out)))


def test_is_elemlike_op():
    data = sym.Variable("data")
    out1 = data * 3
    out2 = sym.dense(data, units=10)
    out = sym.elemwise_add(out1, out2)
    g = nnvm.graph.create(out)
    g_dict = json.loads(g.json())
    node_list = g_dict["nodes"]
    verify_is_elemlike_op(node_list[2], False)
    verify_is_elemlike_op(node_list[4], False)
    verify_is_elemlike_op(node_list[5], True)


def test_get_direct_ancestor():
    data = sym.Variable("data")
    out1 = sym.dense(data, units=10)
    out2 = sym.elemwise_add(out1, data)
    out3 = out2 + 2.5
    out = sym.dense(out3, units=20)
    g = nnvm.graph.create(out)
    g_dict = json.loads(g.json())
    node_list = g_dict["nodes"]
    visited_dict = {}
    out = get_direct_ancestor(node_list, visited_dict, "dense", 8, [0])
    if len(out) != 1 or out[0] != 4:
        raise RuntimeError("Output mismatch: expecting [4] but got %s." % str(out))


def test_get_in_nodes():
    data = sym.Variable("data")
    out1 = sym.dense(data, units=10)
    out2 = sym.elemwise_add(out1, data)
    out3 = out2 + 2.5
    out = sym.dense(out3, units=20)
    g = nnvm.graph.create(out)
    input_names = ["data"]
    out = get_in_nodes(g, "dense", input_names)
    expected_out = {8: [4], 4: [3, 0], 3: [0]}
    diff_set = set(out) ^ set(expected_out)
    if len(diff_set) != 0:
        raise RuntimeError("Output mismatch: expecting %s but got %s." % (str(expected_out), str(out)))


def test_get_out_nodes():
    in_nodes_dict = {8: [4], 4: [3, 0], 3: [0]}
    expected_out = {0: [3, 4], 3: [4], 4: [8], 8: []}
    out = get_out_nodes(in_nodes_dict)
    diff_set = set(out) ^ set(expected_out)
    if len(diff_set) != 0:
        raise RuntimeError("Output mismatch: expecting %s but got %s." % (str(expected_out), str(out)))



if __name__ == "__main__":
    test_get_conv2d_workload()
    test_get_wkl_map()
    test_get_real_node()
    test_shape2layout()
    test_infer_layout_shape_avx()
    test_is_elemlike_op()
    test_get_direct_ancestor()
    test_get_in_nodes()
    test_get_out_nodes()
