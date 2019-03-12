import json
import tvm
import topi

from tvm import autotvm, relay
from tvm.relay.testing import resnet
from tvm.autotvm.task import ConfigEntity
from tvm.autotvm.graph_tuner.utils import get_conv2d_NCHWc_AVX_workload, \
    has_multiple_inputs, get_direct_ancestor, get_in_nodes, get_out_nodes, shape2layout, \
    get_wkl_map, infer_conv2d_layout_shape_avx, expr2graph
from topi.nn.conv2d import conv2d


def create_workload(dshape, kshape, strides,
                    padding, dilation, layout,
                    out_layout, dtype, out_dtype):
    data = tvm.placeholder(dshape, dtype=dtype)
    kernel = tvm.placeholder(kshape, dtype=dtype)
    return autotvm.task.args_to_workload([data, kernel, strides, padding, dilation, layout,
                                          out_dtype], conv2d)


def test_get_conv2d_workload():
    dshape = (1, 3, 224, 224)
    target = "llvm"
    expr, params = resnet.get_workload(num_layers=18, batch_size=1)
    tasks = autotvm.task.extract_from_program(expr,
                                              target=target,
                                              params=params,
                                              ops=(relay.op.nn.conv2d,))
    expected_wkl_list = []
    for task in tasks:
        data, kernel, strides, padding, dilation, layout, dtype = task.args
        data_plc = tvm.placeholder(data[1], name="data")
        kernel_plc = tvm.placeholder(kernel[1], name="kernel")
        workload = autotvm.task.args_to_workload([data_plc, kernel_plc, strides, padding,
                                                 dilation, layout, dtype], conv2d)
        expected_wkl_list.append(workload)

    wkl_list = get_conv2d_NCHWc_AVX_workload(expr, {"data": dshape})
    if len(wkl_list) != len(expected_wkl_list):
        raise RuntimeError("Get workload from relay expr failed: list length mismatch: expecting %d but got %d." %
                           (len(expected_wkl_list), len(wkl_list)))

    for wkl in wkl_list:
        if wkl not in expected_wkl_list:
            raise RuntimeError("Get workload from relay expr failed: %s falsely appear in resnet18." % (str(wkl)))


def test_get_wkl_map():
    dtype = "float32"
    wkl_list = [
        create_workload((1, 3, 224, 224), (64, 3, 7, 7), (2, 2), (3, 3), (1, 1), "NCHW", "NCHW", dtype, dtype),
        create_workload((1, 3, 56, 56), (64, 3, 3, 3), (1, 1), (1, 1), (1, 1), "NCHW", "NCHW", dtype, dtype),
        create_workload((1, 3, 256, 256), (64, 3, 1, 1), (1, 1), (0, 0), (1, 1), "NCHW", "NCHW", dtype, dtype),
        create_workload((1, 64, 256, 256), (32, 64, 3, 3), (1, 1), (1, 1), (1, 1), "NCHW", "NCHW", dtype, dtype),
        create_workload((1, 3, 28, 28), (64, 3, 3, 3), (1, 1), (1, 1), (1, 1), "NCHW", "NCHW", dtype, dtype),
        create_workload((1, 32, 256, 256), (16, 32, 3, 3), (1, 1), (2, 2), (1, 1), "NCHW", "NCHW", dtype, dtype)
    ]

    data = relay.var("data")
    w0 = relay.var("w0")
    conv0 = relay.nn.conv2d(data, w0, channels=64, kernel_size=(1, 1))
    w1 = relay.var("w1")
    conv1 = relay.nn.conv2d(conv0, w1, channels=32, kernel_size=(3, 3), padding=(1, 1))
    w2 = relay.var("w2")
    out = relay.nn.conv2d(conv1, w2, channels=16, kernel_size=(3, 3), padding=(2, 2))
    net = relay.Function(relay.ir_pass.free_vars(out), out)
    node_list = []
    node_dict = {}
    expr2graph(net, node_dict, node_list)

    dshape = (1, 3, 256, 256)
    graph_wkl_list = get_conv2d_NCHWc_AVX_workload(net, {"data": dshape}, unique_wkl=False)
    node_map = get_wkl_map(node_list, wkl_list, "conv2d", graph_wkl_list)
    expected_out = {4: 2, 5: 3, 6: 5}
    diff_set = set(node_map) ^ set(expected_out)
    if len(diff_set) != 0:
        raise RuntimeError("Output mismatch: expecting %s but got %s." % (expected_out, node_map))


def verify_shape2layout(shape, layout_template, expected_layout):
    layout = shape2layout(shape, layout_template)
    if layout != expected_layout:
        raise RuntimeError("Output mismatch: expecting %s but got %s." % (expected_layout, layout))


def test_shape2layout():
    verify_shape2layout((1, 3, 4, 4, 8), "NCHWc", "NCHW8c")
    verify_shape2layout((1, 3, 4, 4, 8), "NcHWC", "N3cHWC")


def verify_infer_layout_shape_avx(wkl, in_sch, out_sch, elemlike_shape,
                                  expected_in_shape, expected_out_shape):
    in_shape, out_shape, _ = infer_conv2d_layout_shape_avx(wkl, in_sch, out_sch,
                                                    elemlike_shape)
    if in_shape != expected_in_shape:
        raise RuntimeError("Input shape mismatch: expecting %s but got %s."
                           % (str(expected_in_shape), str(in_shape)))
    if out_shape != expected_out_shape:
        raise RuntimeError("Output shape mismatch: expecting %s but got %s."
                           % (str(expected_out_shape), str(out_shape)))


def test_infer_layout_shape_avx():
    dtype = "float32"
    wkl = create_workload((1, 32, 224, 224), (64, 32, 7, 7), (2, 2), (3, 3), (1, 1), "NCHW", "NCHW", dtype, dtype)
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


def verify_has_multiple_inputs(node_list, node_idx, input_names, expected_result):
    out = has_multiple_inputs(node_list, node_idx, input_names)
    if out != expected_result:
        raise RuntimeError("Output mismatch: expecting checking %s to be %s but got %s."
                           % (node_list[node_idx]["op"], str(expected_result), str(out)))


def test_has_multiple_inputs():
    data = relay.var("data")
    out1 = data * relay.expr.const(3)
    w0 = relay.var("w0")
    out2 = relay.nn.dense(data, w0, units=10)
    out = relay.add(out1, out2)
    net = relay.Function(relay.ir_pass.free_vars(out), out)
    node_list = []
    node_dict = {}
    expr2graph(net, node_dict, node_list)
    input_names = ["data"]
    verify_has_multiple_inputs(node_list, 2, input_names, False)
    verify_has_multiple_inputs(node_list, 4, input_names, False)
    verify_has_multiple_inputs(node_list, 5, input_names, True)


def test_expr2graph():
    net, _ = resnet.get_workload(num_layers=50, batch_size=1)
    node_dict = {}
    node_list = []
    expected_node_list = []
    def _count_node(node):
        if not isinstance(node, (relay.op.op.Op, relay.expr.TupleGetItem,
                                 relay.expr.Function)):
            expected_node_list.append(node)
    relay.ir_pass.post_order_visit(net, _count_node)

    expr2graph(net, node_dict, node_list)
    for expected_node, node in zip(expected_node_list, node_list):
        if expected_node != node["node"]:
            raise RuntimeError("Node mismatch: expecting %s but got %s"
                               % (str(expected_node), str(node)))


def test_get_direct_ancestor():
    data = relay.var("data")
    w0 = relay.var("w0")
    out1 = relay.nn.dense(data, w0, units=10)
    out2 = relay.add(out1, data * relay.expr.const(5))
    out3 = out2 + relay.expr.const(2.5)
    w1 = relay.var("w1")
    out = relay.nn.dense(out3, w1, units=20)
    net = relay.Function(relay.ir_pass.free_vars(out), out)
    node_list = []
    node_dict = {}
    expr2graph(net, node_dict, node_list)
    visited_dict = {}
    input_names = ["data"]
    out = get_direct_ancestor(node_list, visited_dict, "dense", 6, input_names)
    if len(out) != 2 or out != [3, 0]:
        raise RuntimeError("Output mismatch: expecting [3, 0] but got %s." % str(out))


def test_get_in_nodes():
    data = relay.var("data")
    w0 = relay.var("w0")
    out1 = relay.nn.dense(data, w0, units=10)
    out2 = relay.add(out1, data)
    out3 = out2 + relay.expr.const(2.5)
    w1 = relay.var("w1")
    out = relay.nn.dense(out3, w1, units=20)
    net = relay.Function(relay.ir_pass.free_vars(out), out)
    input_names = ["data"]
    out = get_in_nodes(net, "dense", input_names)
    expected_out = {7: [4], 4: [3, 0], 3: [0]}
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
    test_shape2layout()
    test_infer_layout_shape_avx()
    test_has_multiple_inputs()
    test_expr2graph()
    test_get_direct_ancestor()
    test_get_in_nodes()
    test_get_out_nodes()
