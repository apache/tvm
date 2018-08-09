import os
import json
import numpy as np
import nnvm
import tvm

from nnvm import testing
from nnvm import symbol as sym
from topi.nn.conv2d import Workload
from topi.x86.conv2d_avx_common import AVXConvCommonFwd
from topi.x86.conv2d_avx_1x1 import AVXConv1x1Fwd
from tvm.autotvm.graph_tuner.utils import get_conv2d_workload, \
    run_remote_module, is_elemlike_op, get_direct_ancestor, get_in_nodes, get_out_nodes, \
    get_factor, str2namedtuple, read_sch_from_json, write_sch_to_json, \
    shape2layout, get_wkl_map, get_real_node, infer_layout_shape_avx


def test_get_conv2d_workload():
    expected_wkl = [
        Workload('float32', 'float32', 224, 224, 3, 64, 7, 7, 3, 3, 2, 2),
        Workload('float32', 'float32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
    ]
    dshape = (1, 3, 224, 224)
    sym = testing.resnet.get_symbol(1000, num_layers=18)
    g = nnvm.graph.create(sym)
    wkl_list = get_conv2d_workload(g, {"data": dshape})
    if len(wkl_list) != len(expected_wkl):
        raise RuntimeError("List length mismatch: expecting %d but got %d." %
                           (len(expected_wkl), len(wkl_list)))
    for wkl in wkl_list:
        if wkl not in expected_wkl:
            raise RuntimeError("%s falsely appear in resnet18." % (str(wkl)))


def test_read_sch_from_json():
    sch_json_file = "test_read_sch_from_json.json"
    sch_dict_str = {
        "schedules": {
            "Workload(in_dtype='float32', out_dtype='float32', "
            "height=224, width=224, in_filter=3, out_filter=64, "
            "hkernel=7, wkernel=7, hpad=3, wpad=3, hstride=2, wstride=2)": [
                {
                    "schedule": "AVXConvCommonFwd(ic_bn=16, oc_bn=32, reg_n=2, unroll_kw=True)",
                    "time": 0.25
                },
                {
                    "schedule": "AVXConvCommonFwd(ic_bn=32, oc_bn=16, reg_n=2, unroll_kw=True)",
                    "time": 0.5
                }
            ]
        }
    }
    expected_out = {
        Workload(in_dtype='float32', out_dtype='float32',
                 height=224, width=224, in_filter=3, out_filter=64,
                 hkernel=7, wkernel=7, hpad=3, wpad=3, hstride=2, wstride=2): [
            {
                "schedule": AVXConvCommonFwd(ic_bn=16, oc_bn=32, reg_n=2, unroll_kw=True),
                "time": 0.25
            },
            {
                "schedule": AVXConvCommonFwd(ic_bn=32, oc_bn=16, reg_n=2, unroll_kw=True),
                "time": 0.5
            }
        ]
    }
    with open(sch_json_file, "w") as jf:
        json.dump(sch_dict_str, jf, indent=2)
    out = read_sch_from_json(sch_json_file, Workload, [AVXConvCommonFwd])
    os.remove(sch_json_file)
    diff_set = set(out) ^ set(expected_out)
    if len(diff_set) != 0:
        raise RuntimeError("Output mismatch: expecting %s but got %s." % (str(expected_out), str(out)))


def test_write_sch_to_json():
    sch_json_file = "test_write_sch_to_json.json"
    sch_dict = {
        Workload(in_dtype='float32', out_dtype='float32',
                 height=224, width=224, in_filter=3, out_filter=64,
                 hkernel=7, wkernel=7, hpad=3, wpad=3, hstride=2, wstride=2): [
            {
                "schedule": AVXConvCommonFwd(ic_bn=16, oc_bn=32, reg_n=2, unroll_kw=True),
                "time": 0.25
            },
            {
                "schedule": AVXConvCommonFwd(ic_bn=32, oc_bn=16, reg_n=2, unroll_kw=True),
                "time": 0.5
            }
        ]
    }
    expected_out = {
        "schedules": {
            "Workload(in_dtype='float32', out_dtype='float32', "
            "height=224, width=224, in_filter=3, out_filter=64, "
            "hkernel=7, wkernel=7, hpad=3, wpad=3, hstride=2, wstride=2)": [
                {
                    "schedule": "AVXConvCommonFwd(ic_bn=16, oc_bn=32, reg_n=2, unroll_kw=True)",
                    "time": 0.25
                },
                {
                    "schedule": "AVXConvCommonFwd(ic_bn=32, oc_bn=16, reg_n=2, unroll_kw=True)",
                    "time": 0.5
                }
            ]
        }
    }
    write_sch_to_json(sch_dict, sch_json_file)
    with open(sch_json_file, "r") as jf:
        out = json.load(jf)
    os.remove(sch_json_file)
    diff_set = set(out) ^ set(expected_out)
    if len(diff_set) != 0:
        raise RuntimeError("Output mismatch: expecting %s but got %s." % (str(expected_out), str(out)))


def test_get_wkl_map():
    wkl_list = [
        Workload('float32', 'float32', 224, 224, 3, 64, 7, 7, 3, 3, 2, 2),
        Workload('float32', 'float32', 56, 56, 3, 64, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 256, 256, 3, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 256, 256, 64, 32, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 28, 28, 3, 64, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 256, 256, 32, 16, 3, 3, 2, 2, 1, 1),
    ]
    data = sym.Variable("data")
    conv0 = sym.conv2d(data, channels=64, kernel_size=(1, 1))
    conv1 = sym.conv2d(conv0, channels=32, kernel_size=(3, 3), padding=(1, 1))
    out = sym.conv2d(conv1, channels=16, kernel_size=(3, 3), padding=(2, 2))

    g = nnvm.graph.create(out)
    dshape = (1, 3, 256, 256)
    node_map = get_wkl_map(g, {"data": dshape}, wkl_list, "conv2d", get_conv2d_workload)
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


def verify_factor(num, factors):
    out = get_factor(num)
    if out != factors:
        raise RuntimeError("Output mismatch: expecting %s but got %s." % (str(factors), str(out)))


def test_get_factor():
    verify_factor(4, [1, 2, 4])
    verify_factor(5, [1, 5])
    verify_factor(24, [1, 2, 3, 4, 6, 8, 12, 24])
    verify_factor(32, [1, 2, 4, 8, 16, 32])


def verify_shape2layout(shape, layout_template, expected_layout):
    layout = shape2layout(shape, layout_template)
    if layout != expected_layout:
        raise RuntimeError("Output mismatch: expecting %s but got %s." % (expected_layout, layout))


def test_shape2layout():
    verify_shape2layout((1, 3, 4, 4, 8), "NCHWc", "NCHW8c")
    verify_shape2layout((1, 3, 4, 4, 8), "NcHWC", "N3cHWC")


def verify_infer_layout_shape_avx(wkl, in_sch, out_sch, is_elemlike, elemlike_shape,
                                  expected_in_shape, expected_out_shape):
    in_shape, out_shape, _ = infer_layout_shape_avx(wkl, in_sch, out_sch, 1,
                                                    is_elemlike, elemlike_shape)
    if in_shape != expected_in_shape:
        raise RuntimeError("Input shape mismatch: expecting %s but got %s."
                           % (str(expected_in_shape), str(in_shape)))
    if out_shape != expected_out_shape:
        raise RuntimeError("Output shape mismatch: expecting %s but got %s."
                           % (str(expected_out_shape), str(out_shape)))


def test_infer_layout_shape_avx():
    wkl = Workload('float32', 'float32', 224, 224, 32, 64, 7, 7, 3, 3, 2, 2)
    in_sch = AVXConvCommonFwd(16, 32, 2, True)
    out_sch = AVXConvCommonFwd(16, 32, 2, True)
    expected_in_shape = (1, 1, 224, 224, 32)
    expected_out_shape = (1, 2, 224, 224, 16)
    verify_infer_layout_shape_avx(wkl, in_sch, out_sch, False, None,
                                  expected_in_shape, expected_out_shape)
    wkl = Workload('float32', 'float32', 56, 56, 16, 8, 1, 1, 0, 0, 1, 1)
    in_sch = AVXConv1x1Fwd(2, 4, 2, 4)
    out_sch = AVXConv1x1Fwd(8, 1, 2, 4)
    elemlike_shape = (1, 8, 28, 28)
    height = elemlike_shape[2]
    width = elemlike_shape[3]
    expected_in_shape = (1, 2, height, width, 4)
    expected_out_shape = (1, 8, height, width, 1)
    verify_infer_layout_shape_avx(wkl, in_sch, out_sch, True, elemlike_shape,
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


def verify_str2namedtuple(input_str, namedtuple_obj, expect_namedtuple):
    out = str2namedtuple(input_str, namedtuple_obj)
    if expect_namedtuple != out:
        raise RuntimeError("Expecting %s but got %s: " % (str(expect_namedtuple, out)))


def test_str2namedtuple():
    verify_str2namedtuple("AVXConvCommonFwd(1, 2, 3, False)", AVXConvCommonFwd, AVXConvCommonFwd(1, 2, 3, False))
    verify_str2namedtuple("AVXConv1x1Fwd(1, 2, 3, 4)", AVXConv1x1Fwd, AVXConv1x1Fwd(1, 2, 3, 4))


def test_run_remote_module():
    data = sym.Variable("data")
    conv0 = sym.conv2d(data, kernel_size=(3,3), channels=10)
    net = sym.relu(conv0)
    image_shape = (3, 224, 224)
    batch_size = 1
    target = "llvm"
    dshape = (batch_size,) + image_shape
    net, params = testing.utils.create_workload(net, 1, image_shape)
    with nnvm.compiler.build_config(opt_level=3):
        graph, lib, params = nnvm.compiler.build(net, target, shape={"data": dshape}, params=params)
    remote = tvm.rpc.LocalSession()
    run_remote_module(remote, graph, lib, params, {"data": np.random.uniform(size=dshape).astype("float32")},
                      remote_dev_type=target, run_times=10)


if __name__ == "__main__":
    test_get_conv2d_workload()
    test_read_sch_from_json()
    test_write_sch_to_json()
    test_get_wkl_map()
    test_get_real_node()
    test_get_factor()
    test_shape2layout()
    test_infer_layout_shape_avx()
    test_is_elemlike_op()
    test_get_direct_ancestor()
    test_get_in_nodes()
    test_get_out_nodes()
    test_str2namedtuple()
    test_run_remote_module()
