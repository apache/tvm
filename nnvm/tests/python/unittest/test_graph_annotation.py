"""Unit tests for graph annotation."""

import time
import zipfile
import os
import numpy as np
import numpy.testing as npt

import nnvm.symbol as symbol
import nnvm.graph as graph
import nnvm.compiler.graph_util as graph_util
import nnvm.compiler
from nnvm.testing import utils

import tvm
from tvm.contrib import graph_runtime, util


def execute_original_graph(sym, target, shape, dtype, params):
    subgraph = graph.create(sym)
    deploy_graph, lib, params = nnvm.compiler.build(
        subgraph, target=target, shape=shape, dtype=dtype, params=params)

    ctx = tvm.cpu()
    module = graph_runtime.create(deploy_graph, lib, ctx)
    module.set_input(**params)
    module.run()
    _, oshape = graph_util.infer_shape(deploy_graph)
    module_out = []
    for i in range(len(sym.list_output_names())):
        out = module.get_output(i, out=tvm.nd.empty(oshape[i], dtype))
        module_out.append(out)
    return module_out


def check_annotated_graph(sym, target, op_name_device, expected_num_nodes,
                          fallback_device, data_shape, params):
    deploy_graph, _, params = nnvm.compiler.build(
        sym,
        target=target,
        shape=data_shape,
        dtype="float32",
        params=params,
        op_name_device=op_name_device,
        fallback_device=fallback_device)

    new_sym = deploy_graph.symbol()
    assert len(new_sym.list_input_names()) == len(sym.list_input_names())
    assert len(new_sym.list_output_names()) == len(sym.list_output_names())
    assert deploy_graph.index.num_nodes == expected_num_nodes


def test_conv_network(device, target):
    R""" The network is as following:
            data1       data2
              |           |
            conv2d      conv2d
               \         /
              elemwise_add
                    |
                  conv2d
    """
    if not tvm.module.enabled(device):
        print("Skip test because %s is not enabled." % device)
        return

    out_channels = 16
    data1 = symbol.Variable(name="data1")
    data2 = symbol.Variable(name="data2")
    simple_net1 = symbol.conv2d(data=data1, kernel_size=(3, 3),
                                channels=out_channels, padding=(1, 1),
                                use_bias=True)

    simple_net2 = symbol.conv2d(data=data2, kernel_size=(3, 3),
                                channels=out_channels, padding=(1, 1),
                                use_bias=True)
    ret = symbol.elemwise_add(simple_net1, simple_net2)
    ret = symbol.conv2d(ret, kernel_size=(3, 3),
                        channels=out_channels, padding=(1, 1),
                        use_bias=True)

    batch_size = 1
    data_shape = (batch_size, 3, 224, 224)
    shape_dict = {"data1": data_shape, "data2": data_shape}
    params = {}
    params["data1"] = np.random.uniform(-1, 1,
                                        size=data_shape).astype("float32")
    params["data2"] = np.random.uniform(-1, 1,
                                        size=data_shape).astype("float32")
    op_name_device = {"elemwise_add": "cpu", "conv2d": device}
    fallback_device = tvm.context("cpu")
    target = {"cpu": "llvm", device: target}
    # No op will be fused. 3 additional device copy nodes are required.
    check_annotated_graph(ret, target, op_name_device, 15, fallback_device,
                          shape_dict, params)


def test_fusible_network(device, target):
    R""" The network is as following:
                data
                  |
                 exp
                /   \
             sqrt   log
                \   /
                b_add
                  |
                tanh
    """
    if not tvm.module.enabled(device):
        print("Skip test because %s is not enabled." % device)
        return

    batch_size = 1
    data_shape = (batch_size, 3, 224, 224)
    data = symbol.Variable('data', shape=data_shape, dtype="float32")
    shape_dict = {"data": data_shape}
    params = {}
    params["data"] = np.random.uniform(-1, 1,
                                       size=data_shape).astype("float32")

    exp = symbol.exp(data, name='exp')
    sqrt = symbol.sqrt(exp, name='sqrt')
    log = symbol.log(exp, name='log')
    ret = sqrt + log
    ret = symbol.tanh(ret)

    fallback_device = tvm.context("cpu")
    target = {"cpu": "llvm", device: target}

    # Fuse log and broadcast_add.
    op_name_device = {
        "exp": "cpu",
        "log": "cpu",
        "broadcast_add": "cpu",
        "sqrt": device,
        "elemwise_add": device,
        "tanh": device
    }
    check_annotated_graph(ret, target, op_name_device, 8, fallback_device,
                          shape_dict, params)

    # Fuse log, broadcast_add, and tanh
    op_name_device = {
        "exp": "cpu",
        "log": device,
        "broadcast_add": device,
        "sqrt": "cpu",
        "elemwise_add": "cpu",
        "tanh": device
    }
    check_annotated_graph(ret, target, op_name_device, 6, fallback_device,
                          shape_dict, params)

    # No operator will be fused.
    op_name_device = {
        "exp": device,
        "log": "cpu",
        "broadcast_add": device,
        "sqrt": "cpu",
        "elemwise_add": device,
        "tanh": "cpu"
    }
    check_annotated_graph(ret, target, op_name_device, 11, fallback_device,
                          shape_dict, params)

    # All operators will be fused.
    op_name_device = {
        "exp": device,
        "log": device,
        "broadcast_add": device,
        "sqrt": device,
        "elemwise_add": device,
        "tanh": device
    }
    check_annotated_graph(ret, target, op_name_device, 2, fallback_device,
                          shape_dict, params)

    # All operators will be fused since all of them are annotated to the
    # same device.
    op_name_device = {
        "exp": "cpu",
        "log": "cpu",
        "broadcast_add": "cpu",
        "sqrt": "cpu",
        "elemwise_add": "cpu",
        "tanh": "cpu"
    }
    check_annotated_graph(ret, target, op_name_device, 2, fallback_device,
                          shape_dict, params)

    # Fuse exp, sqrt, log, and boradcast_add
    op_name_device = {
        "exp": device,
        "log": device,
        "broadcast_add": device,
        "sqrt": device,
        "elemwise_add": device,
        "tanh": "cpu"
    }
    check_annotated_graph(ret, target, op_name_device, 4, fallback_device,
                          shape_dict, params)


def check_graph(sym, target, op_name_device, fallback_device, data_shape,
                params):
    dtype = "float32"

    # execute the whole graph on cpu
    shape1 = {k: v for k, v in data_shape.items()}
    params1 = {k: tvm.nd.array(v) for k, v in params.items()}
    orig_out = execute_original_graph(sym, target="llvm", shape=shape1,
                                      dtype=dtype, params=params1)

    # annotate and compile the graph
    deploy_graph, libmod, params = nnvm.compiler.build(
        sym,
        target=target,
        shape=data_shape,
        dtype=dtype,
        params=params,
        op_name_device=op_name_device,
        fallback_device=fallback_device)
    contexts = [tvm.context(dev) for dev in target.keys()]

    def check_load_module():
        temp = util.tempdir()
        path_lib = temp.relpath("deploy.so")
        libmod.export_library(path_lib)
        with open(temp.relpath("deploy.json"), "w") as fo:
            fo.write(deploy_graph.json())
        with open(temp.relpath("deploy.params"), "wb") as fo:
            fo.write(nnvm.compiler.save_param_dict(params))

        # Load lib, json, and params back.
        loaded_lib = tvm.module.load(path_lib)
        loaded_json = open(temp.relpath("deploy.json")).read()
        loaded_json = graph.load_json(loaded_json)
        loaded_params = bytearray(open(temp.relpath("deploy.params"),
                                       "rb").read())

        module = graph_runtime.create(loaded_json, loaded_lib, contexts)
        loaded_params = nnvm.compiler.load_param_dict(loaded_params)
        module.set_input(**loaded_params)
        module.run()
        _, oshape = graph_util.infer_shape(loaded_json)
        module_out = []
        for i in range(len(sym.list_output_names())):
            out = module.get_output(i, out=tvm.nd.empty(oshape[i], dtype))
            module_out.append(out)
            npt.assert_allclose(out.asnumpy(), orig_out[i].asnumpy(),
                                rtol=1e-5, atol=1e-5)

    def check_inmemory_module():
        module = graph_runtime.create(deploy_graph, libmod, contexts)
        module.set_input(**params)
        module.run()
        _, oshape = graph_util.infer_shape(deploy_graph)
        module_out = []
        for i in range(len(sym.list_output_names())):
            out = module.get_output(i, out=tvm.nd.empty(oshape[i], dtype))
            module_out.append(out)
            npt.assert_allclose(out.asnumpy(), orig_out[i].asnumpy(),
                               rtol=1e-5, atol=1e-5)

    check_load_module()
    check_inmemory_module()


def test_duplex_data_transfer(device, target):
    R""" This unittest tests duplex communication between the host and
    accelerator device. The network is as following:
                data
                  |
                conv2d  (acc)
                  |
             batch_norm (cpu)
                  |
                conv2d  (acc)
    """
    if not tvm.module.enabled(device):
        print("Skip test because %s is not enabled." % device)
        return

    out_channels = 16
    data = symbol.Variable(name="data")
    simple_net = symbol.conv2d(data=data, kernel_size=(3, 3),
                               channels=out_channels, padding=(1, 1),
                               use_bias=False)
    simple_net = symbol.batch_norm(simple_net)
    simple_net = symbol.conv2d(data=simple_net, kernel_size=(3, 3),
                               channels=out_channels, padding=(1, 1),
                               use_bias=False)

    batch_size = 1
    data_shape = (batch_size, 3, 224, 224)
    shape_dict = {"data": data_shape}
    net, params = utils.create_workload(simple_net, batch_size,
                                        data_shape[1:])
    params["data"] = data = np.random.uniform(-1, 1,
                                              size=data_shape).astype(
        "float32")

    target = {"cpu": "llvm", device: target}
    op_name_device = {"conv2d": device, "batch_norm": "cpu",
                      "broadcast_add": "cpu", "elemwise_mul": "cpu"}
    fallback_device = tvm.context("cpu")
    check_graph(net, target, op_name_device, fallback_device, shape_dict,
                params)


if __name__ == "__main__":
    for dev, tar in [("opencl", "opencl"), ("cuda", "cuda"),
                     ("opencl", str(tvm.target.intel_graphics()))]:
        test_conv_network(dev, tar)
        test_fusible_network(dev, tar)
        test_duplex_data_transfer(dev, tar)

