# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=too-many-locals
"""Unit tests for heterogeneous runtime"""
import json
import numpy as np

import tvm
from tvm import te
from tvm.contrib import graph_executor, utils
from tvm import topi


def get_simplex_graph(host_dev_type, device_dev_type):
    r""" Return the hand-crafted json object where only one copy node is
    inserted. This node copies data from the target device to cpu.
    The network is constructed as following:
                 A    B
                  \  /
             elemwise_add  (gpu)
                     \
                     copy      C
                       \      /
                     elemwise_sub  (cpu)

    Parameters
    ----------
    host_dev_type : int
        The device type of the host processor, e.g. cpu.
    device_dev_type : int
        The device type of the device processor, e.g. gpu, opencl, etc.

    Returns
    -------
    json : json
        A json encoded object.
    """
    # Construct each node in the graph.
    var_a = {"op": "null", "name": "A", "inputs": []}
    var_b = {"op": "null", "name": "B", "inputs": []}
    elemwise_add = {
        "op": "tvm_op",
        "name": "elemwise_add",
        "attrs": {
            "flatten_data": "1",
            "func_name": "elemwise_add",
            "num_inputs": "2",
            "num_outputs": "1",
        },
        "inputs": [[0, 0, 0], [1, 0, 0]],
    }
    copy = {
        "op": "tvm_op",
        "name": "__copy_add_to_sub",
        "attrs": {
            "flatten_data": "0",
            "func_name": "__copy",
            "num_inputs": "1",
            "num_outputs": "1",
        },
        "inputs": [[2, 0, 0]],
    }
    var_c = {"op": "null", "name": "C", "inputs": []}
    elemwise_sub = {
        "op": "tvm_op",
        "name": "elemwise_sub",
        "attrs": {
            "flatten_data": "0",
            "func_name": "elemwise_sub",
            "num_inputs": "2",
            "num_outputs": "1",
        },
        "inputs": [[3, 0, 0], [4, 0, 0]],
    }

    # Group the nodes.
    nodes = [var_a, var_b, elemwise_add, copy, var_c, elemwise_sub]
    arg_nodes = [0, 1, 4]
    node_row_ptr = [0, 1, 2, 3, 4, 5, 6]
    heads = [[5, 0, 0]]
    shape = (4,)
    attrs = {
        "storage_id": ["list_int", [3, 4, 0, 1, 5, 2]],
        "shape": ["list_shape", [shape, shape, shape, shape, shape, shape]],
        "device_index": [
            "list_int",
            [
                device_dev_type,
                device_dev_type,
                device_dev_type,
                host_dev_type,
                host_dev_type,
                host_dev_type,
            ],
        ],
        "dtype": ["list_int", [0, 0, 0, 0, 0, 0]],
        "dltype": ["list_str", ["float32", "float32", "float32", "float32", "float32", "float32"]],
    }

    # Construct the graph.
    graph = {
        "nodes": nodes,
        "arg_nodes": arg_nodes,
        "node_row_ptr": node_row_ptr,
        "heads": heads,
        "attrs": attrs,
    }
    return json.dumps(graph)


def test_simplex_data_transferring():
    r"""
    Test the heterogeneous execution of a simple network where data
    transferring is from the target device to the host processor at runtime.
    The host processor is always assumed to be cpu, and the device varies.
    """
    host = "cpu"
    target_host = "llvm"
    host_dev = tvm.device(host)
    if not tvm.runtime.enabled(target_host):
        print("Skip test because llvm is not enabled.")
        return

    def check_device(device, target_device):
        if not tvm.runtime.enabled(target_device):
            print("Skip test because {} is not enabled.".format(target_device))
            return

        device_dev = tvm.device(device)
        graph = get_simplex_graph(host_dev.device_type, device_dev.device_type)
        shape = (4,)

        # Create module for add whose target is the device.
        tensor_a = te.placeholder(shape, name="A")
        tensor_b = te.placeholder(shape, name="B")
        elemwise_add = te.compute(
            shape, lambda *i: tensor_a(*i) + tensor_b(*i), name="elemwise_add"
        )
        target = topi.cpp.TEST_create_target(device)
        schedule_add = topi.cpp.cuda.schedule_injective(target, [elemwise_add])
        lower_add = tvm.lower(schedule_add, [tensor_a, tensor_b, elemwise_add], name="elemwise_add")

        # Insert copy. Neither compute nor schedule is required for the copy
        # node. The compute will be performed at runtime which is just data
        # copy from the input to the output.
        tensor_copy = te.placeholder(shape, name="__copy")

        # Create module for sub whose target is the host.
        tensor_c = te.placeholder(shape, name="C")
        elemwise_sub = te.compute(
            shape, lambda *i: tensor_copy(*i) - tensor_c(*i), name="elemwise_sub"
        )
        schedule_sub = te.create_schedule(elemwise_sub.op)
        lower_sub = tvm.lower(
            schedule_sub, [tensor_copy, tensor_c, elemwise_sub], name="elemwise_sub"
        )

        target_flist = {target_device: lower_add, target_host: lower_sub}
        target = tvm.target.Target(target, target_host)
        mhost = tvm.build(target_flist, target=target)
        dev = [host_dev, device_dev]
        mod = graph_executor.create(graph, mhost, dev)
        params = {}
        params["A"] = tensor_a = np.random.uniform(size=shape).astype(tensor_a.dtype)
        params["B"] = tensor_b = np.random.uniform(size=shape).astype(tensor_b.dtype)
        params["C"] = tensor_c = np.random.uniform(size=shape).astype(tensor_c.dtype)
        mod.set_input(**params)
        mod.run()
        out = mod.get_output(0, tvm.nd.empty(shape))
        np.testing.assert_equal(out.numpy(), (tensor_a + tensor_b) - tensor_c)

    dev_tar = {"cuda": "cuda", "opencl": "opencl"}
    for device, target in dev_tar.items():
        with tvm.target.Target(device):
            check_device(device, target)


def get_duplex_graph(host_dev_type, device_dev_type):
    r""" Return the hand-crafted json object where two copy nodes are inserted.
    Data transferring happens back-and-forth between the target device and CPU.
    The network is constructed as following:
                 A    B
                  \  /
             elemwise_add  (gpu)
                     \
                     copy        C
                       \        /
                      elemwise_sub  (cpu)
                         \
                         copy          D
                           \          /
                           elemwise_add  (gpu)

    Parameters
    ----------
    host_dev_type : int
        The device type of the host processor, e.g. cpu.
    device_dev_type : int
        The device type of the device processor, e.g. gpu, opencl, etc.

    Returns
    -------
    json : json
        A json encoded object.
    """
    # Construct each node in the graph.
    var_a = {"op": "null", "name": "A", "inputs": []}
    var_b = {"op": "null", "name": "B", "inputs": []}
    elemwise_add0 = {
        "op": "tvm_op",
        "name": "elemwise_add0",
        "attrs": {
            "flatten_data": "1",
            "func_name": "elemwise_add0",
            "num_inputs": "2",
            "num_outputs": "1",
        },
        "inputs": [[0, 0, 0], [1, 0, 0]],
    }
    copy_add_sub = {
        "op": "tvm_op",
        "name": "__copy_add_to_sub",
        "attrs": {
            "flatten_data": "0",
            "func_name": "__copy",
            "num_inputs": "1",
            "num_outputs": "1",
        },
        "inputs": [[2, 0, 0]],
    }
    var_c = {"op": "null", "name": "C", "inputs": []}
    elemwise_sub = {
        "op": "tvm_op",
        "name": "elemwise_sub",
        "attrs": {
            "flatten_data": "0",
            "func_name": "elemwise_sub",
            "num_inputs": "2",
            "num_outputs": "1",
        },
        "inputs": [[3, 0, 0], [4, 0, 0]],
    }
    copy_sub_add = {
        "op": "tvm_op",
        "name": "__copy_sub_to_add",
        "attrs": {
            "flatten_data": "0",
            "func_name": "__copy",
            "num_inputs": "1",
            "num_outputs": "1",
        },
        "inputs": [[5, 0, 0]],
    }
    var_d = {"op": "null", "name": "D", "inputs": []}
    elemwise_add1 = {
        "op": "tvm_op",
        "name": "elemwise_add1",
        "attrs": {
            "flatten_data": "0",
            "func_name": "elemwise_add1",
            "num_inputs": "2",
            "num_outputs": "1",
        },
        "inputs": [[6, 0, 0], [7, 0, 0]],
    }

    # Group the nodes.
    nodes = [
        var_a,
        var_b,
        elemwise_add0,
        copy_add_sub,
        var_c,
        elemwise_sub,
        copy_sub_add,
        var_d,
        elemwise_add1,
    ]
    arg_nodes = [0, 1, 4, 7]
    node_row_ptr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    heads = [[8, 0, 0]]
    shape = (4,)
    attrs = {
        "storage_id": ["list_int", [4, 5, 0, 1, 6, 2, 0, 7, 3]],
        "shape": ["list_shape", [shape, shape, shape, shape, shape, shape, shape, shape, shape]],
        "device_index": [
            "list_int",
            [
                device_dev_type,
                device_dev_type,
                device_dev_type,
                host_dev_type,
                host_dev_type,
                host_dev_type,
                device_dev_type,
                device_dev_type,
                device_dev_type,
            ],
        ],
        "dtype": ["list_int", [0, 0, 0, 0, 0, 0, 0, 0, 0]],
        "dltype": [
            "list_str",
            [
                "float32",
                "float32",
                "float32",
                "float32",
                "float32",
                "float32",
                "float32",
                "float32",
                "float32",
            ],
        ],
    }

    # Construct the graph.
    graph = {
        "nodes": nodes,
        "arg_nodes": arg_nodes,
        "node_row_ptr": node_row_ptr,
        "heads": heads,
        "attrs": attrs,
    }
    return json.dumps(graph)


def test_duplex_data_transferring():
    r"""
    Test the heterogeneous execution of a simple network where data
    transferring occurs back-and-forth between the target device and host
    processor.
    The host processor is always assumed to be cpu, and the target device
    varies.
    """
    host = "cpu"
    target_host = "llvm"
    host_dev = tvm.device(host)
    if not tvm.runtime.enabled(target_host):
        print("Skip test because llvm is not enabled.")
        return

    def check_device(device, target_device):
        if not tvm.runtime.enabled(target_device):
            print("Skip test because {} is not enabled.".format(target_device))
            return

        device_dev = tvm.device(device)
        graph = get_duplex_graph(host_dev.device_type, device_dev.device_type)
        shape = (4,)

        # Insert copy nodes for data transferring between add and sub nodes.
        # Transfers data from gpu to cpu.
        copy_add_sub = te.placeholder(shape, name="__copy0")
        # Transfers data from cpu to gpu.
        copy_sub_add = te.placeholder(shape, name="__copy1")

        # Create a module containing adds on the device.
        tensor_a = te.placeholder(shape, name="A")
        tensor_b = te.placeholder(shape, name="B")
        tensor_d = te.placeholder(shape, name="D")
        elemwise_add0 = te.compute(
            shape, lambda *i: tensor_a(*i) + tensor_b(*i), name="elemwise_add0"
        )
        elemwise_add1 = te.compute(
            shape, lambda *i: copy_sub_add(*i) + tensor_d(*i), name="elemwise_add1"
        )
        target = topi.cpp.TEST_create_target(device)
        add_schedule0 = topi.cpp.cuda.schedule_injective(target, [elemwise_add0])
        lower_add0 = tvm.lower(
            add_schedule0, [tensor_a, tensor_b, elemwise_add0], name="elemwise_add0"
        )
        add_schedule1 = topi.cpp.cuda.schedule_injective(target, [elemwise_add1])
        lower_add1 = tvm.lower(
            add_schedule1, [tensor_d, copy_sub_add, elemwise_add1], name="elemwise_add1"
        )
        # Create module for sub whose target is the host.
        tensor_c = te.placeholder(shape, name="C")
        elemwise_sub = te.compute(
            shape, lambda *i: copy_add_sub(*i) - tensor_c(*i), name="elemwise_sub"
        )
        sub_schedule = te.create_schedule(elemwise_sub.op)
        lower_sub = tvm.lower(
            sub_schedule, [copy_add_sub, tensor_c, elemwise_sub], name="elemwise_sub"
        )

        lower_add0.update(lower_add1)
        target_flist = {target_device: lower_add0, target_host: lower_sub}
        target = tvm.target.Target(target, target_host)
        mhost = tvm.build(target_flist, target=target)
        dev = [host_dev, device_dev]
        params = {}
        params["A"] = tensor_a = np.random.uniform(size=shape).astype(tensor_a.dtype)
        params["B"] = tensor_b = np.random.uniform(size=shape).astype(tensor_b.dtype)
        params["C"] = tensor_c = np.random.uniform(size=shape).astype(tensor_c.dtype)
        params["D"] = tensor_d = np.random.uniform(size=shape).astype(tensor_d.dtype)

        def check_verify():
            mod = graph_executor.create(graph, mhost, dev)
            mod.set_input(**params)
            mod.run()
            out = mod.get_output(0, tvm.nd.empty(shape))
            np.testing.assert_equal(out.numpy(), tensor_a + tensor_b - tensor_c + tensor_d)

        def check_load_module():
            temp = utils.tempdir()
            path_lib = temp.relpath("deploy.so")
            mhost.export_library(path_lib)
            with open(temp.relpath("deploy.json"), "w") as out_file:
                out_file.write(graph)
            loaded_lib = tvm.runtime.load_module(path_lib)
            loaded_graph = open(temp.relpath("deploy.json")).read()
            mod = graph_executor.create(loaded_graph, loaded_lib, dev)
            mod.set_input(**params)
            mod.run()
            out = mod.get_output(0, tvm.nd.empty(shape))
            np.testing.assert_equal(out.numpy(), tensor_a + tensor_b - tensor_c + tensor_d)

        check_verify()
        check_load_module()

    dev_tar = {"cuda": "cuda", "opencl": "opencl"}
    for device, target in dev_tar.items():
        with tvm.target.Target(device):
            check_device(device, target)


if __name__ == "__main__":
    test_simplex_data_transferring()
    test_duplex_data_transferring()
