# pylint: disable=too-many-locals
"""Unit tests for heterogeneous runtime"""
import json
import numpy as np

import tvm
from tvm.contrib import graph_runtime, util
import topi

def get_simplex_graph(host_dev_type, device_dev_type):
    r""" Return the hand-crafted json object where only one copy node is
    inserted. Tis node copies data from the target device to cpu.
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
    var_a = {"op": "null", "name": "A", "device_type": device_dev_type,
             "inputs": []}
    var_b = {"op": "null", "name": "B", "device_type": device_dev_type,
             "inputs": []}
    elemwise_add = {
        "op": "tvm_op", "name": "elemwise_add", "device_type": device_dev_type,
        "attrs": {
            "flatten_data": "1",
            "func_name": "elemwise_add",
            "num_inputs": "2",
            "num_outputs": "1"
        },
        "inputs": [[0, 0, 0], [1, 0, 0]]
    }
    copy = {
        "op": "device_copy_op",
        "name": "__copy_add_to_sub",
        "device_type": host_dev_type,
        "attrs": {
            "flatten_data": "0",
            "func_name": "__copy",
            "num_inputs": "1",
            "num_outputs": "1"
        },
        "inputs": [[2, 0, 0]]
    }
    var_c = {"op": "null", "name": "C", "device_type": host_dev_type,
             "inputs": []}
    elemwise_sub = {
        "op": "tvm_op", "name": "elemwise_sub",
        "device_type": host_dev_type,
        "attrs": {
            "flatten_data": "0",
            "func_name": "elemwise_sub",
            "num_inputs": "2",
            "num_outputs": "1"
        },
        "inputs": [[3, 0, 0], [4, 0, 0]]
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
        "device_type": ["list_int", [device_dev_type, device_dev_type,
                                     device_dev_type, host_dev_type,
                                     host_dev_type, host_dev_type]],
        "dtype": ["list_int", [0, 0, 0, 0, 0, 0]],
        "dltype": ["list_str", ["float32", "float32", "float32",
                                "float32", "float32", "float32"]]
    }

    # Construct the graph.
    graph = {"nodes": nodes,
             "arg_nodes": arg_nodes,
             "node_row_ptr": node_row_ptr,
             "heads": heads,
             "attrs": attrs}
    return json.dumps(graph)


def test_simplex_data_transferring():
    r"""
    Test the heterogeneous execution of a simple network where data
    transferring is from the target device to the host processor at runtime.
    The host processor is always assumed to be cpu, and the device varies.
    """
    host = "cpu"
    target_host = "llvm"
    host_ctx = tvm.context(host)
    if not tvm.module.enabled(target_host):
        print("Skip test because llvm is not enabled.")
        return

    def check_device(device, target_device):
        if not tvm.module.enabled(target_device):
            print("Skip test because {} is not enabled.".format(target_device))
            return

        device_ctx = tvm.context(device)
        graph = get_simplex_graph(host_ctx.device_type, device_ctx.device_type)
        shape = (4,)

        # Create module for add whose target is the device.
        tensor_a = tvm.placeholder(shape, name="A")
        tensor_b = tvm.placeholder(shape, name="B")
        elemwise_add = tvm.compute(shape, lambda *i: tensor_a(*i)
                                   + tensor_b(*i), name="elemwise_add")
        target = topi.cpp.TEST_create_target(device)
        schedule_add = topi.cpp.cuda.schedule_injective(target, [elemwise_add])
        lower_add = tvm.lower(schedule_add, [tensor_a, tensor_b, elemwise_add],
                              name="elemwise_add")
        lib_add, host_funcs_add = tvm.build(lower_add, target=target_device,
                                            name="elemwise_add",
                                            postpone_host_codegen=True)

        # Insert copy. Neither compute nor schedule is required for the copy
        # node. The compute will be performed at runtime which is just data
        # copy from the input to the output.
        tensor_copy = tvm.placeholder(shape, name="__copy")

        # Create module for sub whose target is the host.
        tensor_c = tvm.placeholder(shape, name="C")
        elemwise_sub = tvm.compute(shape, lambda *i: tensor_copy(*i)
                                   - tensor_c(*i), name="elemwise_sub")
        schedule_sub = tvm.create_schedule(elemwise_sub.op)
        lower_sub = tvm.lower(schedule_sub, [tensor_copy, tensor_c,
                                             elemwise_sub],
                              name="elemwise_sub")

        lib_sub, host_funcs_sub = tvm.build(lower_sub, target=target_host,
                                            name="elemwise_sub",
                                            postpone_host_codegen=True)
        host_funcs = host_funcs_add + host_funcs_sub
        combined_mod = tvm.combine_modules(host_funcs, [lib_add, lib_sub],
                                           target_host=target_host)

        ctx = [host_ctx, device_ctx]
        mod = graph_runtime.create(graph, combined_mod, ctx)
        params = {}
        params["A"] = tensor_a = np.random.uniform(
            size=shape).astype(tensor_a.dtype)
        params["B"] = tensor_b = np.random.uniform(
            size=shape).astype(tensor_b.dtype)
        params["C"] = tensor_c = np.random.uniform(
            size=shape).astype(tensor_c.dtype)
        mod.set_input(**params)
        mod.run()
        out = mod.get_output(0, tvm.nd.empty(shape))
        np.testing.assert_equal(
            out.asnumpy(), (tensor_a + tensor_b) - tensor_c)

    dev_tar = {"gpu": "cuda", "opencl": "opencl"}
    for device, target in dev_tar.items():
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
    var_a = {"op": "null", "name": "A", "device_type": device_dev_type,
             "inputs": []}
    var_b = {"op": "null", "name": "B", "device_type": device_dev_type,
             "inputs": []}
    elemwise_add0 = {
        "op": "tvm_op", "name": "elemwise_add0", "device_type":
        device_dev_type,
        "attrs": {
            "flatten_data": "1",
            "func_name": "elemwise_add0",
            "num_inputs": "2",
            "num_outputs": "1"
        },
        "inputs": [[0, 0, 0], [1, 0, 0]]
    }
    copy_add_sub = {
        "op": "device_copy_op",
        "name": "__copy_add_to_sub",
        "device_type": host_dev_type,
        "attrs": {
            "flatten_data": "0",
            "func_name": "__copy",
            "num_inputs": "1",
            "num_outputs": "1"
        },
        "inputs": [[2, 0, 0]]
    }
    var_c = {"op": "null", "name": "C", "device_type": host_dev_type,
             "inputs": []}
    elemwise_sub = {
        "op": "tvm_op", "name": "elemwise_sub",
        "device_type": host_dev_type,
        "attrs": {
            "flatten_data": "0",
            "func_name": "elemwise_sub",
            "num_inputs": "2",
            "num_outputs": "1"
        },
        "inputs": [[3, 0, 0], [4, 0, 0]]
    }
    copy_sub_add = {
        "op": "device_copy_op",
        "name": "__copy_sub_to_add",
        "device_type": device_dev_type,
        "attrs": {
            "flatten_data": "0",
            "func_name": "__copy",
            "num_inputs": "1",
            "num_outputs": "1"
        },
        "inputs": [[5, 0, 0]]
    }
    var_d = {"op": "null", "name": "D", "device_type": device_dev_type,
             "inputs": []}
    elemwise_add1 = {
        "op": "tvm_op", "name": "elemwise_add1",
        "device_type": device_dev_type,
        "attrs": {
            "flatten_data": "0",
            "func_name": "elemwise_add1",
            "num_inputs": "2",
            "num_outputs": "1"
        },
        "inputs": [[6, 0, 0], [7, 0, 0]]
    }

    # Group the nodes.
    nodes = [var_a, var_b, elemwise_add0, copy_add_sub, var_c, elemwise_sub,
             copy_sub_add, var_d, elemwise_add1]
    arg_nodes = [0, 1, 4, 7]
    node_row_ptr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    heads = [[8, 0, 0]]
    shape = (4,)
    attrs = {
        "storage_id": ["list_int", [4, 5, 0, 1, 6, 2, 0, 7, 3]],
        "shape": ["list_shape", [shape, shape, shape, shape, shape, shape,
                                 shape, shape, shape]],
        "device_type": ["list_int", [device_dev_type, device_dev_type,
                                device_dev_type,
                                host_dev_type, host_dev_type, host_dev_type,
                                device_dev_type, device_dev_type,
                                device_dev_type]],
        "dtype": ["list_int", [0, 0, 0, 0, 0, 0, 0, 0, 0]],
        "dltype": ["list_str", ["float32", "float32", "float32",
                                "float32", "float32", "float32",
                                "float32", "float32", "float32"]]
    }

    # Construct the graph.
    graph = {"nodes": nodes,
             "arg_nodes": arg_nodes,
             "node_row_ptr": node_row_ptr,
             "heads": heads,
             "attrs": attrs}
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
    host_ctx = tvm.context(host)
    if not tvm.module.enabled(target_host):
        print("Skip test because llvm is not enabled.")
        return

    def check_device(device, target_device):
        if not tvm.module.enabled(target_device):
            print("Skip test because {} is not enabled.".format(target_device))
            return

        device_ctx = tvm.context(device)
        graph = get_duplex_graph(host_ctx.device_type, device_ctx.device_type)
        shape = (4,)

        # Insert copy nodes for data transferring between add and sub nodes.
        # Transfers data from gpu to cpu.
        copy_add_sub = tvm.placeholder(shape, name="__copy0")
        # Transfers data from cpu to gpu.
        copy_sub_add = tvm.placeholder(shape, name="__copy1")

        # Create a module containing adds on the device.
        tensor_a = tvm.placeholder(shape, name="A")
        tensor_b = tvm.placeholder(shape, name="B")
        tensor_d = tvm.placeholder(shape, name="D")
        elemwise_add0 = tvm.compute(shape, lambda *i: tensor_a(*i)
                                    + tensor_b(*i), name="elemwise_add0")
        elemwise_add1 = tvm.compute(shape, lambda *i: copy_sub_add(*i)
                                    + tensor_d(*i), name="elemwise_add1")
        target = topi.cpp.TEST_create_target(device)
        add_schedule0 = topi.cpp.cuda.schedule_injective(
            target, [elemwise_add0])
        lower_add0 = tvm.lower(
            add_schedule0, [tensor_a, tensor_b, elemwise_add0],
            name="elemwise_add0")
        add_schedule1 = topi.cpp.cuda.schedule_injective(
            target, [elemwise_add1])
        lower_add1 = tvm.lower(
            add_schedule1, [tensor_d, copy_sub_add, elemwise_add1],
            name="elemwise_add1")
        lib_add, host_funcs_add = tvm.build([lower_add0, lower_add1],
                                            target=target_device,
                                            postpone_host_codegen=True)

        # Create module for sub whose target is the host.
        tensor_c = tvm.placeholder(shape, name="C")
        elemwise_sub = tvm.compute(shape, lambda *i: copy_add_sub(*i)
                                   - tensor_c(*i), name="elemwise_sub")
        sub_schedule = tvm.create_schedule(elemwise_sub.op)
        lower_sub = tvm.lower(sub_schedule, [copy_add_sub, tensor_c,
                                             elemwise_sub],
                              name="elemwise_sub")
        lib_sub, host_funcs_sub = tvm.build(lower_sub, target=target_host,
                                            postpone_host_codegen=True)
        host_funcs = host_funcs_add + host_funcs_sub

        combined_mod = tvm.combine_modules(host_funcs, [lib_add, lib_sub],
                                           target_host=target_host)
        ctx = [host_ctx, device_ctx]
        params = {}
        params["A"] = tensor_a = np.random.uniform(
            size=shape).astype(tensor_a.dtype)
        params["B"] = tensor_b = np.random.uniform(
            size=shape).astype(tensor_b.dtype)
        params["C"] = tensor_c = np.random.uniform(
            size=shape).astype(tensor_c.dtype)
        params["D"] = tensor_d = np.random.uniform(
            size=shape).astype(tensor_d.dtype)

        def check_verify():
            mod = graph_runtime.create(graph, combined_mod, ctx)
            mod.set_input(**params)
            mod.run()
            out = mod.get_output(0, tvm.nd.empty(shape))
            np.testing.assert_equal(
                out.asnumpy(), tensor_a + tensor_b - tensor_c + tensor_d)

        def check_load_module():
            temp = util.tempdir()
            path_lib = temp.relpath("deploy.so")
            combined_mod.export_library(path_lib)
            with open(temp.relpath("deploy.json"), "w") as out_file:
                out_file.write(graph)
            loaded_lib = tvm.module.load(path_lib)
            loaded_graph = open(temp.relpath("deploy.json")).read()
            mod = graph_runtime.create(loaded_graph, loaded_lib, ctx)
            mod.set_input(**params)
            mod.run()
            out = mod.get_output(0, tvm.nd.empty(shape))
            np.testing.assert_equal(
                out.asnumpy(), tensor_a + tensor_b - tensor_c + tensor_d)

        check_verify()
        check_load_module()

    dev_tar = {"gpu": "cuda", "opencl": "opencl"}
    for device, target in dev_tar.items():
        check_device(device, target)

if __name__ == "__main__":
    test_simplex_data_transferring()
    test_duplex_data_transferring()
