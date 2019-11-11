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
"""Unit tests for external runtime."""
from shutil import which
import json
import numpy as np

import tvm
from tvm import relay
from tvm import module as _tvm_module


def generate_csource_module():
    """Generate a binary"""

    code = r'''
    #include <tvm/runtime/c_runtime_api.h>
    #include <dlpack/dlpack.h>
    #include <cstdint>
    #include <cstring>
    #include <iostream>

    #define GCC_BINARY_OP_1D(p_ID_, p_OP_, p_DIM1_)           \
      extern "C" void p_ID_(float* a, float* b, float* out) { \
        for (int64_t i = 0; i < p_DIM1_; ++i) {               \
          out[i] = a[i] p_OP_ b[i];                           \
        }                                                     \
      }

    #define GCC_BINARY_OP_2D(p_ID_, p_OP_, p_DIM1_, p_DIM2_)  \
      extern "C" void p_ID_(float* a, float* b, float* out) { \
        for (int64_t i = 0; i < p_DIM1_; ++i) {               \
          for (int64_t j = 0; j < p_DIM2_; ++j) {             \
            int64_t k = i * p_DIM2_ + j;                      \
            out[k] = a[k] p_OP_ b[k];                         \
          }                                                   \
        }                                                     \
      }
    GCC_BINARY_OP_2D(gcc_1_0, *, 10, 10);
    GCC_BINARY_OP_2D(gcc_1_1, -, 10, 10);
    GCC_BINARY_OP_2D(gcc_1_2, +, 10, 10);

    extern "C" void gcc_1_(float* gcc_input4, float* gcc_input5,
                           float* gcc_input6, float* gcc_input7, float* out) {
      float* buf_0 = (float*)malloc(4 * 100);
      float* buf_1 = (float*)malloc(4 * 100);
      gcc_1_2(gcc_input4, gcc_input5, buf_0);
      gcc_1_1(buf_0, gcc_input6, buf_1);
      gcc_1_0(buf_1, gcc_input7, out);
    }

    extern "C" int gcc_1(TVMValue* value, int* type_code, int nargs) {
      if (nargs != 5) {
        printf("Expect 5 args, but get %d", nargs);
        return 1;
      }
      DLTensor* arg0 = static_cast<DLTensor*>(value[0].v_handle);
      DLTensor* arg1 = static_cast<DLTensor*>(value[1].v_handle);
      DLTensor* arg2 = static_cast<DLTensor*>(value[2].v_handle);
      DLTensor* arg3 = static_cast<DLTensor*>(value[3].v_handle);
      DLTensor* out = static_cast<DLTensor*>(value[4].v_handle);
      gcc_1_(static_cast<float*>(arg0->data), static_cast<float*>(arg1->data),
             static_cast<float*>(arg2->data), static_cast<float*>(arg3->data),
             static_cast<float*>(out->data));
      return 0;
    }

    GCC_BINARY_OP_2D(gcc_0_0, *, 10, 10);
    GCC_BINARY_OP_2D(gcc_0_1, -, 10, 10);
    GCC_BINARY_OP_2D(gcc_0_2, +, 10, 10);

    extern "C" void gcc_0_(float* gcc_input0, float* gcc_input1,
                           float* gcc_input2, float* gcc_input3, float* out) {
      float* buf_0 = (float*)malloc(4 * 100);
      float* buf_1 = (float*)malloc(4 * 100);
      gcc_0_2(gcc_input0, gcc_input1, buf_0);
      gcc_0_1(buf_0, gcc_input2, buf_1);
      gcc_0_0(buf_1, gcc_input3, out);
    }

    extern "C" int gcc_0(TVMValue* value, int* type_code, int nargs) {
      if (nargs != 5) {
        printf("Expect 5 args, but get %d", nargs);
        return 1;
      }
      DLTensor* arg0 = static_cast<DLTensor*>(value[0].v_handle);
      DLTensor* arg1 = static_cast<DLTensor*>(value[1].v_handle);
      DLTensor* arg2 = static_cast<DLTensor*>(value[2].v_handle);
      DLTensor* arg3 = static_cast<DLTensor*>(value[3].v_handle);
      DLTensor* out = static_cast<DLTensor*>(value[4].v_handle);
      gcc_0_(static_cast<float*>(arg0->data), static_cast<float*>(arg1->data),
             static_cast<float*>(arg2->data), static_cast<float*>(arg3->data),
             static_cast<float*>(out->data));
      return 0;
    }
    '''
    csource_module = _tvm_module.csource_module_create(code, "cc")
    return csource_module


def get_synthetic_lib():
    x = relay.var('x', shape=(10, 10))
    w0 = relay.var('w0', shape=(10, 10))
    w1 = relay.var('w1', shape=(10, 10))
    w2 = relay.var('w2', shape=(10, 10))
    w3 = relay.var('w3', shape=(10, 10))
    w4 = relay.var('w4', shape=(10, 10))
    w5 = relay.var('w5', shape=(10, 10))
    w6 = relay.var('w6', shape=(10, 10))
    w7 = relay.var('w7', shape=(10, 10))

    # subgraph0
    gcc_input0 = relay.var('gcc_input0', shape=(10, 10))
    gcc_input1 = relay.var('gcc_input1', shape=(10, 10))
    gcc_input2 = relay.var('gcc_input2', shape=(10, 10))
    gcc_input3 = relay.var('gcc_input3', shape=(10, 10))
    subgraph0 = relay.Function([gcc_input0, gcc_input1, gcc_input2,
                                gcc_input3], relay.copy(gcc_input0))
    subgraph0 = subgraph0.set_attribute(
        "Primitive", tvm.expr.IntImm("int32", 1))

    # Call subgraph0
    subgraph0_ret = relay.Call(subgraph0, [x, w0, w1, w2])

    # subgraph1
    gcc_input4 = relay.var('gcc_input4', shape=(10, 10))
    gcc_input5 = relay.var('gcc_input5', shape=(10, 10))
    gcc_input6 = relay.var('gcc_input6', shape=(10, 10))
    gcc_input7 = relay.var('gcc_input7', shape=(10, 10))
    subgraph1 = relay.Function([gcc_input4, gcc_input5, gcc_input6,
                                gcc_input7], relay.copy(gcc_input4))
    subgraph1 = subgraph1.set_attribute(
        "Primitive", tvm.expr.IntImm("int32", 1))

    # Call subgraph1
    subgraph1_ret = relay.Call(subgraph1, [x, w3, w4, w5])

    # Other ops that will be executed on TVM.
    add2 = relay.add(x, w6)
    sub2 = relay.subtract(add2, w7)
    ret = relay.concatenate((subgraph0_ret, subgraph1_ret, sub2), 0)
    func = relay.Function([x, w0, w1, w2, w3, w4, w5, w6, w7], ret)
    mod = relay.Module.from_expr(func)
    _, lib, _ = relay.build(mod, "llvm")
    return lib


def get_json():
    nodex = {"op": "null", "name": "x", "inputs": []}
    node0 = {"op": "null", "name": "w0", "inputs": []}
    node1 = {"op": "null", "name": "w1", "inputs": []}
    node2 = {"op": "null", "name": "w2", "inputs": []}
    node3 = {"op": "null", "name": "w3", "inputs": []}
    node4 = {"op": "null", "name": "w4", "inputs": []}
    node5 = {"op": "null", "name": "w5", "inputs": []}
    node6 = {"op": "null", "name": "w6", "inputs": []}
    node7 = {"op": "null", "name": "w7", "inputs": []}

    subgraph0 = {
        "op": "tvm_op",
        "name": "gcc_0",
        "attrs": {
            "num_outputs": "1",
            "num_inputs": "4",
            "func_name": "gcc_0",
            "flatten_data": "0"
        },
        "inputs": [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
        ]
    }
    subgraph1 = {
        "op": "tvm_op",
        "name": "gcc_1",
        "attrs": {
            "num_outputs": "1",
            "num_inputs": "4",
            "func_name": "gcc_1",
            "flatten_data": "0"
        },
        "inputs": [
            [0, 0, 0],
            [4, 0, 0],
            [5, 0, 0],
            [6, 0, 0],
        ]
    }

    fused_op = {
        "op": "tvm_op",
        "name": "fused_add_subtract_concatenate",
        "attrs": {
            "num_outputs": "1",
            "num_inputs": "5",
            "func_name": "fused_add_subtract_concatenate",
            "flatten_data": "0"
        },
        "inputs": [
            [9, 0, 0],
            [10, 0, 0],
            [0, 0, 0],
            [7, 0, 0],
            [8, 0, 0]
        ]
    }
    nodes = [nodex, node0, node1, node2, node3, node4,
             node5, node6, node7, subgraph0, subgraph1, fused_op]
    arg_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    heads = [[11, 0, 0]]
    node_row_ptr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    storage_id = ["list_int", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

    shape = ["list_shape", [
        [10, 10], [10, 10], [10, 10], [10, 10], [10, 10], [10, 10],
        [10, 10], [10, 10], [10, 10], [10, 10], [10, 10], [30, 10]]]

    dltype = ["list_str", [
        "float32", "float32", "float32", "float32", "float32", "float32",
        "float32", "float32", "float32", "float32", "float32", "float32"]]

    attrs = {
        "shape": shape,
        "dltype": dltype,
        "storage_id": storage_id,
    }

    graph = {"nodes": nodes,
             "arg_nodes": arg_nodes,
             "node_row_ptr": node_row_ptr,
             "heads": heads,
             "attrs": attrs}

    return json.dumps(graph)


def test_extern_dso_runtime():
    if which("gcc") is None:
        print("Skip test because gcc is not available.")

    # Get Json and the compiled library.
    json = get_json()
    lib = get_synthetic_lib()
    cur_lib = lib.save("lib.o")

    # library that contains external code.
    csource_module = generate_csource_module()
    # csource_module.save("external.cc", "cc")
    kwargs = {"options": ["lib.o", "-O2", "-std=c++11"]}
    # csource_module.save("external.cc")
    csource_module.export_library("external.so", fcompile=False, **kwargs)
    # load module for execution.
    lib = tvm.module.load("external.so")
    mod = tvm.contrib.graph_runtime.create(json, lib, tvm.cpu(0))

    x_data = np.random.rand(10, 10).astype('float32')
    mod.set_input("x", x_data)
    w_data = []
    for i in range(8):
        data = np.random.rand(10, 10).astype('float32')
        w_data.append(data)
        var = "w" + str(i)
        mod.set_input(var, data)
    mod.run()
    out = tvm.nd.empty((30, 10), ctx=tvm.cpu())
    out = mod.get_output(0, out)
    tvm.testing.assert_allclose(
        out.asnumpy(),
        np.concatenate(
            (((x_data + w_data[0]) - w_data[1]) * w_data[2],
             ((x_data + w_data[3]) - w_data[4]) * w_data[5],
             x_data + w_data[6] - w_data[7]),
            axis=0))


if __name__ == "__main__":
    test_extern_dso_runtime()
