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
from shutil import which
import json
import pytest
import sys
import numpy as np

import tvm
from tvm import te
import tvm.runtime._ffi_api
from tvm import relay
from tvm.contrib import util

tmp_path = util.tempdir()


def generate_csource_module():
    """Mock the codegen with an external library (e.g., CBLAS/cuDNN)"""

    code = r'''
    #include <tvm/runtime/c_runtime_api.h>
    #include <tvm/runtime/packed_func.h>
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
      free(buf_0);
      free(buf_1);
    }

    extern "C" int ccompiler_wrapper_1_(DLTensor* arg0, DLTensor* arg1,
                                        DLTensor* arg2, DLTensor* arg3,
                                        DLTensor* out) {
        gcc_1_(static_cast<float*>(arg0->data), static_cast<float*>(arg1->data),
               static_cast<float*>(arg2->data), static_cast<float*>(arg3->data),
               static_cast<float*>(out->data));
        return 0;
    }

    TVM_DLL_EXPORT_TYPED_FUNC(json_rt_1, ccompiler_wrapper_1_);

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
      free(buf_0);
      free(buf_1);
    }

    extern "C" int ccompiler_wrapper_0_(DLTensor* arg0, DLTensor* arg1,
                                        DLTensor* arg2, DLTensor* arg3,
                                        DLTensor* out) {
        gcc_0_(static_cast<float*>(arg0->data), static_cast<float*>(arg1->data),
               static_cast<float*>(arg2->data), static_cast<float*>(arg3->data),
               static_cast<float*>(out->data));
        return 0;
    }

    TVM_DLL_EXPORT_TYPED_FUNC(json_rt_0, ccompiler_wrapper_0_);

    '''
    csource_module = tvm.runtime._ffi_api.CSourceModuleCreate(code, "cc")
    return csource_module


def generate_engine_module():
    """
    Mock the codegen of an external backend with its own runtime engine
    (e.g., MKL-DNN/TensorRT)
    """

    code = r'''
    #include <tvm/runtime/c_runtime_api.h>
    #include <tvm/runtime/packed_func.h>
    #include <dlpack/dlpack.h>
    #include "json_engine.h"

    extern "C" void json_1_(float* json_input4, float* json_input5,
                            float* json_input6, float* json_input7, float* out) {

        std::string graph =
            "add_2d,10,10\n"
            "sub_2d,10,10\n"
            "mul_2d,10,10\n";

        Engine engine;
        engine.run(graph, {json_input4, json_input5, json_input6, json_input7}, out);
    }

    extern "C" int json_wrapper_1_(DLTensor* arg0, DLTensor* arg1,
                                   DLTensor* arg2, DLTensor* arg3,
                                   DLTensor* out) {
        json_1_(static_cast<float*>(arg0->data), static_cast<float*>(arg1->data),
                static_cast<float*>(arg2->data), static_cast<float*>(arg3->data),
                static_cast<float*>(out->data));
        return 0;
    }

    TVM_DLL_EXPORT_TYPED_FUNC(json_rt_1, json_wrapper_1_);

    extern "C" void json_0_(float* json_input0, float* json_input1,
                            float* json_input2, float* json_input3, float* out) {

        std::string graph =
            "add_2d,10,10\n"
            "sub_2d,10,10\n"
            "mul_2d,10,10\n";

        Engine engine;
        engine.run(graph, {json_input0, json_input1, json_input2, json_input3}, out);

    }

    extern "C" int json_wrapper_0_(DLTensor* arg0, DLTensor* arg1,
                                   DLTensor* arg2, DLTensor* arg3,
                                   DLTensor* out) {
        json_0_(static_cast<float*>(arg0->data), static_cast<float*>(arg1->data),
                static_cast<float*>(arg2->data), static_cast<float*>(arg3->data),
                static_cast<float*>(out->data));
        return 0;
    }

    TVM_DLL_EXPORT_TYPED_FUNC(json_rt_0, json_wrapper_0_);

    '''

    gen_json_engine()
    csource_module = tvm.runtime._ffi_api.CSourceModuleCreate(code, "cc")
    return csource_module


def gen_json_engine():
    """An example of external backend runtime engine. This is supposed to be provided
      by third-party vendors and included when building the generated external kernel code.
    """

    code = r'''
    #ifndef _JSON_ENGINE_H_
    #define _JSON_ENGINE_H_
    #include <cstdint>
    #include <string>
    #include <sstream>
    #include <vector>

    #define GCC_BINARY_OP_2D(p_ID_, p_OP_)  \
      void p_ID_(int64_t dim1, int64_t dim2, float* a, float* b, float* out) { \
        for (int64_t i = 0; i < dim1; ++i) {                                   \
          for (int64_t j = 0; j < dim2; ++j) {                                 \
            int64_t k = i * dim2 + j;                                          \
            out[k] = a[k] p_OP_ b[k];                                          \
          }                                                                    \
        }                                                                      \
      }
    GCC_BINARY_OP_2D(add_2d, +);
    GCC_BINARY_OP_2D(sub_2d, -);
    GCC_BINARY_OP_2D(mul_2d, *);

    struct Layer {
        void (*op)(int64_t, int64_t, float*, float*, float*);
        std::vector<int64_t> shapes;
        std::vector<float*> args;
    };

    class Engine {
    public:
        float* alloc_buffer(int64_t size) {
            float* buf = (float*)malloc(sizeof(float) * size);
            buffers.push_back(buf);
            return buf;
        }
        void add(std::string op, int64_t dim1, int64_t dim2, float* in1, float* in2, float* out) {
            Layer layer;
            layer.shapes.push_back(dim1);
            layer.shapes.push_back(dim2);
            layer.args.push_back(in1);
            layer.args.push_back(in2);
            layer.args.push_back(out);

            if (op == "add_2d")
                layer.op = &add_2d;
            else if (op == "sub_2d")
                layer.op = &sub_2d;
            else if (op == "mul_2d")
                layer.op = &mul_2d;
            net.push_back(layer);
            return ;
        }

        void run(std::string graph, std::vector<float*> args, float* out) {
            std::stringstream ss(graph);
            std::string line;
            int layer_idx = 0;
            int arg_idx = 0;
            float* buf = nullptr;

            while (std::getline(ss, line, '\n')) {
                std::stringstream ss2(line);
                std::string token;
                std::vector<std::string> attrs;
                while (std::getline(ss2, token, ',')) {
                    attrs.push_back(token);
                }
                int64_t dim1 = stoll(attrs[1]);
                int64_t dim2 = stoll(attrs[2]);
                auto out_buf = this->alloc_buffer(dim1 * dim2);

                if (layer_idx == 0) {
                    this->add(attrs[0], dim1, dim2, args[0], args[1], out_buf);
                    buf = out_buf;
                    arg_idx = 2;
                }
                else {
                    this->add(attrs[0], dim1, dim2, buf, args[arg_idx], out_buf);
                    buf = out_buf;
                    arg_idx++;
                }
                layer_idx++;
            }
            this->net.back().args.back() = out;

            for (auto layer : net) {
                (*layer.op)(layer.shapes[0], layer.shapes[1], layer.args[0], layer.args[1], layer.args[2]);
            }
        }
        ~Engine() {
            for (auto buf : buffers) {
                free(buf);
            }
        }
    private:
        std::vector<Layer> net;
        std::vector<float*> buffers;
    };

    #endif  // _JSON_ENGINE_H_
    '''
    header_file = tmp_path.relpath("json_engine.h")
    with open(header_file, 'w') as f:
        f.write(code)


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
    subgraph0 = subgraph0.with_attr(
        "Primitive", tvm.tir.IntImm("int32", 1))

    # Call subgraph0
    subgraph0_ret = relay.Call(subgraph0, [x, w0, w1, w2])

    # subgraph1
    gcc_input4 = relay.var('gcc_input4', shape=(10, 10))
    gcc_input5 = relay.var('gcc_input5', shape=(10, 10))
    gcc_input6 = relay.var('gcc_input6', shape=(10, 10))
    gcc_input7 = relay.var('gcc_input7', shape=(10, 10))
    subgraph1 = relay.Function([gcc_input4, gcc_input5, gcc_input6,
                                gcc_input7], relay.copy(gcc_input4))
    subgraph1 = subgraph1.with_attr(
        "Primitive", tvm.tir.IntImm("int32", 1))

    # Call subgraph1
    subgraph1_ret = relay.Call(subgraph1, [x, w3, w4, w5])

    # Other ops that will be executed on TVM.
    add2 = relay.add(x, w6)
    sub2 = relay.subtract(add2, w7)
    ret = relay.concatenate((subgraph0_ret, subgraph1_ret, sub2), 0)
    func = relay.Function([x, w0, w1, w2, w3, w4, w5, w6, w7], ret)
    mod = tvm.IRModule.from_expr(func)
    _, lib, _ = relay.build(mod, "llvm")
    return lib

def get_whole_graph_json():
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
        "name": "json_rt_0",
        "attrs": {
            "num_outputs": "1",
            "num_inputs": "4",
            "func_name": "json_rt_0",
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
        "name": "json_rt_1",
        "attrs": {
            "num_outputs": "1",
            "num_inputs": "4",
            "func_name": "json_rt_1",
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


def run_extern(label, get_extern_src, **kwargs):
    if which("gcc") is None:
        print("Skip test because gcc is not available.")
        return

    obj_name = "{}.o".format(label)
    lib_name = "external_{}.so".format(label)

    # Get Json and the compiled library.
    graph_json = get_whole_graph_json()
    lib = get_synthetic_lib()
    lib.save(obj_name)

    # library that contains external code.
    csource_module = get_extern_src()
    kwargs["options"] = [obj_name] + kwargs["options"]
    lib_path = tmp_path.relpath(lib_name)
    csource_module.export_library(lib_path, fcompile=False, **kwargs)
    # load module for execution.
    lib = tvm.runtime.load_module(lib_path)
    mod = tvm.contrib.graph_runtime.create(graph_json, lib, tvm.cpu(0))

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
        np.concatenate((((x_data + w_data[0]) - w_data[1]) * w_data[2],
                        ((x_data + w_data[3]) - w_data[4]) * w_data[5],
                        x_data + w_data[6] - w_data[7]),
                       axis=0))


def test_dso_extern():
    run_extern("lib", generate_csource_module, options=["-O2", "-std=c++14"])


def test_engine_extern():
    run_extern("engine",
               generate_engine_module,
               options=["-O2", "-std=c++14", "-I" + tmp_path.relpath("")])

def test_json_extern():
    if not tvm.get_global_func("module.loadfile_examplejson", True):
        print("Skip because JSON example runtime is not enabled.")
        return

    # Get subgraph Json.
    subgraph_json = ("json_rt_0\n" +
                     "input 0 10 10\n" +
                     "input 1 10 10\n" +
                     "input 2 10 10\n" +
                     "input 3 10 10\n" +
                     "add 4 inputs: 0 1 shape: 10 10\n" +
                     "sub 5 inputs: 4 2 shape: 10 10\n" +
                     "mul 6 inputs: 5 3 shape: 10 10\n" +
                     "json_rt_1\n" +
                     "input 0 10 10\n" +
                     "input 1 10 10\n" +
                     "input 2 10 10\n" +
                     "input 3 10 10\n" +
                     "add 4 inputs: 0 1 shape: 10 10\n" +
                     "sub 5 inputs: 4 2 shape: 10 10\n" +
                     "mul 6 inputs: 5 3 shape: 10 10")

    subgraph_path = tmp_path.relpath('subgraph.examplejson')
    with open(subgraph_path, 'w') as f:
        f.write(subgraph_json)

    # Get Json and module.
    graph_json = get_whole_graph_json()


    lib = get_synthetic_lib()
    ext_lib = tvm.runtime.load_module(subgraph_path, "examplejson")
    lib.import_module(ext_lib)
    lib_name = 'external.so'
    lib_path = tmp_path.relpath(lib_name)
    lib.export_library(lib_path)

    # load module for execution.
    lib = tvm.runtime.load_module(lib_path)
    mod = tvm.contrib.graph_runtime.create(graph_json, lib, tvm.cpu(0))

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
        np.concatenate((((x_data + w_data[0]) - w_data[1]) * w_data[2],
                        ((x_data + w_data[3]) - w_data[4]) * w_data[5],
                        x_data + w_data[6] - w_data[7]),
                       axis=0))


if __name__ == "__main__":
    test_dso_extern()
    test_engine_extern()
    test_json_extern()
