#!/usr/bin/env python3
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

"""Builds a simple graph for testing."""
import argparse
import os
import subprocess
import sys

import onnx
import tvm
from tvm import relay


def _get_mod_and_params(model_file):
    onnx_model = onnx.load(model_file)
    shape_dict = {}
    for input in onnx_model.graph.input:
        shape_dict[input.name] = [dim.dim_value for dim in input.type.tensor_type.shape.dim]

    return relay.frontend.from_onnx(onnx_model, shape_dict)


def build_graph_lib(model_file, opt_level):
    """Compiles the pre-trained model with TVM"""
    out_dir = os.path.join(sys.path[0], "../lib")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Compile the relay mod
    mod, params = _get_mod_and_params(model_file)
    target = "llvm -target=wasm32-unknown-unknown -mattr=+simd128 --system-lib"
    with tvm.transform.PassContext(opt_level=opt_level):
        graph_json, lib, params = relay.build(mod, target=target, params=params)

    # Save the model artifacts to obj_file
    obj_file = os.path.join(out_dir, "graph.o")
    lib.save(obj_file)
    # Run llvm-ar to archive obj_file into lib_file
    lib_file = os.path.join(out_dir, "libgraph_wasm32.a")
    cmds = [os.environ.get("LLVM_AR", "llvm-ar-10"), "rcs", lib_file, obj_file]
    subprocess.run(cmds)

    with open(os.path.join(out_dir, "graph.json"), "w") as f_graph:
        f_graph.write(graph_json)

    with open(os.path.join(out_dir, "graph.params"), "wb") as f_params:
        f_params.write(relay.save_param_dict(params))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX model build example")
    parser.add_argument("model_file", type=str, help="the path of onnx model file")
    parser.add_argument(
        "-O",
        "--opt-level",
        type=int,
        default=0,
        help="level of optimization. 0 is unoptimized and 3 is the highest level",
    )
    args = parser.parse_args()

    build_graph_lib(args.model_file, args.opt_level)
