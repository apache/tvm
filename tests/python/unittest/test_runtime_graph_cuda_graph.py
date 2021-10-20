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
import json
import os
import re
import sys
import time

import pytest

import tvm
import tvm.testing
from tvm import te
import numpy as np

from tvm.contrib import utils, graph_executor
from tvm.contrib.cuda_graph import cuda_graph_executor


bx = te.thread_axis("blockIdx.x")
tx = te.thread_axis("threadIdx.x")


@tvm.testing.requires_cudagraph
def test_graph_simple():
    n = 32
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    s = te.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], factor=8)
    s[B].bind(xo, bx)
    s[B].bind(xi, tx)

    node0 = {"op": "null", "name": "x", "inputs": []}
    node1 = {
        "op": "tvm_op",
        "name": "add",
        "inputs": [[0, 0, 0]],
        "attrs": {"func_name": "myadd", "flatten_data": "1", "num_inputs": "1", "num_outputs": "1"},
    }
    nodes = [node0, node1]
    arg_nodes = [0]
    node_row_ptr = [0, 1, 2]
    outputs = [[1, 0, 0]]
    shape = (n,)
    attrs = {
        "shape": ["list_shape", [shape, shape]],
        "dltype": ["list_str", ["float32", "float32"]],
        "storage_id": ["list_int", [0, 1]],
    }
    graph = {
        "nodes": nodes,
        "arg_nodes": arg_nodes,
        "node_row_ptr": node_row_ptr,
        "heads": outputs,
        "attrs": attrs,
    }
    graph = json.dumps(graph)

    def check_verify():
        mlib = tvm.build(s, [A, B], "cuda", name="myadd")
        dev = tvm.cuda(0)
        try:
            mod = cuda_graph_executor.create(graph, mlib, dev)
        except ValueError:
            return

        for i in range(3):
            a = np.random.uniform(size=(n,)).astype(A.dtype)
            mod.run(x=a)  # The first run captured a CUDA graph
            out = mod.get_output(0, tvm.nd.empty((n,)))
            np.testing.assert_equal(out.numpy(), a + 1)

        # capture / run CUDA graph manually
        mod.capture_cuda_graph()
        a = np.random.uniform(size=(n,)).astype(A.dtype)
        mod.set_input(x=a)
        mod.run_cuda_graph()
        out = mod.get_output(0, tvm.nd.empty((n,)))
        np.testing.assert_equal(out.numpy(), a + 1)

    check_verify()


if __name__ == "__main__":
    test_graph_simple()
