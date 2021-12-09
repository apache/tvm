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

import logging
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import graph_runtime as runtime
from tvm.relay.debug import PassCounter
from tvm.relay import testing
import subprocess
from os import remove
import re


def create_model():
    data = relay.var("data", relay.TensorType((1, 42, 42, 42), "float32"))
    simple_net = relay.nn.conv2d(
        data=data,
        weight=relay.var("weight"),
        kernel_size=(3, 3),
        channels=64,
        padding=(1, 1),
        data_layout="NCHW",
        kernel_layout="OIHW",
        out_dtype="float32",
    )

    simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)
    return simple_net


def run_pass(net):
    net, params = testing.create_workload(net)
    with tvm.target.create("llvm"):
        count_pass_inst = PassCounter()
        with tvm.transform.PassContext(opt_level=3, instruments=[count_pass_inst]):
            graph, lib, params = relay.build_module.build(net, params=params)


def execute():
    program = "from test_tvm_pass_count import create_model, run_pass; run_pass(create_model());"
    f = open("file.py", "w")
    f.write(program)
    f.close()

    output = subprocess.run(["python", "file.py"], stdout=subprocess.PIPE).stdout.decode("utf-8")

    remove("file.py")
    return output


def is_working(outpout):
    return bool(re.match("1  [a-z]*", output))


if __name__ == "__main__":
    output = execute()
    assert is_working(output), "First pass not found, pass counter is broken!"
