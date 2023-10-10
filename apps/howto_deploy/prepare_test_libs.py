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
"""Script to prepare test_addone.so"""
import tvm
import numpy as np
from tvm import te
from tvm import relay
import os


def prepare_test_libs(base_path):
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    s = te.create_schedule(B.op)
    # Compile library as dynamic library
    fadd_dylib = tvm.build(s, [A, B], "llvm", name="addone")
    dylib_path = os.path.join(base_path, "test_addone_dll.so")
    fadd_dylib.export_library(dylib_path)

    # Compile library in system library mode
    fadd_syslib = tvm.build(
        s,
        [A, B],
        "llvm",
        name="addonesys",
        runtime=relay.backend.Runtime("cpp", {"system-lib": True}),
    )
    syslib_path = os.path.join(base_path, "test_addone_sys.o")
    fadd_syslib.save(syslib_path)


def prepare_graph_lib(base_path):
    x = relay.var("x", shape=(2, 2), dtype="float32")
    y = relay.var("y", shape=(2, 2), dtype="float32")
    params = {"y": np.ones((2, 2), dtype="float32")}
    mod = tvm.IRModule.from_expr(relay.Function([x, y], x + y))
    # build a module
    compiled_lib = relay.build(mod, tvm.target.Target("llvm"), params=params)
    # export it as a shared library
    # If you are running cross compilation, you can also consider export
    # to tar and invoke host compiler later.
    dylib_path = os.path.join(base_path, "test_relay_add.so")
    compiled_lib.export_library(dylib_path)


if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    prepare_test_libs(os.path.join(curr_path, "lib"))
    prepare_graph_lib(os.path.join(curr_path, "lib"))
