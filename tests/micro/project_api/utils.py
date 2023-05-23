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

import numpy as np

import tvm
from tvm import relay
from tvm.relay.backend import Runtime
from tvm.micro.testing import get_target


def build_project_api(platform: str):
    """Build a relay module with Project API."""
    shape = (10,)
    dtype = "int8"
    x = relay.var("x", relay.TensorType(shape=shape, dtype=dtype))
    xx = relay.multiply(x, x)
    z = relay.add(xx, relay.const(np.ones(shape=shape, dtype=dtype)))
    func = relay.Function([x], z)
    ir_mod = tvm.IRModule.from_expr(func)

    if platform == "arduino":
        board = "due"
    elif platform == "zephyr":
        board = "qemu_x86"

    runtime = Runtime("crt", {"system-lib": True})
    target = get_target(platform, board)
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = tvm.relay.build(ir_mod, target=target, runtime=runtime)

    project_options = {
        "project_type": "host_driven",
        "board": board,
    }

    temp_dir = tvm.contrib.utils.tempdir()
    project = tvm.micro.generate_project(
        tvm.micro.get_microtvm_template_projects(platform),
        mod,
        temp_dir / "project",
        project_options,
    )
    project.build()
