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
import re
import numpy as np

import tvm.testing
from tvm import relay
from tvm.relay.backend import Executor
from tvm.contrib.hexagon.session import Session

from .infrastructure import get_hexagon_target


@tvm.testing.requires_hexagon
def test_vmpy_intrinsic_presence():
    """
    check intrinsic lowering for fixed_point_multiply operation
    """
    ishape = (1, 128)
    a = relay.var("a", relay.TensorType(ishape, "int32"))

    y = relay.fixed_point_multiply(a, 1395864320, 1)  # 1.3

    relay_mod = tvm.IRModule.from_expr(y)

    params = {}
    executor = Executor("graph", {"link-params": True})

    with tvm.transform.PassContext(opt_level=3):
        hexagon_lowered = tvm.relay.build(
            relay_mod,
            get_hexagon_target("v68"),
            executor=executor,
            params=params,
        )

    asm = hexagon_lowered.lib.get_source("asm")

    # Check that 'vmpye' instruction was generated in asm file.
    vmpye_regex = re.compile(r"v\d{1,2}.w = vmpye\(v\d{1,2}.w,v\d{1,2}.uh\)")
    assert vmpye_regex.search(asm) is not None

    # Check that 'vmpyo' instruction was generated in asm file.
    vmpyo_regex = re.compile(r"v\d{1,2}.w \+= vmpyo\(v\d{1,2}.w,v\d{1,2}.h\):<<1:rnd:sat:shift")
    assert vmpyo_regex.search(asm) is not None


def build_module(relay_mod, target):
    params = {}
    executor = Executor("graph", {"link-params": True})
    lowered = tvm.relay.build(
        relay_mod,
        tvm.target.Target(target, host=target),
        executor=executor,
        params=params,
    )
    return lowered


def run_module(graph_mod, inputs):
    graph_mod.set_input(**inputs)
    graph_mod.run()
    output = graph_mod.get_output(0).numpy()
    return output


@tvm.testing.requires_hexagon
def test_fixed_point_multiply_positive_shift(hexagon_session: Session):
    ishape = (6, 32)
    a = relay.var("a", relay.TensorType(ishape, "int32"))
    multiplier, shift = (1395864320, 1)  # 1.3
    fpm = relay.fixed_point_multiply(a, multiplier, shift)
    relay_mod = tvm.IRModule.from_expr(fpm)

    with tvm.transform.PassContext(opt_level=3):
        # Compile for Hexagon...
        hexagon_lowered = build_module(relay_mod, tvm.target.hexagon("v68"))

        # Compile for LLVM...
        llvm_lowered = build_module(relay_mod, tvm.target.Target("llvm"))

    data_in = np.arange(-96, 96).reshape(ishape)
    inputs = {"a": data_in}

    # Run hexagon...
    graph_mod = hexagon_session.get_executor_from_factory(hexagon_lowered)
    hexagon_output = run_module(graph_mod, inputs)

    # Run llvm...
    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    expected_output = run_module(llvm_graph_mod, inputs)

    tvm.testing.assert_allclose(hexagon_output, expected_output)


@tvm.testing.requires_hexagon
def test_fixed_point_multiply_negative_shift(hexagon_session: Session):
    ishape = (6, 32)
    a = relay.var("a", relay.TensorType(ishape, "int32"))
    multiplier, shift = (1288490240, -2)  # 0.15
    fpm = relay.fixed_point_multiply(a, multiplier, shift)
    relay_mod = tvm.IRModule.from_expr(fpm)

    with tvm.transform.PassContext(opt_level=3):
        # Compile for Hexagon...
        hexagon_lowered = build_module(relay_mod, tvm.target.hexagon("v68"))

        # Compile for LLVM...
        llvm_lowered = build_module(relay_mod, tvm.target.Target("llvm"))

    data_in = np.arange(-96, 96).reshape(ishape)
    inputs = {"a": data_in}

    # Run hexagon...
    graph_mod = hexagon_session.get_executor_from_factory(hexagon_lowered)
    hexagon_output = run_module(graph_mod, inputs)

    # Run llvm...
    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    expected_output = run_module(llvm_graph_mod, inputs)

    tvm.testing.assert_allclose(hexagon_output, expected_output, atol=1)


if __name__ == "__main__":
    tvm.testing.main()
