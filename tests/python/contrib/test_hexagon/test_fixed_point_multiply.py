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
"""Test Fixed Point Multiply on Hexagon."""

import re
import numpy as np

import tvm.testing
from tvm import relay
from tvm import te
from tvm.relay.backend import Executor
from tvm.contrib.hexagon.session import Session
from tvm.contrib.hexagon.pytest_plugin import HEXAGON_AOT_LLVM_TARGET

from .infrastructure import get_hexagon_target


@tvm.testing.requires_hexagon
def test_vmpy_intrinsic_presence():
    """
    check intrinsic lowering for fixed_point_multiply operation.
    GraphExecutor is used here since get_source("asm") is not supported with aot.
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
    executor = Executor("aot", {"link-params": True})
    lowered = tvm.relay.build(
        relay_mod,
        tvm.target.Target(target, host=target),
        executor=executor,
        params=params,
    )
    return lowered


def run_module(mod, inputs):
    mod.set_input(**inputs)
    mod.run()
    output = mod.get_output(0).numpy()
    return output


class TestFixedPointMultiply:
    """Fixed point Multiply test class"""

    in_scale_const, out_scale_const = tvm.testing.parameters(
        (1.3, 30.0),
        (1.37, 1.0),
        (0.6, 1.0),
        ((1.7, 0.6), 1.0),
        ((0.007, 1.9), 1.0),
    )

    multiplier, shift = tvm.testing.parameters(
        (1288490240, -2),  # 0.15
        (1395864320, 1),  # 1.3
        (1288490188, 0),  # 0.6
    )

    @tvm.testing.requires_hexagon
    def test_per_tensor(self, hexagon_session: Session, multiplier: int, shift: int):
        """Fixed point multiply test."""
        ishape = (6, 32)
        a = relay.var("a", relay.TensorType(ishape, "int32"))
        fpm = relay.fixed_point_multiply(a, multiplier, shift)
        relay_mod = tvm.IRModule.from_expr(fpm)

        with tvm.transform.PassContext(opt_level=3):
            # Compile for Hexagon...
            hexagon_lowered = build_module(relay_mod, HEXAGON_AOT_LLVM_TARGET)

            # Compile for LLVM...
            llvm_lowered = build_module(relay_mod, tvm.target.Target("llvm"))

        data_in = np.arange(-96, 96).reshape(ishape)
        inputs = {"a": data_in}

        # Run hexagon...
        hexagon_mod = hexagon_session.get_executor_from_factory(hexagon_lowered)
        hexagon_output = run_module(hexagon_mod, inputs)

        # Run llvm...
        llvm_mod = tvm.runtime.executor.AotModule(llvm_lowered["default"](tvm.cpu(0)))
        expected_output = run_module(llvm_mod, inputs)

        tvm.testing.assert_allclose(hexagon_output, expected_output)

    @tvm.testing.requires_hexagon
    def test_per_channel(self, hexagon_session: Session, in_scale_const, out_scale_const):
        """Per channel multiply test."""
        ishape = [1, 128, 56, 56]
        axis = 1
        a = relay.var("a", shape=ishape, dtype="int32")

        # Make list of input scales from in_scale_const parameter.
        if isinstance(in_scale_const, tuple):
            in_scale = list(in_scale_const) * (ishape[axis] // len(in_scale_const))
        else:
            in_scale = [in_scale_const] * ishape[axis]
        assert len(in_scale) == ishape[axis]

        # qnn.requantize is lowered to fixed_point_multiply if zp == 0 and in_dtype == out_dtype.
        iscale = relay.const(in_scale)
        izero = relay.const(0)
        oscale = relay.const(out_scale_const)
        ozero = relay.const(0)
        op = relay.qnn.op.requantize(a, iscale, izero, oscale, ozero, axis=axis, out_dtype="int32")
        mod = tvm.IRModule.from_expr(op)

        with tvm.transform.PassContext(opt_level=3):
            # Compile for Hexagon...
            hexagon_lowered = build_module(mod, HEXAGON_AOT_LLVM_TARGET)

            # Compile for LLVM...
            llvm_lowered = build_module(mod, tvm.target.Target("llvm"))

        a_np = np.random.randint(-1000, 1000, size=np.prod(ishape)).reshape(ishape)
        inputs = {"a": a_np}

        # Run hexagon...
        hexagon_mod = hexagon_session.get_executor_from_factory(hexagon_lowered)
        hexagon_output = run_module(hexagon_mod, inputs)

        # Run llvm...
        llvm_mod = tvm.runtime.executor.AotModule(llvm_lowered["default"](tvm.cpu(0)))
        expected_output = run_module(llvm_mod, inputs)

        tvm.testing.assert_allclose(hexagon_output, expected_output)

    vector_size = tvm.testing.parameter(32, 64, 128, 256)

    def test_per_tensor_with_lanes(self, hexagon_session: Session, vector_size):
        """Test fixed point multiply with vectorization.
        Vectorization size is more than hw vector length"""
        ishape = [2, 256, 16]

        def q_mul_shift(shape):
            x = te.placeholder(shape, name="X", dtype="int32")
            out = te.compute(
                shape,
                lambda i, j, k: tvm.tir.q_multiply_shift(
                    x[i, j, k],
                    tvm.tir.const(1395864320, "int32"),
                    tvm.tir.const(31, "int32"),
                    tvm.tir.const(1, "int32"),
                ),
                name="compute",
            )
            return te.create_prim_func([x, out])

        mod = q_mul_shift(ishape)

        # Schedule with vectorization
        sch = tvm.tir.Schedule(mod)
        b00 = sch.get_block(name="compute", func_name="main")
        fused = sch.fuse(*sch.get_loops(block=b00))
        _, v = sch.split(loop=fused, factors=[None, vector_size])
        sch.vectorize(v)

        with tvm.transform.PassContext(opt_level=3):
            hex_lib = tvm.build(sch.mod["main"], target=get_hexagon_target("v68"))
            host_lib = tvm.build(mod, target=tvm.target.Target("llvm"))

        asm = hex_lib.get_source("asm")

        # Check that 'vmpye' instruction was generated in asm file.
        vmpye_regex = re.compile(r"v\d{1,2}.w = vmpye\(v\d{1,2}.w,v\d{1,2}.uh\)")
        assert vmpye_regex.search(asm) is not None

        # Check that 'vmpyo' instruction was generated in asm file.
        vmpyo_regex = re.compile(r"v\d{1,2}.w \+= vmpyo\(v\d{1,2}.w,v\d{1,2}.h\):<<1:rnd:sat:shift")
        assert vmpyo_regex.search(asm) is not None

        # Verify accuracy
        a_np = np.random.randint(-1000, 1000, size=np.prod(ishape)).reshape(ishape).astype("int32")
        b_np = np.random.randint(-1000, 1000, size=np.prod(ishape)).reshape(ishape).astype("int32")
        hex_args = [
            tvm.runtime.ndarray.array(arg, device=hexagon_session.device, mem_scope="global")
            for arg in [a_np, b_np]
        ]
        host_args = [tvm.runtime.ndarray.array(arg) for arg in [a_np, b_np]]

        hex_rt = hexagon_session.load_module(hex_lib)
        hex_rt(*hex_args)
        host_lib(*host_args)

        assert np.allclose(hex_args[1].numpy(), host_args[1].numpy())

    def test_per_channel_with_lanes(self, hexagon_session: Session, vector_size):
        """Test fixed point multiply with vectorization.
        Vectorization size is more than hw vector length"""
        a_shape = [2, 256, 16]
        b_shape = [256]

        def q_mul_shift(shape):
            shift_shape = [shape[1]]
            x = te.placeholder(shape, name="X", dtype="int32")
            y = te.placeholder(shift_shape, name="X", dtype="int32")
            l_shift = te.placeholder(shift_shape, name="X", dtype="int32")
            r_shift = te.placeholder(shift_shape, name="X", dtype="int32")

            out = te.compute(
                shape,
                lambda i, j, k: tvm.tir.q_multiply_shift_per_axis(
                    x[i, j, k],
                    y[j],
                    l_shift[j],
                    r_shift[j],
                    tvm.tir.const(31, "int32"),
                    tvm.tir.const(1, "bool"),
                    tvm.tir.const(0, "bool"),
                ),
                name="compute",
            )
            return te.create_prim_func([x, y, l_shift, r_shift, out])

        mod = q_mul_shift(a_shape)

        # Schedule with vectorization
        sch = tvm.tir.Schedule(mod)
        b00 = sch.get_block(name="compute", func_name="main")
        fused = sch.fuse(*sch.get_loops(block=b00))
        _, v = sch.split(loop=fused, factors=[None, vector_size])
        sch.vectorize(v)

        with tvm.transform.PassContext(opt_level=3):
            hex_lib = tvm.build(sch.mod["main"], target=get_hexagon_target("v68"))
            host_lib = tvm.build(mod, target=tvm.target.Target("llvm"))

        asm = hex_lib.get_source("asm")

        # Check that 'vmpye' instruction was generated in asm file.
        vmpye_regex = re.compile(r"v\d{1,2}.w = vmpye\(v\d{1,2}.w,v\d{1,2}.uh\)")
        assert vmpye_regex.search(asm) is not None

        # Check that 'vmpyo' instruction was generated in asm file.
        vmpyo_regex = re.compile(r"v\d{1,2}.w \+= vmpyo\(v\d{1,2}.w,v\d{1,2}.h\):<<1:rnd:sat:shift")
        assert vmpyo_regex.search(asm) is not None

        # Verify accuracy
        x_np = (
            np.random.randint(-1000, 1000, size=np.prod(a_shape)).reshape(a_shape).astype("int32")
        )
        y_np = (
            np.random.randint(-1000, 1000, size=np.prod(b_shape)).reshape(b_shape).astype("int32")
        )
        lsh_np = np.random.randint(0, 10, size=np.prod(b_shape)).reshape(b_shape).astype("int32")
        rsh_np = np.random.randint(0, 10, size=np.prod(b_shape)).reshape(b_shape).astype("int32")
        b_np = (
            np.random.randint(-1000, 1000, size=np.prod(a_shape)).reshape(a_shape).astype("int32")
        )
        np_args = [x_np, y_np, lsh_np, rsh_np, b_np]
        hex_args = [
            tvm.runtime.ndarray.array(arg, device=hexagon_session.device, mem_scope="global")
            for arg in np_args
        ]
        host_args = [tvm.runtime.ndarray.array(arg) for arg in np_args]

        hex_rt = hexagon_session.load_module(hex_lib)
        hex_rt(*hex_args)
        host_lib(*host_args)

        assert np.allclose(hex_args[4].numpy(), host_args[4].numpy())


if __name__ == "__main__":
    tvm.testing.main()
