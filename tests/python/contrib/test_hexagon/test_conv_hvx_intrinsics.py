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
import tvm.testing
import tvm.topi.testing

from numpy.random import default_rng
from tvm.tir.function import TensorIntrin

from tests.python.contrib.test_hexagon.conv_uint8 import get_const_tuple, run_conv_te
from tests.python.contrib.test_hexagon.conv_uint8_hvx_intrin import get_conv_uint8_hvx_intrin
from tests.python.contrib.test_hexagon.quantization_utils import quantize_array, quantize_uint8


class TestConvHVX:
    def create_inputs(input_shape, filter_shape, mem_scope):

        w_size, _, _, _ = filter_shape
        input_padding = w_size // 2

        rng = default_rng()
        a = rng.integers(1, 255, input_shape, dtype="uint8")
        w = rng.integers(1, 8, filter_shape, dtype="uint8")

        a_q, a_min, a_max = quantize_array(a.reshape(a.size), a.size)
        w_q, b_min, b_max = quantize_array(w.reshape(w.size), w.size)

        a_q = np.array(a_q, dtype="uint8").reshape(input_shape)
        w_q = np.array(w_q, dtype="uint8").reshape(filter_shape)

        a_offset = quantize_uint8(0.0, a_min, a_max)
        w_offset = quantize_uint8(0.0, b_min, b_max)

        a_f = np.array(a_q, dtype="uint8").reshape(get_const_tuple(a.shape))
        w_f = np.array(w_q, dtype="uint8").reshape(get_const_tuple(w.shape))
        expected_output = tvm.topi.testing.conv2d_nhwc_python(a_f, w_f, 1, input_padding).astype(
            "int32"
        )

        return a_q, w_q, a_offset, w_offset, expected_output, mem_scope

    a, w, a_offset, w_offset, expected_output, mem_scope = tvm.testing.parameters(
        (create_inputs((2, 128, 128, 3), (3, 3, 3, 2), "local")),
        (create_inputs((2, 128, 128, 3), (3, 3, 3, 2), "global")),
        (create_inputs((2, 128, 128, 3), (3, 3, 3, 2), "global.vtcm")),
        (create_inputs((1, 128, 128, 3), (7, 7, 3, 1), "local")),
        (create_inputs((1, 128, 128, 3), (5, 5, 3, 1), "local")),
        (create_inputs((1, 128, 128, 3), (3, 3, 3, 1), "local")),
        (create_inputs((4, 128, 128, 1), (3, 3, 1, 4), "local")),
        (create_inputs((2, 32, 32, 32), (7, 7, 32, 2), "local")),
        (create_inputs((2, 34, 34, 29), (5, 5, 29, 2), "local")),
        (create_inputs((1, 512, 512, 1), (9, 9, 1, 1), "local")),
    )

    @tvm.testing.requires_hexagon
    def test_vrmpy_conv(
        self, hexagon_session, a, w, a_offset, w_offset, expected_output, mem_scope
    ):

        # TODO even sized kernels and stride are currently not working.

        batches, input_size, _, in_c = a.shape
        w_size, _, _, filters = w.shape

        out_height = (input_size - w_size + 2 * (w_size // 2)) + 1
        out_width = (input_size - w_size + 2 * (w_size // 2)) + 1
        out_shape = (batches, out_height, out_width, filters)
        c = np.zeros(out_shape, dtype="int32")

        (
            conv2d_vrmpy_description,
            conv2d_vrmpy_intrinsic,
            conv2d_operator,
        ) = get_conv_uint8_hvx_intrin(a.shape, w.shape, a_offset, w_offset, mem_scope)

        intrin_name = "conv2d.uint8_{}x{}x{}x{}_{}".format(
            input_size, input_size, w_size, w_size, mem_scope
        )
        try:
            TensorIntrin.register(intrin_name, conv2d_vrmpy_description, conv2d_vrmpy_intrinsic)
        except:
            print("Intrinsic already registered.")

        ir_module = conv2d_operator
        sch = tvm.tir.Schedule(ir_module, debug_mask="all")

        block = sch.get_block("C")

        w_block_local = sch.get_block("W_local")
        sch.transform_layout(
            w_block_local, buffer=("write", 0), index_map=lambda h, w, c, f: (f, c, h, w)
        )

        a_block_local = sch.get_block("A_local")
        sch.transform_layout(
            a_block_local, buffer=("write", 0), index_map=lambda b, h, w, c: (b, c, h, w)
        )

        n, f, y, x, ry, rx, rc = sch.get_loops(block)
        sch.reorder(n, f, rc, y, x, ry, rx)

        sch.tensorize(y, intrin_name)

        target_hexagon = tvm.target.hexagon("v68", link_params=True)

        A = tvm.tir.decl_buffer(a.shape, name="A", dtype="uint8")
        W = tvm.tir.decl_buffer(w.shape, name="W", dtype="uint8")
        C = tvm.tir.decl_buffer(out_shape, name="C", dtype="int32")

        func_tir = tvm.build(
            sch.mod,
            [A, W, C],
            tvm.target.Target(target_hexagon, host=target_hexagon),
            name="hvx_op",
        )

        module = hexagon_session.load_module(func_tir)

        a_hexagon = tvm.runtime.ndarray.array(a, device=hexagon_session.device)
        w_hexagon = tvm.runtime.ndarray.array(w, device=hexagon_session.device)
        c_hexagon = tvm.runtime.ndarray.array(c, device=hexagon_session.device)

        module(a_hexagon, w_hexagon, c_hexagon)
        out = c_hexagon.numpy()
        out = out[:, :, :out_width, :]

        tvm.testing.assert_allclose(out, expected_output)

        timer = module.time_evaluator(module.entry_name, hexagon_session.device, number=1, repeat=1)
        time_ms = timer(a_hexagon, w_hexagon, c_hexagon).mean * 1000
        print(
            "Input Shape: {} Kernel Shape: {} Mem_scope: {}. HVX: {} ms.".format(
                a.shape, w.shape, mem_scope, time_ms
            )
        )

    @tvm.testing.requires_hexagon
    def test_te_conv(self, hexagon_session, a, w, a_offset, w_offset, expected_output, mem_scope):
        batches, input_size, _, in_c = a.shape
        w_size, _, _, filters = w.shape
        baseline_output, baseline_time = run_conv_te(
            hexagon_session, a, w, a_offset, w_offset, w_size // 2
        )
        tvm.testing.assert_allclose(baseline_output, expected_output)
        print(
            "Input Shape: {} Kernel Shape: {}. TE Baseline: {} ms".format(
                a.shape, w.shape, baseline_time
            )
        )


if __name__ == "__main__":
    tvm.testing.main()
