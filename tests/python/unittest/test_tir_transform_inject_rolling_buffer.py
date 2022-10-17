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
import tvm
import tvm.script
from tvm.script import tir as T
from tvm import te
from tvm import topi
from tvm.driver.build_module import get_binds
import numpy as np

import pytest


def _tile_nd(s, tensor, tile):
    outer_indices = []
    inner_indices = []
    for i, size in enumerate(tile):
        outer, inner = s[tensor].split(tensor.op.axis[i], size)
        outer_indices.append(outer)
        inner_indices.append(inner)

    s[tensor].reorder(*outer_indices, *inner_indices)
    return outer_indices, inner_indices


@tvm.tir.transform.prim_func_pass(opt_level=0)
def remove_rolling_buffer_attr(func, mod, ctx):
    def unwrap(node):
        if isinstance(node, tvm.tir.AttrStmt) and node.attr_key == "rolling_buffer_scope":
            return node.body
        else:
            return node

    return func.with_body(
        tvm.tir.stmt_functor.ir_transform(
            func.body, None, postorder=unwrap, only_enable=["tir.AttrStmt"]
        )
    )


@tvm.tir.transform.prim_func_pass(opt_level=0)
def verify_no_rolling_buffer_attr(func, mod, ctx):
    def verify(node):
        if isinstance(node, tvm.tir.AttrStmt):
            assert node.attr_key != "rolling_buffer_scope", "Failed to lower rolling buffers"

    tvm.tir.stmt_functor.post_order_visit(func.body, verify)

    return func


def _verify_schedule(sch, inputs, output):
    user_pass_lists = [
        [(0, remove_rolling_buffer_attr), (0, verify_no_rolling_buffer_attr)],
        [(0, tvm.tir.transform.InjectRollingBuffer()), (0, verify_no_rolling_buffer_attr)],
    ]
    built_funcs = []
    for user_pass_list in user_pass_lists:
        with tvm.transform.PassContext(config={"tir.add_lower_pass": user_pass_list}):
            built_funcs.append(tvm.build(sch, inputs + [output]))

    outputs = []
    ctx = tvm.cpu(0)
    input_data = []
    for tensor in inputs:
        shape = [i.value for i in tensor.shape]
        input_data.append(
            tvm.nd.array(np.random.randint(low=-100, high=100, size=shape).astype("int8"), ctx)
        )
    shape = [i.value for i in output.shape]
    out = tvm.nd.array(np.zeros(shape, dtype="int8"), ctx)
    for func in built_funcs:
        func(*input_data, out)
        outputs.append(out.numpy())

    np.testing.assert_equal(outputs[0], outputs[1])


@pytest.mark.parametrize("tile_shape", [(1, 4, 8, 16), (1, 8, 7, 11), (1, 8, 3, 8), (1, 7, 5, 3)])
def test_tile_shapes(tile_shape):
    A = te.placeholder((1, 12, 14, 16), name="A", dtype="int8")
    pool_a = topi.nn.pool2d(A, (3, 3), (1, 1), (1, 1), (0, 0, 0, 0), "max", layout="NHWC")
    pool_b = topi.nn.pool2d(pool_a, (3, 5), (1, 1), (1, 1), (0, 0, 0, 0), "max", layout="NHWC")

    sch = tvm.te.create_schedule([pool_b.op])
    oi, ii = _tile_nd(sch, pool_b, tile_shape)
    sch[pool_a].compute_at(sch[pool_b], oi[-1])
    sch[pool_a].rolling_buffer()

    _verify_schedule(sch, [A], pool_b)


def test_implied_split():
    A = te.placeholder((1, 12, 12, 16), name="A", dtype="int8")
    pool_a = topi.nn.pool2d(A, (3, 3), (1, 1), (1, 1), (0, 0, 0, 0), "max", layout="NHWC")
    pool_b = topi.nn.pool2d(pool_a, (3, 3), (1, 1), (1, 1), (0, 0, 0, 0), "max", layout="NHWC")

    sch = tvm.te.create_schedule([pool_b.op])
    n, h, w, c = pool_b.op.axis
    oi, ii = sch[pool_b].split(w, 4)
    sch[pool_a].compute_at(sch[pool_b], oi)
    sch[pool_a].rolling_buffer()

    _verify_schedule(sch, [A], pool_b)


@pytest.mark.parametrize("kernel_shape", [(1, 1), (3, 3)])
def test_upscale(kernel_shape):
    output_shape = (1, 24, 24, 16)
    input_shape = (
        output_shape[0],
        output_shape[1] // 2 + 2 * (kernel_shape[0] - 1),
        output_shape[2] // 2 + 2 * (kernel_shape[1] - 1),
        output_shape[3],
    )
    A = te.placeholder(input_shape, name="A", dtype="int8")
    pool_a = topi.nn.pool2d(A, kernel_shape, (1, 1), (1, 1), (0, 0, 0, 0), "max", layout="NHWC")
    pool_b = topi.nn.pool2d(
        pool_a, kernel_shape, (1, 1), (1, 1), (0, 0, 0, 0), "max", layout="NHWC"
    )
    upscale = te.compute((1, 24, 24, 16), lambda nn, hh, ww, cc: pool_b[nn, hh // 2, ww // 2, cc])

    sch = tvm.te.create_schedule([upscale.op])
    oi, ii = _tile_nd(sch, upscale, (1, 5, 5, 16))
    sch[pool_b].compute_at(sch[upscale], oi[-1])
    sch[pool_b].rolling_buffer()
    sch[pool_a].compute_at(sch[upscale], oi[-1])
    sch[pool_a].rolling_buffer()

    _verify_schedule(sch, [A], upscale)


@pytest.mark.parametrize("tile_shape", [(1, 4, 8, 16), (1, 8, 7, 11), (1, 8, 3, 8), (1, 7, 5, 3)])
def test_3_tiled_poolings(tile_shape):
    A = te.placeholder((1, 14, 14, 16), name="A", dtype="int8")
    pool_a = topi.nn.pool2d(A, (3, 3), (1, 1), (1, 1), (0, 0, 0, 0), "max", layout="NHWC")
    pool_b = topi.nn.pool2d(pool_a, (3, 3), (1, 1), (1, 1), (0, 0, 0, 0), "max", layout="NHWC")
    pool_c = topi.nn.pool2d(pool_b, (3, 3), (1, 1), (1, 1), (0, 0, 0, 0), "max", layout="NHWC")

    sch = tvm.te.create_schedule([pool_c.op])
    oi, ii = _tile_nd(sch, pool_c, tile_shape)
    sch[pool_b].compute_at(sch[pool_c], oi[-1])
    sch[pool_b].rolling_buffer()
    sch[pool_a].compute_at(sch[pool_c], oi[-1])
    sch[pool_a].rolling_buffer()

    _verify_schedule(sch, [A], pool_c)


@pytest.mark.parametrize("tile_shape", [(1, 4, 8, 16), (1, 8, 7, 11), (1, 8, 3, 8), (1, 7, 5, 3)])
def test_tiled_added_poolings(tile_shape):
    A = te.placeholder((1, 12, 12, 16), name="A", dtype="int8")
    B = te.placeholder((1, 14, 14, 16), name="A", dtype="int8")
    pool_a = topi.nn.pool2d(A, (3, 3), (1, 1), (1, 1), (0, 0, 0, 0), "max", layout="NHWC")
    pool_b = topi.nn.pool2d(B, (5, 5), (1, 1), (1, 1), (0, 0, 0, 0), "max", layout="NHWC")
    add = topi.add(pool_a, pool_b)
    pool_c = topi.nn.pool2d(add, (3, 3), (1, 1), (1, 1), (0, 0, 0, 0), "max", layout="NHWC")

    sch = tvm.te.create_schedule([pool_c.op])
    oi, ii = _tile_nd(sch, pool_c, tile_shape)
    sch[add].compute_at(sch[pool_c], oi[-1])
    sch[add].rolling_buffer()
    sch[pool_b].compute_at(sch[pool_c], oi[-1])
    sch[pool_b].rolling_buffer()
    sch[pool_a].compute_at(sch[pool_c], oi[-1])
    sch[pool_a].rolling_buffer()

    _verify_schedule(sch, [A, B], pool_c)


@pytest.mark.parametrize("make_rolling", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_mixed_buffers(make_rolling):
    A = te.placeholder((1, 14, 14, 16), name="A", dtype="int8")
    pool_a = topi.nn.pool2d(A, (3, 3), (1, 1), (1, 1), (0, 0, 0, 0), "max", layout="NHWC")
    pool_b = topi.nn.pool2d(pool_a, (3, 3), (1, 1), (1, 1), (0, 0, 0, 0), "max", layout="NHWC")
    pool_c = topi.nn.pool2d(pool_b, (3, 3), (1, 1), (1, 1), (0, 0, 0, 0), "max", layout="NHWC")

    sch = tvm.te.create_schedule([pool_c.op])
    oi, ii = _tile_nd(sch, pool_c, (1, 4, 8, 16))
    sch[pool_b].compute_at(sch[pool_c], oi[-1])
    if make_rolling[0]:
        sch[pool_b].rolling_buffer()
    sch[pool_a].compute_at(sch[pool_c], oi[-1])
    if make_rolling[1]:
        sch[pool_a].rolling_buffer()

    _verify_schedule(sch, [A], pool_c)


# fmt: off
@tvm.script.ir_module
class PreRollingBuffer:
    @T.prim_func
    def main(A: T.handle, tensor: T.handle) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        # buffer definition
        tensor_2 = T.buffer_decl([1, 10, 12, 16], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        A_1 = T.match_buffer(A, [1, 12, 14, 16], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        tensor_1 = T.match_buffer(tensor, [1, 8, 8, 16], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        # body
        T.realize(tensor_1[0:1, 0:8, 0:8, 0:16], "")
        for ax1_outer in T.serial(0, 2):
            T.realize(tensor_2[0:1, (ax1_outer*4):((ax1_outer*4) + 6), 0:12, 0:16], "")
            T.attr(tensor_2, "rolling_buffer_scope", True)
            for ax1 in T.serial(0, 6):
                for ax2 in T.serial(0, 12):
                    for ax3 in T.serial(0, 16):
                        tensor_2[0, (ax1 + (ax1_outer*4)), ax2, ax3] = T.int8(0)
                        for dh in T.serial(0, 3):
                            for dw in T.serial(0, 3):
                                tensor_2[0, (ax1 + (ax1_outer*4)), ax2, ax3] = T.max(tensor_2[0, (ax1 + (ax1_outer*4)), ax2, ax3], A_1[0, ((ax1 + (ax1_outer*4)) + dh), (ax2 + dw), ax3])
            for ax1_inner in T.serial(0, 4):
                for ax2_inner in T.serial(0, 8):
                    for ax3_inner in T.serial(0, 16):
                        tensor_1[0, (ax1_inner + (ax1_outer*4)), ax2_inner, ax3_inner] = T.int8(0)
                        for dh_1 in T.serial(0, 3):
                            for dw_1 in T.serial(0, 5):
                                tensor_1[0, (ax1_inner + (ax1_outer*4)), ax2_inner, ax3_inner] = T.max(tensor_1[0, (ax1_inner + (ax1_outer*4)), ax2_inner, ax3_inner], tensor_2[0, ((ax1_inner + (ax1_outer*4)) + dh_1), (ax2_inner + dw_1), ax3_inner])
    __tvm_meta__ = None


@tvm.script.ir_module
class PostRollingBuffer:
    @T.prim_func
    def main(A: T.handle, tensor: T.handle) -> None:
        # function attr dict
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        # buffer definition
        tensor_2 = T.buffer_decl([1, 10, 12, 16], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        A_1 = T.match_buffer(A, [1, 12, 14, 16], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        tensor_1 = T.match_buffer(tensor, [1, 8, 8, 16], dtype="int8", elem_offset=0, align=64, offset_factor=1)
        # body
        T.realize(tensor_1[0:1, 0:8, 0:8, 0:16], "")
        T.realize(tensor_2[0:1, 0:6, 0:12, 0:16], "")
        for ax1_outer in T.serial(0, 2):
            for ax1 in T.serial(0, 6):
                for ax2 in T.serial(0, 12):
                    for ax3 in T.serial(0, 16):
                        if T.likely(((ax1_outer < 1) or (ax1 >= 2)), dtype='bool') :
                            tensor_2[0, T.floormod((ax1 + (ax1_outer*4)), 6), ax2, ax3] = T.int8(0)
                        for dh in T.serial(0, 3):
                            for dw in T.serial(0, 3):
                                if T.likely(((ax1_outer < 1) or (ax1 >= 2)), dtype='bool'):
                                    tensor_2[0, T.floormod((ax1 + (ax1_outer*4)), 6), ax2, ax3] = T.max(tensor_2[0, T.floormod((ax1 + (ax1_outer*4)), 6), ax2, ax3], A_1[0, ((ax1 + (ax1_outer*4)) + dh), (ax2 + dw), ax3])
            for ax1_inner in T.serial(0, 4):
                for ax2_inner in T.serial(0, 8):
                    for ax3_inner in T.serial(0, 16):
                        tensor_1[0, (ax1_inner + (ax1_outer*4)), ax2_inner, ax3_inner] = T.int8(0)
                        for dh_1 in T.serial(0, 3):
                            for dw_1 in T.serial(0, 5):
                                tensor_1[0, (ax1_inner + (ax1_outer*4)), ax2_inner, ax3_inner] = T.max(tensor_1[0, (ax1_inner + (ax1_outer*4)), ax2_inner, ax3_inner], tensor_2[0, T.floormod(((ax1_inner + (ax1_outer*4)) + dh_1), 6), (ax2_inner + dw_1), ax3_inner])
    __tvm_meta__ = None
# fmt: on


def test_rolling_buffer_ir_transform():
    mod = PreRollingBuffer
    mod = tvm.tir.transform.InjectRollingBuffer()(mod)
    script = mod.script(show_meta=True)
    mod = tvm.script.from_source(script)
    tvm.ir.assert_structural_equal(mod["main"], PostRollingBuffer["main"], True)


if __name__ == "__main__":
    pytest.main([__file__])
