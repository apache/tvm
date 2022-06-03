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

import pytest
import numpy as np
from tvm.contrib.hexagon.session import Session

import tvm.testing
from tvm import te, tir
from tvm.script import tir as T
from tvm.contrib.hexagon.session import Session


def intrin_mem_copy(shape, dtype, dst_scope, src_scope):
    src = te.placeholder(shape=shape, dtype=dtype, name="src")
    dst = te.compute(shape, lambda i: src[i], name="dst")
    size = shape[0] * np.dtype(dtype).itemsize

    src_buffer = tvm.tir.decl_buffer(
        shape,
        dtype,
        scope=src_scope,
        offset_factor=1,
        name="mem_copy_src_buffer",
    )

    dst_buffer = tvm.tir.decl_buffer(
        shape,
        dtype,
        scope=dst_scope,
        offset_factor=1,
        name="mem_copy_dst_buffer",
    )

    zero_indices = [0 for _ in shape]

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        _src = ins[0]
        _dst = outs[0]

        dst_handle = ib.buffer_ptr(dst_buffer)
        src_handle = ib.buffer_ptr(src_buffer)

        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.mem_copy",
                tvm.tir.call_intrin("handle", "tir.address_of", dst_handle[zero_indices]),
                tvm.tir.call_intrin("handle", "tir.address_of", src_handle[zero_indices]),
                size,
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(dst.op, intrin_func, binds={src: src_buffer, dst: dst_buffer})


def verify(hexagon_session: Session, s, x, y, z, size):
    print(tvm.lower(s, [x, y, z]))

    target_hexagon = tvm.target.hexagon("v68", link_params=True)
    func = tvm.build(
        s, [x, y, z], tvm.target.Target(target_hexagon, host=target_hexagon), name="dmacpy"
    )

    mod = hexagon_session.load_module(func)
    xt = tvm.nd.array(
        np.random.randint(low=-128, high=127, size=size, dtype=x.dtype),
        device=hexagon_session.device,
    )
    yt = tvm.nd.array(
        np.random.randint(low=-128, high=127, size=size, dtype=y.dtype),
        device=hexagon_session.device,
    )
    zt = tvm.nd.array(
        np.random.randint(low=-128, high=127, size=size, dtype=z.dtype),
        device=hexagon_session.device,
    )
    mod["dmacpy"](xt, yt, zt)

    ref = xt.numpy() + yt.numpy()
    np.testing.assert_equal(zt.numpy(), ref)


@tvm.testing.requires_hexagon
def test_cache_read_write(hexagon_session: Session):
    size = 128
    outer_shape = (size,)
    factor = 16
    inner_shape = (factor,)
    dtype = "int8"

    x = te.placeholder(shape=outer_shape, dtype=dtype, name="x")
    y = te.placeholder(shape=outer_shape, dtype=dtype, name="y")
    z = te.compute(outer_shape, lambda i: x[i] + y[i], name="z")
    s = te.create_schedule(z.op)

    x_vtcm = s.cache_read(x, "global.vtcm", [z])
    y_vtcm = s.cache_read(y, "global.vtcm", [z])
    z_vtcm = s.cache_write(z, "global.vtcm")

    zouter, zinner = s[z_vtcm].split(z_vtcm.op.axis[0], factor=factor)

    s[x_vtcm].compute_at(s[z_vtcm], zouter)
    s[y_vtcm].compute_at(s[z_vtcm], zouter)

    mem_copy_read = intrin_mem_copy(inner_shape, dtype, "global.vtcm", "global")

    (cache_read_x,) = s[x_vtcm].op.axis
    s[x_vtcm].tensorize(cache_read_x, mem_copy_read)

    (cache_read_y,) = s[y_vtcm].op.axis
    s[y_vtcm].tensorize(cache_read_y, mem_copy_read)

    mem_copy_write = intrin_mem_copy(outer_shape, dtype, "global", "global.vtcm")

    (cache_write_z,) = s[z].op.axis
    s[z].tensorize(cache_write_z, mem_copy_write)

    verify(hexagon_session, s, x, y, z, size)


def layout_transform_2d(n):
    return [n // 16, te.AXIS_SEPARATOR, n % 16]


@tvm.testing.requires_hexagon
def test_cache_read_write_2d(hexagon_session: Session):
    size = 128
    outer_shape = (size,)
    factor = 16
    inner_shape = (factor,)
    dtype = "int8"

    x = te.placeholder(shape=outer_shape, dtype=dtype, name="x")
    y = te.placeholder(shape=outer_shape, dtype=dtype, name="y")
    z = te.compute(outer_shape, lambda i: x[i] + y[i], name="z")
    s = te.create_schedule(z.op)

    x_vtcm = s.cache_read(x, "global.vtcm", [z])
    y_vtcm = s.cache_read(y, "global.vtcm", [z])
    z_vtcm = s.cache_write(z, "global.vtcm")

    layout_x_vtcm = s[x_vtcm].transform_layout(layout_transform_2d)
    layout_y_vtcm = s[y_vtcm].transform_layout(layout_transform_2d)
    layout_z_vtcm = s[z_vtcm].transform_layout(layout_transform_2d)

    mem_copy_read = intrin_mem_copy(inner_shape, dtype, "global.vtcm", "global")
    s[x_vtcm].tensorize(layout_x_vtcm[1], mem_copy_read)
    s[y_vtcm].tensorize(layout_y_vtcm[1], mem_copy_read)

    # The loop schedule over `z` is not modified when calling `transform_layout`
    # on `z_vtcm` above therefore we must call `split` to modify the loop schedule
    # over `z` to match the layout of `z_vtcm` such that we can accurately write
    # `z_vtcm` back to `z` using memory copy intrinsic
    zouter, zinner = s[z].split(z.op.axis[0], factor=factor)
    mem_copy_write = intrin_mem_copy(inner_shape, dtype, "global", "global.vtcm")
    s[z].tensorize(zinner, mem_copy_write)

    verify(hexagon_session, s, x, y, z, size)


@T.prim_func
def scale_by_two(A: T.Buffer[(8192,), "int8"], C: T.Buffer[(8192,), "int8"]):
    for i in T.serial(
        0,
        8192,
    ):
        with T.block("C"):
            C[i] = A[i] * T.int8(2)


def test_vtcm_lowering():
    mod = tvm.IRModule.from_expr(scale_by_two.with_attr("global_symbol", "main"))
    sch = tir.Schedule(mod, debug_mask="all")
    block_c = sch.get_block("C")
    (flat,) = sch.get_loops(block_c)
    o, i, ii, iii = sch.split(flat, factors=[8, 4, 2, 128])
    cache_block = sch.cache_read(block_c, 0, storage_scope="global.vtcm")
    sch.compute_at(cache_block, o)
    lowered = tvm.lower(sch.mod["main"])

    def ir_module_has_allocate_nodes(irmod):
        nallocs = 0

        def _visit(stmt):
            nonlocal nallocs
            if isinstance(stmt, tvm.tir.Allocate):
                nallocs += 1

        tvm.tir.stmt_functor.post_order_visit(irmod["main"].body, _visit)
        return nallocs

    assert not ir_module_has_allocate_nodes(lowered), (
        "AllocateNode found in lowered IRModule, "
        "VTCM allocations should have been lowered to tir.nd_mem_alloc_with_scope"
    )
