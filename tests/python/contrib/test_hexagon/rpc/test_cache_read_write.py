import numpy as np
import tvm
from tvm import te

from tvm.contrib import utils, ndk
from tvm.contrib.hexagon.build import HexagonLauncher
import tvm.contrib.hexagon.hexagon as hexagon

from ..conftest import requires_rpc_tracker, requires_hexagon_toolchain


def intrin_mem_copy(shape, dtype, dst_scope, src_scope):
    assert len(shape) == 1
    src = te.placeholder(shape=shape, dtype=dtype, name="src")
    dst = te.compute(shape, lambda i: src[i], name="dst")
    size = shape[0] * np.dtype(dtype).itemsize

    src_buffer = tvm.tir.decl_buffer(
        shape,
        dtype,
        scope=src_scope,
        offset_factor=1,
    )

    dst_buffer = tvm.tir.decl_buffer(
        shape,
        dtype,
        scope=dst_scope,
        offset_factor=1,
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        _src = ins[0]
        _dst = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle", "tir.mem_copy", _dst.access_ptr("w"), _src.access_ptr("r"), size
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(dst.op, intrin_func, binds={src: src_buffer, dst: dst_buffer})


@requires_rpc_tracker
@requires_hexagon_toolchain
def test_hexagon(tvm_tracker_host, tvm_tracker_port, android_serial_number):
    size = 128
    outer_shape = (size,)
    factor = 16
    inner_shape = (factor,)
    dtype = "int8"

    x = te.placeholder(shape=outer_shape, dtype=dtype, name="x")
    y = te.placeholder(shape=outer_shape, dtype=dtype, name="y")
    z = te.compute(outer_shape, lambda i: x[i] + y[i], name="z")
    s = te.create_schedule(z.op)

    x_global = s.cache_read(x, "global.vtcm", [z])
    y_global = s.cache_read(y, "global.vtcm", [z])
    z_global = s.cache_write(z, "global.vtcm")

    zouter, zinner = s[z_global].split(z_global.op.axis[0], factor=factor)

    s[x_global].compute_at(s[z_global], zouter)
    s[y_global].compute_at(s[z_global], zouter)

    mem_copy_read = intrin_mem_copy(inner_shape, dtype, "global.vtcm", "global")

    (cache_read_x,) = s[x_global].op.axis
    s[x_global].tensorize(cache_read_x, mem_copy_read)

    (cache_read_y,) = s[y_global].op.axis
    s[y_global].tensorize(cache_read_y, mem_copy_read)

    mem_copy_write = intrin_mem_copy(outer_shape, dtype, "global", "global.vtcm")

    (cache_write_z,) = s[z].op.axis
    s[z].tensorize(cache_write_z, mem_copy_write)

    print(tvm.lower(s, [x, y, z]))

    target_hexagon = tvm.target.hexagon("v68", link_params=True)
    func = tvm.build(
        s, [x, y, z], tvm.target.Target(target_hexagon, host=target_hexagon), name="dmacpy"
    )
    temp = utils.tempdir()
    dso_binary = "test_binary.so"
    dso_binary_path = temp.relpath(dso_binary)
    func.save(dso_binary_path)

    launcher = HexagonLauncher(serial_number=android_serial_number)
    launcher.android_run_rpc(rpc_tracker_host=tvm_tracker_host, rpc_tracker_port=tvm_tracker_port)
    launcher.hexagon_setup()
    remote_kw = {
        "host": tvm_tracker_host,
        "port": tvm_tracker_port,
        "priority": 0,
        "timeout": 60,
    }
    launcher.hexagon_session_setup(remote_kw)
    launcher.upload(dso_binary_path, dso_binary)

    with launcher.session as sess:
        mod = launcher.get_module(dso_binary)
        xt = tvm.nd.array(np.random.uniform(size=size).astype(x.dtype), device=sess.device)
        yt = tvm.nd.array(np.random.uniform(size=size).astype(y.dtype), device=sess.device)
        zt = tvm.nd.array(np.random.uniform(size=size).astype(z.dtype), device=sess.device)
        mod["dmacpy"](xt, yt, zt)
    launcher.close()

    ref = xt.numpy() + yt.numpy()
    np.testing.assert_equal(zt.numpy(), ref)


def test_cpu():
    size = 128
    outer_shape = (size,)
    factor = 16
    inner_shape = (factor,)
    dtype = "int8"

    x = te.placeholder(shape=outer_shape, dtype=dtype, name="x")
    y = te.placeholder(shape=outer_shape, dtype=dtype, name="y")
    z = te.compute(outer_shape, lambda i: x[i] + y[i], name="z")
    s = te.create_schedule(z.op)

    x_global = s.cache_read(x, "global", [z])
    y_global = s.cache_read(y, "global", [z])
    z_global = s.cache_write(z, "global")

    zouter, zinner = s[z_global].split(z_global.op.axis[0], factor=factor)

    s[x_global].compute_at(s[z_global], zouter)
    s[y_global].compute_at(s[z_global], zouter)

    mem_copy_read = intrin_mem_copy(inner_shape, dtype, "global", "global")

    (cache_read_x,) = s[x_global].op.axis
    s[x_global].tensorize(cache_read_x, mem_copy_read)

    (cache_read_y,) = s[y_global].op.axis
    s[y_global].tensorize(cache_read_y, mem_copy_read)

    mem_copy_write = intrin_mem_copy(outer_shape, dtype, "global", "global")

    (cache_write_z,) = s[z].op.axis
    s[z].tensorize(cache_write_z, mem_copy_write)

    print(tvm.lower(s, [x, y, z]))
    func = tvm.build(s, [x, y, z], target="llvm")

    dev = tvm.device("llvm", 0)

    xt = tvm.nd.array(np.random.uniform(size=size).astype(x.dtype), dev)
    yt = tvm.nd.array(np.random.uniform(size=size).astype(y.dtype), dev)
    zt = tvm.nd.array(np.random.uniform(size=size).astype(z.dtype), dev)
    func(xt, yt, zt)

    ref = xt.numpy() + yt.numpy()
    np.testing.assert_equal(zt.numpy(), ref)
