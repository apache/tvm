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
"""Test code for broadcasting operators."""
import numpy as np
import pytest
import tvm
from tvm import te
from tvm import tir
from tvm import topi
from tvm import relay
import tvm.topi.testing
from tvm.contrib.nvcc import have_fp16
from tvm.script import tir as T

import tvm.testing


def verify_expand_dims(in_shape, out_shape, axis, num_newaxis):
    A = te.placeholder(shape=in_shape, name="A")
    B = topi.expand_dims(A, axis, num_newaxis)

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_broadcast_schedule(target)(B)
        foo = tvm.build(s, [A, B], target, name="expand_dims")
        data_npy = np.random.uniform(size=in_shape).astype(A.dtype)
        out_npy = data_npy.reshape(out_shape)
        data_nd = tvm.nd.array(data_npy, dev)
        out_nd = tvm.nd.array(np.empty(out_shape).astype(B.dtype), dev)
        foo(data_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.numpy(), out_npy)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_reinterpret(in_shape, in_dtype, out_dtype, generator):
    A = te.placeholder(shape=in_shape, name="A", dtype=in_dtype)
    B = topi.reinterpret(A, out_dtype)

    def check_device(target, dev):
        if in_dtype == "float16" and target == "cuda" and not have_fp16(dev.compute_version):
            print("Skip because %s does not have fp16 support" % target)
            return
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_elemwise_schedule(target)(B)
        foo = tvm.build(s, [A, B], target, name="reinterpret")
        data_npy = generator(in_shape).astype(in_dtype)
        out_npy = data_npy.view(B.dtype)
        data_nd = tvm.nd.array(data_npy, dev)
        out_nd = tvm.nd.array(np.empty(in_shape).astype(B.dtype), dev)
        foo(data_nd, out_nd)
        np.testing.assert_equal(out_nd.numpy(), out_npy)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_transpose(in_shape, axes):
    A = te.placeholder(shape=in_shape, name="A")
    B = topi.transpose(A, axes)

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(B)
        foo = tvm.build(s, [A, B], target, name="transpose")
        data_npy = np.arange(np.prod(in_shape)).reshape(in_shape).astype(A.dtype)
        out_npy = data_npy.transpose(axes)
        data_nd = tvm.nd.array(data_npy, dev)
        out_nd = tvm.nd.empty(out_npy.shape, device=dev, dtype=B.dtype)
        foo(data_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.numpy(), out_npy)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_reshape(src_shape, dst_shape):
    A = te.placeholder(shape=src_shape, name="A")
    B = topi.reshape(A, dst_shape)

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(B)
        foo = tvm.build(s, [A, B], target, name="reshape")
        data_npy = np.random.normal(size=src_shape).astype(A.dtype)
        out_npy = np.reshape(data_npy, newshape=dst_shape)
        data_nd = tvm.nd.array(data_npy, dev)
        out_nd = tvm.nd.empty(dst_shape, device=dev, dtype=B.dtype)
        foo(data_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.numpy(), out_npy)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_squeeze(src_shape, axis):
    A = te.placeholder(shape=src_shape, name="A")
    B = topi.squeeze(A, axis=axis)

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(B)

        foo = tvm.build(s, [A, B], target, name="squeeze")
        data_npy = np.random.normal(size=src_shape).astype(A.dtype)
        out_npy = np.squeeze(data_npy, axis=axis)
        data_nd = tvm.nd.array(data_npy, dev)
        out_nd_shape = out_npy.shape
        out_nd = tvm.nd.empty(out_nd_shape, device=dev, dtype=B.dtype)
        foo(data_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.numpy(), out_npy)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_concatenate(shapes, axis):
    def get_concat_schedule(target):
        schedule_map = {
            "cpu": topi.x86.schedule_concatenate,
            "arm_cpu": topi.arm_cpu.schedule_concatenate,
        }
        if isinstance(target, str):
            target = tvm.target.Target(target)
        for key in target.keys:
            if key in schedule_map:
                return schedule_map[key]
        return tvm.topi.testing.get_injective_schedule(target)

    tensor_l = []
    for i, shape in enumerate(shapes):
        tensor_l.append(te.placeholder(shape, name="A" + str(i)))
    out_tensor = topi.concatenate(a_tuple=tensor_l, axis=axis)

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = get_concat_schedule(target)(out_tensor)

        foo = tvm.build(s, tensor_l + [out_tensor], target, name="concatenate")
        data_npys = [np.random.normal(size=shape).astype(tensor_l[0].dtype) for shape in shapes]
        out_npy = np.concatenate(data_npys, axis=axis)
        data_nds = [tvm.nd.array(data_npy, dev) for data_npy in data_npys]
        out_nd = tvm.nd.empty(out_npy.shape, device=dev, dtype=out_tensor.dtype)
        foo(*(data_nds + [out_nd]))
        tvm.testing.assert_allclose(out_nd.numpy(), out_npy)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_stack(shapes, axis):
    tensor_l = []
    for i, shape in enumerate(shapes):
        tensor_l.append(te.placeholder(shape, name="A" + str(i)))
    out_tensor = topi.stack(tensor_l, axis)

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_broadcast_schedule(target)(out_tensor)

        foo = tvm.build(s, tensor_l + [out_tensor], target, name="stack")
        data_npys = [np.random.normal(size=shape).astype(tensor_l[0].dtype) for shape in shapes]
        out_npy = np.stack(data_npys, axis=axis)
        data_nds = [tvm.nd.array(data_npy, dev) for data_npy in data_npys]
        out_nd = tvm.nd.empty(out_npy.shape, device=dev, dtype=out_tensor.dtype)
        foo(*(data_nds + [out_nd]))
        tvm.testing.assert_allclose(out_nd.numpy(), out_npy)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_split(src_shape, indices_or_sections, axis):
    A = te.placeholder(shape=src_shape, name="A")
    tensor_l = topi.split(A, indices_or_sections, axis=axis)

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(tensor_l)

        foo = tvm.build(s, [A] + list(tensor_l), target, name="split")
        data_npy = np.random.normal(size=src_shape).astype(A.dtype)
        out_npys = np.split(data_npy, indices_or_sections, axis=axis)
        data_nd = tvm.nd.array(data_npy, dev)
        out_nds = [
            tvm.nd.empty(out_npy.shape, device=dev, dtype=tensor_l[0].dtype) for out_npy in out_npys
        ]
        foo(*([data_nd] + out_nds))
        for out_nd, out_npy in zip(out_nds, out_npys):
            tvm.testing.assert_allclose(out_nd.numpy(), out_npy)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_expand_like(in_shape, out_shape, axis):
    A = te.placeholder(shape=in_shape, name="A")
    B = te.placeholder(shape=out_shape, name="B")
    C = topi.expand_like(A, B, axis)
    s = te.create_schedule([C.op])

    def check_device(target):
        print("Running on target: %s" % target)

        dev = tvm.device(target, 0)
        f = tvm.build(s, [A, B, C], target, name="expand_like")
        input = np.random.uniform(size=in_shape).astype(A.dtype)
        tvm_input = tvm.nd.array(input, dev)

        odim = len(out_shape)
        real_axis = [x if x >= 0 else x + odim for x in axis]
        real_axis = sorted(real_axis)
        for x in real_axis:
            input = np.expand_dims(input, x).astype(A.dtype)
        for x in real_axis:
            input = np.concatenate([input] * out_shape[x], axis=x).astype(A.dtype)
        assert input.shape == out_shape

        tvm_shape_like = tvm.nd.array(np.zeros(out_shape).astype(B.dtype), dev)
        out = tvm.nd.array(np.zeros(out_shape).astype(A.dtype), dev)
        f(tvm_input, tvm_shape_like, out)
        tvm.testing.assert_allclose(out.numpy(), input)

    for target in ["llvm"]:
        check_device(target)


def verify_flip(in_shape, axis):
    A = te.placeholder(shape=in_shape, name="A")
    B = topi.flip(A, axis) + 1

    def check_device(target):
        dev = tvm.device(target, 0)
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            return
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(B)

        foo = tvm.build(s, [A, B], target, name="reverse")
        x_np = np.random.uniform(size=in_shape).astype(A.dtype)
        out_npy = np.flip(x_np, axis) + 1
        data_nd = tvm.nd.array(x_np, dev)
        out_nd = tvm.nd.empty(out_npy.shape, device=dev, dtype=A.dtype)
        foo(data_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.numpy(), out_npy)

    for target in ["llvm", "cuda", "opencl", "sdaccel", "aocl_sw_emu"]:
        check_device(target)


@tvm.testing.uses_gpu
def test_reverse_sequence():
    def verify_reverse_sequence(in_data, seq_lengths, batch_axis, seq_axis, ref_res):
        seq_lengths = np.array(seq_lengths).astype("int32")
        A = te.placeholder(shape=in_data.shape, name="A", dtype=str(in_data.dtype))
        B = te.placeholder(shape=seq_lengths.shape, name="B", dtype=str(seq_lengths.dtype))
        C = topi.reverse_sequence(A, B, seq_axis, batch_axis)

        def check_device(target, dev):
            print("Running on target: %s" % target)
            with tvm.target.Target(target):
                s = tvm.topi.testing.get_injective_schedule(target)(C)

            foo = tvm.build(s, [A, B, C], target, name="reverse_sequence")

            data_nd = tvm.nd.array(in_data, dev)
            seq_lengths_nd = tvm.nd.array(seq_lengths, dev)
            out_nd = tvm.nd.empty(in_data.shape, device=dev, dtype=A.dtype)
            foo(data_nd, seq_lengths_nd, out_nd)
            tvm.testing.assert_allclose(out_nd.numpy(), ref_res)

        for target, dev in tvm.testing.enabled_targets():
            check_device(target, dev)

    indata = np.array(np.arange(0, 16)).reshape([4, 4]).astype("int32")
    result = [[0, 5, 10, 15], [4, 1, 6, 11], [8, 9, 2, 7], [12, 13, 14, 3]]
    verify_reverse_sequence(indata, [1, 2, 3, 4], 1, 0, np.array(result))
    verify_reverse_sequence(indata, [1, 2, 3, 4], -1, 0, np.array(result))
    verify_reverse_sequence(
        indata.astype("float32"), [1, 2, 3, 4], 1, 0, np.array(result).astype("float32")
    )

    indata = np.array(np.arange(0, 16)).reshape([4, 4]).astype("int32")
    result = [[0, 1, 2, 3], [5, 4, 6, 7], [10, 9, 8, 11], [15, 14, 13, 12]]
    verify_reverse_sequence(indata, [1, 2, 3, 4], 0, 1, np.array(result))
    verify_reverse_sequence(indata, [1, 2, 3, 4], 0, -1, np.array(result))
    verify_reverse_sequence(
        indata.astype("float32"), [1, 2, 3, 4], 0, 1, np.array(result).astype("float32")
    )

    indata = np.array(np.arange(0, 16)).reshape([4, 4]).astype("int32")
    result = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [15, 14, 13, 12]]
    verify_reverse_sequence(indata, [-1, 0, 1, 5], 0, 1, np.array(result))

    indata = np.array(np.arange(0, 54)).reshape([2, 3, 3, 3]).astype("int32")
    result = [
        [
            [[18, 19, 20], [21, 22, 23], [24, 25, 26]],
            [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        ],
        [
            [[45, 46, 47], [48, 49, 50], [51, 52, 53]],
            [[36, 37, 38], [39, 40, 41], [42, 43, 44]],
            [[27, 28, 29], [30, 31, 32], [33, 34, 35]],
        ],
    ]
    verify_reverse_sequence(indata, [3, 3], 0, 1, np.array(result))

    indata = np.array(np.arange(0, 54)).reshape([2, 3, 3, 3]).astype("int32")
    result = [
        [
            [[9, 10, 11], [21, 22, 23], [15, 16, 17]],
            [[0, 1, 2], [12, 13, 14], [6, 7, 8]],
            [[18, 19, 20], [3, 4, 5], [24, 25, 26]],
        ],
        [
            [[36, 37, 38], [48, 49, 50], [42, 43, 44]],
            [[27, 28, 29], [39, 40, 41], [33, 34, 35]],
            [[45, 46, 47], [30, 31, 32], [51, 52, 53]],
        ],
    ]
    verify_reverse_sequence(indata, [2, 3, 2], 2, 1, np.array(result))

    indata = np.array(np.arange(0, 16)).reshape([4, 4]).astype("int32")
    result = []
    with pytest.raises(Exception) as execinfo:
        verify_reverse_sequence(indata, [2, 3, 2, 4, 5], 1, 0, np.array(result))

    assert (
        "For reverse_sequnece seq_lengths size should match with dimension of batch axis,"
        " but got dimension of batch_axis = 4, and seq_length size = 5" in execinfo.value.args[0]
    )


def verify_take(src_shape, indices_src, axis=None, mode="clip", indices_dtype="int32"):
    src_dtype = "float32"
    indices_src = np.array(indices_src, dtype=indices_dtype)
    A = te.placeholder(shape=src_shape, dtype=src_dtype, name="A")
    indices = te.placeholder(shape=indices_src.shape, dtype=indices_dtype, name="indices")
    if axis is None:
        out_tensor = topi.take(a=A, indices=indices, mode=mode)
    else:
        out_tensor = topi.take(a=A, indices=indices, axis=axis, mode=mode)

    def check_device(target):
        dev = tvm.device(target, 0)
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            return
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(out_tensor)

        foo = tvm.build(s, [A] + [indices] + [out_tensor], target, name="take")
        shape_size = 1
        for i in range(len(src_shape)):
            shape_size = shape_size * src_shape[i]
        data_npy = np.arange(shape_size, dtype=src_dtype).reshape((src_shape))

        if axis is None:
            np_mode = "raise" if mode == "fast" else mode
            out_npys = np.take(data_npy, indices_src, mode=np_mode)
        else:
            np_mode = "raise" if mode == "fast" else mode
            out_npys = np.take(data_npy, indices_src, axis=axis, mode=np_mode)
        data_nd = tvm.nd.array(data_npy, dev)
        indices_nd = tvm.nd.array(indices_src, dev)
        out_nd = tvm.nd.empty(out_npys.shape, device=dev, dtype=src_dtype)
        foo(data_nd, indices_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.numpy(), out_npys)

    for target in ["llvm", "opencl", "sdaccel", "aocl_sw_emu"]:
        check_device(target)


def verify_strided_slice(in_shape, begin, end, strides=None, axes=None):
    A = te.placeholder(shape=in_shape, name="A")
    strides = [1, 1, 1] if strides is None else strides
    if axes:
        strides = [strides[axis] for axis in axes]
    B = topi.strided_slice(A, begin, end, strides, axes) + 1

    def check_device(target):
        dev = tvm.device(target, 0)
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            return
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(B)

        foo = tvm.build(s, [A, B], target, name="stride_slice")
        x_np = np.random.uniform(size=in_shape).astype(A.dtype)
        out_npy = tvm.topi.testing.strided_slice_python(x_np, begin, end, strides, axes=axes) + 1
        data_nd = tvm.nd.array(x_np, dev)
        out_nd = tvm.nd.empty(out_npy.shape, device=dev, dtype=A.dtype)
        foo(data_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.numpy(), out_npy)

    for target in ["llvm", "opencl", "sdaccel", "aocl_sw_emu"]:
        check_device(target)


def verify_dynamic_strided_slice(in_shape, begin, end, strides=None):
    A = te.placeholder(shape=in_shape, name="A")
    Begin = te.placeholder(shape=[len(in_shape)], name="begin", dtype="int64")
    End = te.placeholder(shape=[len(in_shape)], name="end", dtype="int64")
    Strides = te.placeholder(shape=[len(in_shape)], name="strides", dtype="int64")
    strides = [1, 1, 1] if strides is None else strides
    B = topi.strided_slice(A, Begin, End, Strides) + 1

    def check_device(target):
        dev = tvm.device(target, 0)
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            return
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(B)

        foo = tvm.build(s, [A, Begin, End, Strides, B], target, name="stride_slice")
        x_np = np.random.uniform(size=in_shape).astype(A.dtype)
        out_npy = tvm.topi.testing.strided_slice_python(x_np, begin, end, strides) + 1
        data_nd = tvm.nd.array(x_np, dev)
        out_nd = tvm.nd.empty(out_npy.shape, device=dev, dtype=A.dtype)
        begin_nd = tvm.nd.array(np.array(begin).astype("int64"), dev)
        end_nd = tvm.nd.array(np.array(end).astype("int64"), dev)
        strides_nd = tvm.nd.array(np.array(strides).astype("int64"), dev)
        foo(data_nd, begin_nd, end_nd, strides_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.numpy(), out_npy)

    for target in ["llvm", "opencl", "sdaccel", "aocl_sw_emu", "opencl --device=adreno"]:
        check_device(target)


def verify_relax_dynamic_strided_slice(in_shape, begin, end, strides, output_shape):
    A = te.placeholder(shape=in_shape, name="A")
    Begin = te.placeholder(shape=[len(in_shape)], name="begin", dtype="int64")
    End = te.placeholder(shape=[len(in_shape)], name="end", dtype="int64")
    Strides = te.placeholder(shape=[len(in_shape)], name="strides", dtype="int64")

    B = topi.dynamic_strided_slice(A, Begin, End, Strides, output_shape) + 1

    def check_device(target):
        dev = tvm.device(target, 0)
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            return
        print("Running on target: %s" % target)
        x_np = np.random.uniform(size=in_shape).astype(A.dtype)
        out_npy = tvm.topi.testing.strided_slice_python(x_np, begin, end, strides) + 1
        data_nd = tvm.nd.array(x_np, dev)
        tvm_out = tvm.nd.empty(out_npy.shape, device=dev, dtype=A.dtype)
        begin_nd = tvm.nd.array(np.array(begin).astype("int64"), dev)
        end_nd = tvm.nd.array(np.array(end).astype("int64"), dev)
        strides_nd = tvm.nd.array(np.array(strides).astype("int64"), dev)

        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(B)
        foo = tvm.build(s, [A, Begin, End, Strides, B], target, name="stride_slice")
        foo(data_nd, begin_nd, end_nd, strides_nd, tvm_out)
        tvm_out_npy = tvm_out.numpy()
        assert out_npy.shape == tvm_out_npy.shape
        tvm.testing.assert_allclose(tvm_out_npy, out_npy)

    for target in ["llvm", "opencl", "sdaccel", "aocl_sw_emu"]:
        check_device(target)


def verify_strided_set(in_shape, v_shape, begin, end, strides=None):
    A = te.placeholder(shape=in_shape, name="A")
    V = te.placeholder(shape=v_shape, name="V")
    b = te.placeholder(shape=(len(begin),), name="b", dtype="int32")
    e = te.placeholder(shape=(len(end),), name="e", dtype="int32")
    if strides is not None:
        st = te.placeholder(shape=(len(strides),), name="st", dtype="int32")
        B = topi.strided_set(A, V, b, e, st) + 1
    else:
        B = topi.strided_set(A, V, b, e) + 1

    def check_device(target):
        dev = tvm.device(target, 0)
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            return
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(B)

        if strides is not None:
            foo = tvm.build(s, [A, V, b, e, st, B], target, name="stride_set")
            s_np = np.asarray(strides).astype("int32")
            s_nd = tvm.nd.array(s_np, dev)
        else:
            foo = tvm.build(s, [A, V, b, e, B], target, name="stride_set")
        x_np = np.random.uniform(size=in_shape).astype(A.dtype)
        v_np = np.random.uniform(size=v_shape).astype(V.dtype)
        b_np = np.asarray(begin).astype("int32")
        e_np = np.asarray(end).astype("int32")
        out_npy = tvm.topi.testing.strided_set_python(x_np, v_np, begin, end, strides) + 1
        data_nd = tvm.nd.array(x_np, dev)
        v_nd = tvm.nd.array(v_np, dev)
        b_nd = tvm.nd.array(b_np, dev)
        e_nd = tvm.nd.array(e_np, dev)
        out_nd = tvm.nd.empty(out_npy.shape, device=dev, dtype=A.dtype)
        if strides is not None:
            foo(data_nd, v_nd, b_nd, e_nd, s_nd, out_nd)
        else:
            foo(data_nd, v_nd, b_nd, e_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.numpy(), out_npy)

    for target in ["llvm", "opencl", "sdaccel", "aocl_sw_emu"]:
        check_device(target)


def verify_gather(data, axis, indices):
    data = np.asarray(data)
    indices = np.asarray(indices)

    var_data = te.placeholder(shape=data.shape, dtype=data.dtype.name, name="data")
    var_indices = te.placeholder(shape=indices.shape, dtype=indices.dtype.name, name="indices")
    out_tensor = topi.gather(var_data, axis, var_indices)

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(out_tensor)

        func = tvm.build(s, [var_data, var_indices, out_tensor], target, name="gather")
        out_npys = tvm.topi.testing.gather_python(data, axis, indices)

        data_nd = tvm.nd.array(data, dev)
        indices_nd = tvm.nd.array(indices, dev)
        out_nd = tvm.nd.empty(out_npys.shape, device=dev, dtype=data.dtype.name)
        func(data_nd, indices_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.numpy(), out_npys)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_gather_nd(src_shape, indices_src, indices_dtype):
    src_dtype = "float32"
    indices_src = np.array(indices_src, dtype=indices_dtype)
    A = te.placeholder(shape=src_shape, dtype=src_dtype, name="A")
    indices = te.placeholder(shape=indices_src.shape, dtype=indices_dtype, name="indices")
    out_tensor = topi.gather_nd(a=A, indices=indices)

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(out_tensor)

        func = tvm.build(s, [A, indices, out_tensor], target, name="take")
        shape_size = 1
        for i in range(len(src_shape)):
            shape_size = shape_size * src_shape[i]
        data_npy = np.arange(shape_size, dtype=src_dtype).reshape((src_shape))
        out_npys = tvm.topi.testing.gather_nd_python(data_npy, indices_src)

        data_nd = tvm.nd.array(data_npy, dev)
        indices_nd = tvm.nd.array(indices_src, dev)
        out_nd = tvm.nd.empty(out_npys.shape, device=dev, dtype=src_dtype)
        func(data_nd, indices_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.numpy(), out_npys)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_arange(start, stop, step):
    if start is None and step is None:
        A = topi.arange(stop)
        a_np = np.arange(stop)
    elif start is None:
        A = topi.arange(stop, step=step)
        a_np = np.arange(stop, step=step)
    elif step is None:
        A = topi.arange(start, stop)
        a_np = np.arange(start, stop)
    else:
        A = topi.arange(start, stop, step)
        a_np = np.arange(start, stop, step)

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(A)
        f = tvm.build(s, [A], target, name="arange")
        a_nd = tvm.nd.empty(a_np.shape, dtype="float32", device=dev)
        f(a_nd)
        tvm.testing.assert_allclose(a_nd.numpy(), a_np)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_repeat(in_shape, repeats, axis):
    A = te.placeholder(shape=in_shape, name="A")
    B = topi.repeat(A, repeats, axis)

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_broadcast_schedule(target)(B)
        foo = tvm.build(s, [A, B], target, name="repeat")
        data_npy = np.random.uniform(size=in_shape).astype(A.dtype)
        out_npy = np.repeat(data_npy, repeats, axis)
        data_nd = tvm.nd.array(data_npy, dev)
        out_nd = tvm.nd.array(np.empty(out_npy.shape).astype(B.dtype), dev)
        foo(data_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.numpy(), out_npy)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_tile(in_shape, reps):
    A = te.placeholder(shape=in_shape, name="A")
    B = topi.tile(A, reps)

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_broadcast_schedule(target)(B)
        foo = tvm.build(s, [A, B], target, name="tile")
        data_npy = np.random.uniform(size=in_shape).astype(A.dtype)
        out_npy = np.tile(data_npy, reps)
        data_nd = tvm.nd.array(data_npy, dev)
        out_nd = tvm.nd.array(np.empty(out_npy.shape).astype(B.dtype), dev)
        foo(data_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.numpy(), out_npy)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_where(in_shape):
    Cond = te.placeholder(shape=in_shape, name="cond")
    dtype = Cond.dtype
    A = te.placeholder(shape=in_shape, name="A")
    B = te.placeholder(shape=in_shape, name="B")
    C = topi.where(Cond, A, B)

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_broadcast_schedule(target)(C)
        f = tvm.build(s, [Cond, A, B, C], target, name="where")
        cond_npy = np.random.uniform(low=-1, high=1, size=in_shape).astype(dtype)
        x_npy = np.random.uniform(size=in_shape).astype(dtype)
        y_npy = np.random.uniform(size=in_shape).astype(dtype)
        out_npy = np.where(cond_npy, x_npy, y_npy)
        cond_nd = tvm.nd.array(cond_npy, dev)
        x_nd = tvm.nd.array(x_npy, dev)
        y_nd = tvm.nd.array(y_npy, dev)
        out_nd = tvm.nd.array(np.empty(out_npy.shape).astype(C.dtype), dev)
        f(cond_nd, x_nd, y_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.numpy(), out_npy)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_one_hot(indices_shape, depth, on_value, off_value, axis, dtype):
    indices = te.placeholder(shape=indices_shape, name="indices", dtype="int32")
    on_value_const = tvm.tir.const(on_value, dtype)
    off_value_const = tvm.tir.const(off_value, dtype)
    one_hot_result = topi.transform.one_hot(
        indices, on_value_const, off_value_const, depth, axis, dtype
    )

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(one_hot_result)
        fn = tvm.build(s, [indices, one_hot_result], target, name="one_hot")
        indices_npy = np.random.randint(0, depth, size=indices_shape).astype(indices.dtype)
        out_npy = tvm.topi.testing.one_hot(indices_npy, on_value, off_value, depth, axis, dtype)
        indices_nd = tvm.nd.array(indices_npy, dev)
        out_nd = tvm.nd.array(np.empty(out_npy.shape).astype(one_hot_result.dtype), dev)
        fn(indices_nd, out_nd)
        out_topi = out_nd.numpy()
        tvm.testing.assert_allclose(out_topi, out_npy)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_unravel_index(indices, shape, dtype, indice_dtype="int64"):
    x_data = np.array(indices).astype(indice_dtype)
    y_data = np.array(shape).astype(dtype)
    if len(x_data.shape) == 1:
        dst_shape = [y_data.shape[0], x_data.shape[0]]
    else:
        dst_shape = [y_data.shape[0]]

    X = te.placeholder(shape=x_data.shape, dtype=indice_dtype, name="X")
    Y = te.placeholder(shape=y_data.shape, dtype=dtype, name="Y")
    Z = topi.unravel_index(X, Y)

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(Z)
        foo = tvm.build(s, [X, Y, Z], target, name="unravel_index")

        out_npy = np.unravel_index(x_data, y_data)
        datax_nd = tvm.nd.array(x_data, dev)
        datay_nd = tvm.nd.array(y_data, dev)
        out_nd = tvm.nd.empty(dst_shape, device=dev, dtype=Z.dtype)
        foo(datax_nd, datay_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.numpy(), out_npy)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_sparse_to_dense(sparse_indices, sparse_values, default_value, output_shape, xpected):
    sparse_indices_data = np.array(sparse_indices)
    sparse_values_data = np.array(sparse_values)
    output_shape_data = np.array(output_shape)
    default_value_data = np.array(default_value)

    A = te.placeholder(
        shape=sparse_indices_data.shape, name="sparse_indices", dtype=str(sparse_indices_data.dtype)
    )
    B = te.placeholder(
        shape=sparse_values_data.shape, name="sparse_values", dtype=str(sparse_values_data.dtype)
    )
    if default_value is None:
        args = [A, B]
        D = topi.sparse_to_dense(A, output_shape, B)
    else:
        C = te.placeholder(shape=(), name="default_value", dtype=str(default_value_data.dtype))
        args = [A, B, C]
        D = topi.sparse_to_dense(A, output_shape, B, C)

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(D)

        foo = tvm.build(s, args + [D], target, name="sparse_to_dense")

        sparse_indices_nd = tvm.nd.array(sparse_indices_data, dev)
        sparse_values_nd = tvm.nd.array(sparse_values_data, dev)
        out_nd = tvm.nd.empty(output_shape_data, device=dev, dtype=B.dtype)

        if default_value is None:
            foo(sparse_indices_nd, sparse_values_nd, out_nd)
        else:
            default_value_nd = tvm.nd.array(default_value_data, dev)
            foo(sparse_indices_nd, sparse_values_nd, default_value_nd, out_nd)

        tvm.testing.assert_allclose(out_nd.numpy(), np.array(xpected))

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_matrix_set_diag(input_shape, diagonal_shape, dtype, k=0, align="RIGHT_LEFT"):
    input = te.placeholder(shape=input_shape, name="input", dtype=dtype)
    diagonal = te.placeholder(shape=diagonal_shape, name="diagonal", dtype=dtype)
    matrix_set_diag_result = topi.transform.matrix_set_diag(input, diagonal, k, align)

    def check_device(target, dev):
        dev = tvm.device(target, 0)
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(matrix_set_diag_result)
        fn = tvm.build(s, [input, diagonal, matrix_set_diag_result], target, name="matrix_set_diag")
        input_npy = np.random.randint(-100, 100, size=input_shape).astype(dtype)
        diagonal_npy = np.random.randint(-100, 100, size=diagonal_shape).astype(dtype)
        out_npy = tvm.topi.testing.matrix_set_diag(input_npy, diagonal_npy, k, align)
        input_nd = tvm.nd.array(input_npy, dev)
        diagonal_nd = tvm.nd.array(diagonal_npy, dev)
        out_nd = tvm.nd.array(np.empty(out_npy.shape).astype(matrix_set_diag_result.dtype), dev)
        fn(input_nd, diagonal_nd, out_nd)
        out_topi = out_nd.numpy()
        tvm.testing.assert_allclose(out_topi, out_npy)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_adv_index(data_shape, index_shapes, indice_dtype="int64"):
    dtype = "float32"
    data = te.placeholder(shape=data_shape, name="data", dtype=dtype)
    indices = []
    np_data = np.random.uniform(size=data_shape).astype(dtype)
    np_indices = []
    for i, index_shape in enumerate(index_shapes):
        limit = data_shape[i]
        np_indices.append(np.random.uniform(0, limit - 1, size=index_shape).astype(indice_dtype))
        indices.append(
            te.placeholder(shape=index_shape, name="index_{}".format(i), dtype=indice_dtype)
        )
    np_out = np_data[tuple(np_indices)]
    out = topi.adv_index(data, indices)

    def check_device(target, dev):
        dev = tvm.device(target, 0)
        if not dev.exist:
            print("Skip because %s is not enabled" % target)
            return
        print("Running on target: %s" % target)
        with tvm.target.create(target):
            s = tvm.topi.testing.get_injective_schedule(target)(out)

        func = tvm.build(s, [data] + indices + [out], target, name="adv_index")

        nd_list = [tvm.nd.array(np_data, dev)]
        for np_index in np_indices:
            nd_list.append(tvm.nd.array(np_index, dev))
        nd_list.append(tvm.nd.empty(out.shape, device=dev, dtype=data.dtype))

        func(*nd_list)
        tvm.testing.assert_allclose(nd_list[-1].numpy(), np.array(np_out))

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


def verify_trilu(input_shape, upper, k=0):
    x = te.placeholder(shape=input_shape, name="x", dtype="float32")
    k_tir = tvm.tir.const(k, dtype="int32")
    trilu_result = topi.transform.trilu(x, k_tir, upper)

    def check_device(target, dev):
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(trilu_result)
        fn = tvm.build(s, [x, trilu_result], target, name="trilu")
        x_npy = np.random.normal(size=input_shape).astype(x.dtype)
        if upper:
            out_npy = np.triu(x_npy, k)
        else:
            out_npy = np.tril(x_npy, k)
        x_nd = tvm.nd.array(x_npy, dev)
        out_nd = tvm.nd.array(np.empty(x_npy.shape).astype(trilu_result.dtype), dev)
        fn(x_nd, out_nd)
        out_topi = out_nd.numpy()
        tvm.testing.assert_allclose(out_topi, out_npy)

    for target, dev in tvm.testing.enabled_targets():
        check_device(target, dev)


@tvm.testing.uses_gpu
def test_strided_slice():
    verify_strided_slice((3, 4, 3), [0, 0, 0], [4, -5, 4], [1, -1, 2])
    verify_strided_slice((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1])
    verify_strided_slice((3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1])
    verify_strided_slice((3, 4, 3), [1, 0, 0], [2, 2, 3], [1, 1, 2])
    verify_strided_slice((3, 4, 3), [1, -1, 0], [2, -3, 3], [1, -1, 1])
    verify_strided_slice((3, 4, 3), [1, 1, 0], [4, 4, 3])
    verify_strided_slice((3, 4, 3), [0, 2, 0], [1, 2, 3])
    verify_strided_slice((3, 4, 3), [0, 0, 0], [None, None, None])


def test_strided_slice_with_dynamic_bounds():
    """The begin/end of strided_slice can be a PrimExpr

    Where topi.dynamic_strided_slice uses begin/end values provided at
    runtime, strided_slice takes begin/end values at compile-time.
    However, these begin/end values may depend on dynamic variables.
    Previously, these resulted in dispatch to
    `tvm::topi::dynamic_strided_slice`, ignoring the `axes` argument.
    """
    A = te.placeholder(shape=[16, 32, 64], name="A")
    begins = [tir.Var("begin1", "int32"), tir.Var("begin2", "int32")]
    ends = [tir.Var("end1", "int32"), tir.Var("end2", "int32")]
    strides = [1, 1]
    axes = [2, 1]

    # Dummy tensor to provide begin/end variables in PrimFunc scope.
    # Outside of a test case, these would typically be provided
    # through another means, or bound to a static value at a later
    # point.
    Dummy = te.placeholder(shape=[*begins, *ends], name="Dummy")

    B = topi.strided_slice(A, begins, ends, strides, axes)

    func = te.create_prim_func([A, Dummy, B]).without_attr("global_symbol")

    @T.prim_func(private=True)
    def expected(
        A: T.Buffer((16, 32, 64), "float32"),
        var_Dummy: T.handle,
        B_handle: T.handle,
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        begin1, begin2, end1, end2 = T.int32(), T.int32(), T.int32(), T.int32()
        Dummy = T.match_buffer(var_Dummy, (begin1, begin2, end1, end2))
        B = T.match_buffer(B_handle, (16, end2 - begin2, end1 - begin1))
        for iters in T.grid(*B.shape):
            with T.block("T_dynamic_strided_slice_with_axes"):
                i, j, k = T.axis.remap("SSS", iters)
                B[i, j, k] = A[i, j + begin2, k + begin1]

    tvm.ir.assert_structural_equal(expected, func)


@tvm.testing.uses_gpu
def test_dynamic_strided_slice():
    verify_dynamic_strided_slice((3, 4, 3), [0, 0, 0], [4, -5, 4], [1, -1, 2])
    verify_dynamic_strided_slice((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1])
    verify_dynamic_strided_slice((3, 4, 3), [1, 0, 0], [2, 2, 3], [1, 1, 2])
    verify_dynamic_strided_slice((3, 4, 3), [1, 1, 0], [4, 4, 3])
    verify_dynamic_strided_slice((3, 4, 3), [0, 2, 0], [1, 2, 3])


@tvm.testing.uses_gpu
def test_relax_dynamic_strided_slice():
    verify_relax_dynamic_strided_slice((3, 4, 3), [0, 0, 0], [4, -5, 4], [1, -1, 2], [3, 1, 2])
    verify_relax_dynamic_strided_slice((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1], [1, 3, 3])
    verify_relax_dynamic_strided_slice((3, 4, 3), [1, 0, 0], [2, 2, 3], [1, 1, 2], [1, 2, 2])
    verify_relax_dynamic_strided_slice((3, 4, 3), [1, 1, 0], [4, 4, 3], [1, 1, 1], [2, 3, 3])
    verify_relax_dynamic_strided_slice((3, 4, 3), [0, 2, 0], [1, 2, 3], [1, 1, 1], [1, 0, 3])


@tvm.testing.uses_gpu
def test_strided_set():
    verify_strided_set((3, 4, 3), (3, 2, 2), [0, 3, 0], [4, 1, 4], [1, -1, 2])
    verify_strided_set((3, 4, 3), (3, 1, 2), [0, 0, 0], [4, -5, 4], [1, -1, 2])
    verify_strided_set((3, 4, 3), (1, 3, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1])
    verify_strided_set((3, 4, 3), (1, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1])
    verify_strided_set((3, 4, 3), (1, 2, 2), [1, 0, 0], [2, 2, 3], [1, 1, 2])
    verify_strided_set((3, 4, 3), (1, 2, 3), [1, -1, 0], [2, -3, 3], [1, -1, 1])
    verify_strided_set((3, 4, 3), (1, 2, 3), [1, 1, 0], [2, 3, 3], [1])
    verify_strided_set((3, 4, 3), (2, 3, 3), [1, 1, 0], [4, 4, 3])
    verify_strided_set((3, 4, 3), (2, 3, 3), [1, 1], [4, 4, 3])


@tvm.testing.uses_gpu
def test_expand_dims():
    verify_expand_dims((3, 10), (3, 10, 1, 1), 2, 2)
    verify_expand_dims((3, 10), (1, 3, 10), -3, 1)


@tvm.testing.uses_gpu
def test_reinterpret():
    verify_reinterpret((1000,), "float32", "int32", lambda shape: np.random.randn(*shape) * 1000)
    verify_reinterpret((1000,), "float16", "int16", lambda shape: np.random.randn(*shape) * 100)
    verify_reinterpret(
        (1000,), "int16", "uint16", lambda shape: np.random.randint(-1000, 1000, size=shape)
    )
    verify_reinterpret(
        (1000,), "uint32", "int32", lambda shape: np.random.randint(0, 2**32 - 1, size=shape)
    )
    verify_reinterpret(
        (1000,), "uint32", "int32", lambda shape: np.random.randint(0, 2**32 - 1, size=shape)
    )


@tvm.testing.uses_gpu
def test_transpose():
    verify_transpose((3, 10, 2), (1, 0, 2))
    verify_transpose((3, 10, 5), (2, 0, 1))
    verify_transpose((3, 10), None)


@tvm.testing.parametrize_targets("cuda", "rocm")
def test_transpose_unfused_schedule(target, dev):
    shape = (100, tvm.target.Target(target).thread_warp_size + 3)
    x = relay.var("x", relay.TensorType(shape, "float32"))
    f = relay.transpose(x)
    r = np.random.rand(*shape)
    func = relay.create_executor(
        kind="graph", mod=tvm.IRModule.from_expr(relay.Function([x], f)), device=dev, target=target
    ).evaluate()
    tvm.testing.assert_allclose(func(r).numpy(), np.transpose(r))

    # We want to make sure schedule does not fire here, but there is no way of
    # inspecting which schedules were used.
    x = relay.var("x", relay.TensorType(shape, "float32"))
    y = relay.var("y", relay.TensorType(shape, "float32"))
    f = relay.transpose(x + y)
    func = relay.create_executor(
        kind="graph",
        mod=tvm.IRModule.from_expr(relay.Function([x, y], f)),
        device=dev,
        target=target,
    ).evaluate()
    tvm.testing.assert_allclose(func(r, r).numpy(), np.transpose(r + r))


@tvm.testing.uses_gpu
def test_reshape():
    verify_reshape((1, 2, 3, 4), (2, 3, 4))
    verify_reshape((4, 2, 3, 4), (2, 4, 12))
    verify_reshape((4, 2, 3, 4), (2, 48))
    verify_reshape((16,), (2, 2, 2, 2))
    verify_reshape((4, 0), (2, 0, 2))


@tvm.testing.uses_gpu
def test_where():
    verify_where(())
    verify_where((1, 2, 3, 4))


@tvm.testing.uses_gpu
def test_squeeze():
    verify_squeeze((1, 2, 3, 4), 0)
    verify_squeeze((1, 2, 1, 4), None)
    verify_squeeze((1, 1, 1, 4), (1, 2))
    verify_squeeze((1, 1, 1, 1), None)
    verify_squeeze((1, 1, 1, 1), ())

    # a special case to trigger inline let expression
    A = te.placeholder((2,), "float32", "A")
    E = topi.squeeze(A)
    C = te.compute((1,), lambda i: E[(2 * A[0] - 1).astype("int32")])
    for target in ["llvm", "cuda", "opencl"]:
        dev = tvm.device(target, 0)
        if tvm.testing.device_enabled(target):
            with tvm.target.Target(target):
                s = tvm.topi.testing.get_injective_schedule(target)(C)
                func = tvm.build(s, [A, C])
            a = tvm.nd.array(np.array((1, 2)).astype("float32"), device=dev)
            c = tvm.nd.empty((1,), dtype="float32", device=dev)
            func(a, c)
            assert c.numpy()[0] == 2


@tvm.testing.uses_gpu
def test_concatenate():
    verify_concatenate([(2,), (2,), (2,)], -1)
    verify_concatenate([(2, 3, 4), (2, 2, 4), (2, 5, 4)], 1)
    verify_concatenate([(1, 2, 4), (1, 2, 3), (1, 2, 7), (1, 2, 8), (1, 2, 1)], -1)
    verify_concatenate([(5, 6, 7, 3), (16, 6, 7, 3), (12, 6, 7, 3), (8, 6, 7, 3), (2, 6, 7, 3)], 0)
    verify_concatenate([(1, 14400), (1, 2400), (1, 640), (1, 240)], 1)


@tvm.testing.uses_gpu
def test_stack():
    verify_stack([(2,), (2,), (2,)], -1)
    verify_stack([(2,), (2,), (2,)], 1)
    verify_stack([(2,), (2,), (2,)], 0)
    verify_stack([(2, 2, 4), (2, 2, 4), (2, 2, 4)], 1)
    verify_stack([(2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4)], -1)


@tvm.testing.uses_gpu
def test_split():
    verify_split((2, 12, 3), 3, 1)
    verify_split((2, 12, 3), [2, 4], 1)
    verify_split((10, 12, 24), [5, 7, 9], -1)


@tvm.testing.uses_gpu
def test_flip():
    verify_flip((3, 4, 3), 1)
    verify_flip((3, 4, 3), 0)
    verify_flip((3, 4, 3), 2)
    verify_flip((3, 4, 3), -1)
    verify_flip((3, 4, 3), -3)
    verify_flip((3, 4, 3), -2)


@tvm.testing.requires_llvm
def test_expand_like():
    verify_expand_like((3,), (2, 3), [0])
    verify_expand_like((2,), (2, 3), [1])
    verify_expand_like((3, 4), (3, 5, 4), [1])
    verify_expand_like((5, 7), (5, 6, 7, 8), [1, 3])


@tvm.testing.uses_gpu
def test_take():
    verify_take((4,), [1])
    verify_take((4,), [[0, 1, 2, 3]])
    verify_take((3, 3, 3), [[11, 25]])
    verify_take((4,), [[0, 1], [2, 3]])
    verify_take((4,), [1], 0)
    verify_take((2, 2), [[[1, 0], [0, 1]]], 0)
    verify_take((2, 2), [[[1, 0], [0, 1]]], 1)
    verify_take((4, 3, 5, 6), [[2, 1, 0, 0]], -2)
    verify_take((3, 4), [-5, 20])
    verify_take((3, 4), [-5, 20], mode="wrap")
    verify_take((3, 4), [-1, 2], axis=0)
    verify_take((3, 4), [-1, 2], axis=0, mode="wrap")
    verify_take((3, 4), [-1, 2], axis=1)
    verify_take((3, 4), [-1, 2], axis=1, mode="wrap")
    verify_take((3, 3, 3), [[11, 25]], mode="fast")
    verify_take((3, 4), [0, 2], axis=0, mode="fast")
    verify_take((3, 4), [0, 2], axis=1, mode="fast")
    verify_take((3, 5, 7), [[0, 2], [0, 2], [0, 2], [0, 2]], axis=0, mode="fast")
    verify_take((3, 5, 7), [[0, 2], [0, 2], [0, 2], [0, 2]], axis=1, mode="fast")
    verify_take((3, 5, 7), [[0, 2], [0, 2], [0, 2], [0, 2]], axis=2, mode="fast")
    verify_take((3, 4), [1, 2], axis=1, indices_dtype="uint32")
    verify_take((3, 4), [1, 2], axis=1, mode="wrap", indices_dtype="uint16")
    verify_take((3, 3, 3), [[11, 20]], mode="fast", indices_dtype="uint8")


@tvm.testing.uses_gpu
def test_gather():
    verify_gather([[1, 2], [3, 4]], 1, [[0, 0], [1, 0]])
    verify_gather(np.random.randn(4, 7, 5), 0, np.random.randint(low=0, high=4, size=(1, 7, 5)))
    verify_gather(np.random.randn(4, 7, 5), 0, np.random.randint(low=0, high=4, size=(4, 7, 5)))
    verify_gather(np.random.randn(4, 7, 5), 1, np.random.randint(low=0, high=7, size=(4, 10, 5)))
    verify_gather(np.random.randn(4, 7, 5), 1, np.random.randint(low=0, high=7, size=(4, 10, 5)))
    verify_gather(np.random.randn(4, 7, 5), 2, np.random.randint(low=0, high=5, size=(4, 7, 2)))
    verify_gather(np.random.randn(4, 7, 5), 2, np.random.randint(low=0, high=5, size=(4, 7, 10)))
    verify_gather(np.random.randn(4, 7, 2), 0, np.random.randint(low=0, high=4, size=(4, 7, 2)))


@tvm.testing.uses_gpu
def test_gather_nd():
    for indices_dtype in ["int32", "float32", "uint8"]:
        verify_gather_nd((4,), [[1.8]], indices_dtype)
        verify_gather_nd((4,), [[1, 3, 2]], indices_dtype)
        verify_gather_nd((2, 3), [[1]], indices_dtype)
        verify_gather_nd((2, 3), [[1], [0]], indices_dtype)
        verify_gather_nd((2, 3), [[1, 0], [0, 2]], indices_dtype)
        verify_gather_nd((2, 3, 4), [[1, 0], [0, 2]], indices_dtype)
        verify_gather_nd((2, 3, 4), [[1, 0], [0, 2], [3, 1]], indices_dtype)
        verify_gather_nd(
            (2, 3, 4), [[[1, 0], [0, 1]], [[0, 2], [1, 2]], [[3, 1], [0, 2]]], indices_dtype
        )
        verify_gather_nd((2, 3, 4, 5), [[1, 0], [0, 2]], indices_dtype)
        verify_gather_nd((2, 3, 4, 5), [[1, 0], [2, 1], [3, 2], [4, 2]], indices_dtype)


@tvm.testing.uses_gpu
def test_arange():
    verify_arange(None, 20, None)
    verify_arange(None, 20, 2)
    verify_arange(1, 20, None)
    verify_arange(1, 20, 2)
    verify_arange(1, 20, 1.5)
    verify_arange(1, 20.5, None)
    verify_arange(1, 20, 3)
    verify_arange(20, 1, -1)
    verify_arange(20, 1, -1.5)


@tvm.testing.uses_gpu
def test_repeat():
    verify_repeat((2,), 1, 0)
    verify_repeat((3, 2), 2, 0)
    verify_repeat((3, 2, 4), 3, 1)
    verify_repeat((1, 3, 2, 4), 4, -1)


@tvm.testing.uses_gpu
def test_tile():
    verify_tile((3, 2), (2, 3))
    verify_tile((3, 2, 5), (2,))
    verify_tile((3,), (2, 3, 3))
    verify_tile((4, 0), (5,))


@tvm.testing.uses_gpu
def test_layout_transform():
    in_shape = (1, 32, 8, 8)
    A = te.placeholder(shape=in_shape, dtype="float32", name="A")
    B = topi.layout_transform(A, "NCHW", "NCHW16c")

    input = np.random.uniform(size=in_shape).astype(A.dtype)
    output = np.transpose(input, axes=(0, 2, 3, 1))
    output = np.reshape(output, newshape=(1, 8, 8, 2, 16))
    output = np.transpose(output, axes=(0, 3, 1, 2, 4))

    def check_device(target, dev):
        tvm_input = tvm.nd.array(input, dev)
        tvm_output = tvm.nd.empty(output.shape, device=dev, dtype=B.dtype)
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(B)
        f = tvm.build(s, [A, B], target, name="layout_transform")
        f(tvm_input, tvm_output)
        tvm.testing.assert_allclose(tvm_output.numpy(), output)

    for backend, dev in tvm.testing.enabled_targets():
        check_device(backend, dev)


@tvm.testing.uses_gpu
def test_shape():
    in_shape = (8, 7, 13)
    dtype = "int32"
    A = te.placeholder(shape=in_shape, dtype="float32", name="A")
    B = topi.shape(A, dtype)

    input = np.random.uniform(size=in_shape).astype(A.dtype)
    output = np.asarray(in_shape).astype(dtype)

    def check_device(target, dev):
        tvm_input = tvm.nd.array(input, dev)
        tvm_output = tvm.nd.empty(output.shape, device=dev, dtype=dtype)
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(B)
        f = tvm.build(s, [A, B], target, name="shape")
        f(tvm_input, tvm_output)
        tvm.testing.assert_allclose(tvm_output.numpy(), output)

    for backend, dev in tvm.testing.enabled_targets():
        check_device(backend, dev)


@tvm.testing.uses_gpu
def test_sequence_mask():
    for in_shape in (5, 10), (3, 4, 5, 4):
        for axis in [0, 1]:
            for mask_value in [0.0, 1.0]:
                max_length = in_shape[axis]
                batch_size = in_shape[1 - axis]
                A = te.placeholder(shape=in_shape, dtype="float32", name="A")
                B = te.placeholder(shape=(batch_size,), dtype="int32", name="B")
                C = topi.sequence_mask(A, B, axis=axis, mask_value=mask_value)
                A_data = np.random.normal(0, 1, in_shape).astype(np.float32)
                B_data = np.random.randint(1, max_length, (batch_size,)).astype(np.int32)
                C_gt_data = tvm.topi.testing.sequence_mask(A_data, B_data, mask_value, axis)

                def check_device(target, dev):
                    tvm_A = tvm.nd.array(A_data, dev)
                    tvm_B = tvm.nd.array(B_data, dev)
                    tvm_C = tvm.nd.empty(in_shape, device=dev, dtype="float32")
                    print("Running on target: %s" % target)
                    with tvm.target.Target(target):
                        s = tvm.topi.testing.get_injective_schedule(target)(C)
                    f = tvm.build(s, [A, B, C], target, name="SequenceMask")
                    f(tvm_A, tvm_B, tvm_C)
                    tvm.testing.assert_allclose(tvm_C.numpy(), C_gt_data)

                for backend, dev in tvm.testing.enabled_targets():
                    check_device(backend, dev)


@tvm.testing.uses_gpu
def test_ndarray_size():
    in_shape = (5, 11, 7)
    dtype = "int32"
    A = te.placeholder(shape=in_shape, dtype="float32", name="A")
    B = topi.ndarray_size(A, dtype)

    input = np.random.uniform(size=in_shape).astype(A.dtype)
    output = np.asarray(np.size(input)).astype(dtype)

    def check_device(target, dev):
        tvm_input = tvm.nd.array(input, device=dev)
        tvm_output = tvm.nd.empty((), device=dev, dtype=B.dtype)
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            s = tvm.topi.testing.get_injective_schedule(target)(B)
        f = tvm.build(s, [A, B], target, name="ndarray_size")
        f(tvm_input, tvm_output)
        tvm.testing.assert_allclose(tvm_output.numpy(), output)

    for backend, dev in tvm.testing.enabled_targets():
        check_device(backend, dev)


@tvm.testing.uses_gpu
def test_where_fusion():
    """integration test that where and zeros should be properly inlined"""

    def check_device(target, dev):
        with tvm.target.Target(target):
            print("Running on target: %s" % target)
            conv2d_compute, conv2d_schedule = tvm.topi.testing.get_conv2d_nchw_implement(target)
            data = te.placeholder((2, 1, 2, 4), "int8", "data")
            w = te.placeholder((3, 1, 2, 2), "int8", "w")
            conv1 = conv2d_compute(data, w, 1, 0, 1, "int32")
            zeros = topi.full((2, 3, 1, 3), "int32", tvm.tir.const(0, dtype="int32"))
            gt = topi.greater_equal(conv1, zeros)
            one = topi.full((2, 3, 1, 3), "int32", tvm.tir.const(1, dtype="int32"))
            two = topi.full((2, 3, 1, 3), "int32", tvm.tir.const(2, dtype="int32"))
            where = topi.where(gt, one, two)
            add = topi.add(conv1, where)
            outs = [add]
            s = conv2d_schedule(outs)
            tvm.build(s, [data, w, add], target=backend)

    for backend, dev in tvm.testing.enabled_targets():
        check_device(backend, dev)


@tvm.testing.uses_gpu
def test_one_hot():
    verify_one_hot((3,), 3, 1, 0, -1, "int32")
    verify_one_hot((3,), 3, 1.0, 0.0, -1, "float32")
    verify_one_hot((2, 2), 5, 2, -2, 0, "int32")
    verify_one_hot((2, 2), 5, 0.5, -0.5, 1, "float32")
    verify_one_hot((3, 2, 4, 5), 6, 1, 0, 1, "int32")
    verify_one_hot((3, 2, 4, 5), 6, 1.0, 0.0, 0, "float32")


@tvm.testing.uses_gpu
def test_unravel_index():
    for dtype in ["int32", "int64"]:
        for indice_dtype in ["int64", "uint8", "uint16", "uint32"]:
            verify_unravel_index([0, 1, 2, 3], [2, 2], dtype, indice_dtype)
            verify_unravel_index([144], [5, 5, 5, 2], dtype, indice_dtype)
            verify_unravel_index(144, [5, 5, 5, 2], dtype, indice_dtype)
            verify_unravel_index([100, 13, 5], [5, 5, 5, 2], dtype, indice_dtype)


@tvm.testing.uses_gpu
def test_sparse_to_dense():
    verify_sparse_to_dense(1, 3, 0, [5], [0, 3, 0, 0, 0])  # scalar
    verify_sparse_to_dense([0, 1, 4], [3, 3, 3], 0, [5], [3, 3, 0, 0, 3])  # vector
    verify_sparse_to_dense(
        [[0, 0], [1, 2]], [1, 2], 0, [3, 4], [[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]
    )  # nXd
    verify_sparse_to_dense(
        [[0, 0, 0], [1, 2, 3]],
        [1, 2],
        4,
        [2, 3, 4],
        [[[1, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]], [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 2]]],
    )  # nXd
    verify_sparse_to_dense(
        [0, 1, 4], [3.1, 3.1, 3.1], 3.5, [5], [3.1, 3.1, 3.5, 3.5, 3.1]
    )  # floats
    verify_sparse_to_dense(1, 3, None, [5], [0, 3, 0, 0, 0])  # default value not specified

    # negative test cases
    # sparse indices should be ints
    # verify_sparse_to_dense([[0.1, 1.1, 4.1], [0,2,4]], [3.1, 3.1, 3.1], 3.5, [5], [3.1, 3.1, 3.5, 3.5, 3.1])
    # sparse_values should be 0d or 1d only
    # verify_sparse_to_dense([[0, 1, 4], [0, 2, 4]], [[[3.1, 3.1, 3.1]]], 3.5, [5], [3.1, 3.1, 3.5, 3.5, 3.1])
    # sparse_indices should not be > 2d tensor
    # verify_sparse_to_dense([[[[0, 1, 4], [0, 2, 4]]]], [[[3.1, 3.1, 3.1]]], 3.5, [5], [3.1, 3.1, 3.5, 3.5, 3.1])


@tvm.testing.uses_gpu
def test_matrix_set_diag():
    for dtype in ["float32", "int32"]:
        verify_matrix_set_diag((2, 2), (2,), dtype)
        verify_matrix_set_diag((4, 3, 3), (4, 3), dtype)
        verify_matrix_set_diag((2, 3, 4), (2, 3), dtype, 1)
        verify_matrix_set_diag((2, 3, 4), (2, 4, 3), dtype, (-1, 2), "LEFT_RIGHT")
        verify_matrix_set_diag((2, 3, 4), (2, 4, 3), dtype, (-1, 2), "LEFT_LEFT")
        verify_matrix_set_diag((2, 3, 4), (2, 4, 3), dtype, (-1, 2), "RIGHT_RIGHT")


@tvm.testing.uses_gpu
def test_adv_index():
    for indice_dtype in ["int32", "int64", "uint8", "uint16", "uint32"]:
        verify_adv_index((3, 4, 5), [(2,), (2,), (1,)], indice_dtype=indice_dtype)
        verify_adv_index((10, 15, 5), [(4, 1), (1, 7)], indice_dtype=indice_dtype)
        verify_adv_index((10, 5, 15), [(1, 2, 1), (1, 2, 7)], indice_dtype=indice_dtype)


@tvm.testing.uses_gpu
def test_trilu():
    # Test upper and lower triangle
    verify_trilu((3, 3), True, 0)
    verify_trilu((3, 3), False, 0)
    # Test larger matrices with offset.
    verify_trilu((6, 6), True, 1)
    verify_trilu((6, 6), False, 2)
    verify_trilu((6, 6), False, -2)
    # Test batch size
    verify_trilu((8, 6, 6), False, -2)


if __name__ == "__main__":
    test_strided_slice()
    test_concatenate()
    test_stack()
    test_transpose()
    test_expand_dims()
    test_reshape()
    test_where()
    test_squeeze()
    test_split()
    test_flip()
    test_expand_like()
    test_take()
    test_gather_nd()
    test_arange()
    test_layout_transform()
    test_repeat()
    test_tile()
    test_shape()
    test_sequence_mask()
    test_ndarray_size()
    test_where_fusion()
    test_one_hot()
    test_unravel_index()
    test_sparse_to_dense()
    test_matrix_set_diag()
    test_adv_index()
    test_trilu()
