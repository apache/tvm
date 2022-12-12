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
# pylint: disable=invalid-name, too-many-locals, too-many-arguments
"""Example code to do convolution."""

import numpy as np
import tvm
import os
import tvm.testing
import tvm.topi.testing
from tvm import te, autotvm, topi, relay
from tvm.contrib.pickle_memoize import memoize
from tvm.contrib import nvcc
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple

_conv2d_hwnc_tensorcore_implement = {
    "cuda": (topi.cuda.conv2d_hwnc_tensorcore, topi.cuda.schedule_conv2d_hwnc_tensorcore)
}


def verify_conv2d_hwnc(
    batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation=1, dtype="int4"
):
    """Test the conv2d with tensorcore for hwnc layout"""
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel, kernel))
    padding_sum = pad_top + pad_left + pad_bottom + pad_right
    print(
        "Workload: (%d, %d, %d, %d, %d, %d, %d, %d)"
        % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation)
    )
    # choose dtype from int4, int8
    assert dtype in ["int4", "int8"]

    in_height = in_width = in_size

    A = te.placeholder((in_height, in_width, batch, in_channel), name="A", dtype=dtype)
    W = te.placeholder((kernel, kernel, num_filter, in_channel), name="W", dtype=dtype)

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)

    @memoize("topi.tests.test_topi_conv2d_hwnc.verify_conv2d_hwnc")
    def get_ref_data():
        if dtype == "int4":
            a_np = np.random.randint(low=-8, high=7, size=a_shape).transpose((2, 0, 1, 3))
            w_np = np.random.randint(low=-8, high=7, size=w_shape)
            dw_np = topi.testing.dilate_python(
                w_np.transpose((0, 1, 3, 2)), (1, 1, dilation, dilation)
            )
        elif dtype == "int8":
            a_np = (
                np.random.randint(low=-128, high=127, size=a_shape)
                .transpose((2, 0, 1, 3))
                .astype(dtype)
            )
            w_np = np.random.randint(low=-128, high=127, size=w_shape).astype(dtype)
            dw_np = topi.testing.dilate_python(
                w_np.transpose((0, 1, 3, 2)), (1, 1, dilation, dilation)
            )

        c_np = topi.testing.conv2d_nhwc_python(a_np, dw_np, stride, padding)
        return a_np, w_np, c_np

    def convert_int32_into_int4(a_int32):
        """convert int32 values into int4
        Parameters
        ----------
        a_int32 : int

        Return
        ------
        a_int4 : int
        """
        I, J, K, L = a_int32.shape
        a_int4 = np.zeros(shape=(I, J, K, L // 8), dtype=np.int32)
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    for l in range(L // 8):
                        for m in range(min(8, L - l * 8)):
                            a_int4[i, j, k, l] = a_int4[i, j, k, l] | (
                                (a_int32[i, j, k, l * 8 + m] & 0xF) << ((7 - m) * 4)
                            )
        return a_int4

    a_np, w_np, c_np = get_ref_data()
    if dtype == "int4":
        a_np = convert_int32_into_int4(a_np)
        w_np = convert_int32_into_int4(w_np)

    def check_target(target):
        dev = tvm.device(target, 0)
        if not tvm.testing.device_enabled(target):
            print("Skip because %s is not enabled" % target)
            return
        if not nvcc.have_tensorcore(dev.compute_version):
            print("skip because gpu does not support Tensor Cores")
            return
        print("Running on target: %s" % target)
        with tvm.target.Target(target):
            fcompute, fschedule = topi.testing.dispatch(target, _conv2d_hwnc_tensorcore_implement)
            C = fcompute(A, W, stride, padding, dilation, dtype, "int32")
            s = fschedule([C])

        a = tvm.nd.array(a_np.transpose((1, 2, 0, 3)), dev)
        w = tvm.nd.array(w_np, dev)
        c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=C.dtype), dev)

        func = tvm.build(
            s,
            [A, W, C],
            target,
            name="relu_%d_%d_%d_%d_%d_%d_%d_%d"
            % (batch, in_channel, in_size, num_filter, kernel, stride, padding_sum, dilation),
        )
        func(a, w, c)

        rtol = 1e-3
        tvm.testing.assert_allclose(c.numpy().transpose((2, 0, 1, 3)), c_np, rtol=rtol)

    check_target("cuda")


def verify_feature_length():
    np.random.seed(123)
    target = "cuda"
    ctx = tvm.device(target)

    batch_size = 32

    input_shape = (32, 512, 7, 7)
    kernel_shape = (512, 512, 3, 3)

    def get_mod():
        x = relay.var("x", relay.TensorType(input_shape, "float32"))
        y = relay.var("y", relay.TensorType(kernel_shape, "float32"))
        f = relay.Function(
            [x, y], relay.nn.conv2d(x, y, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3])
        )
        mod = tvm.IRModule()
        mod["main"] = f
        mod = relay.transform.InferType()(mod)
        return mod, {}

    mod, params = get_mod()
    layout_config = relay.transform.LayoutConfig()
    desired_layouts = {"nn.conv2d": ["HWNC", "default"]}
    with layout_config:
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
    mod = relay.transform.recast(mod, "int4", "int32")

    tasks = autotvm.task.extract_from_program(
        mod, target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    assert len(tasks) == 1
    task = tasks[0]

    space = task.config_space

    idx1 = space.get_rand_index()
    idx2 = space.get_rand_index()

    cfg = space.get(idx1)
    sch, arg_bufs = task.instantiate(cfg)
    fea1 = autotvm.feature.get_itervar_feature_flatten(sch, arg_bufs, take_log=True)

    cfg = space.get(idx2)
    sch, arg_bufs = task.instantiate(cfg)
    fea2 = autotvm.feature.get_itervar_feature_flatten(sch, arg_bufs, take_log=True)

    assert len(fea1) == len(fea2)


@tvm.testing.requires_tensorcore
def test_conv2d_hwnc_tensorcore():
    """Test the conv2d with tensorcore for hwnc layout"""
    verify_conv2d_hwnc(8, 64, 56, 64, 3, 1, 1, dtype="int8")
    verify_conv2d_hwnc(8, 64, 56, 64, 1, 1, 0, dtype="int4")
    verify_conv2d_hwnc(8, 64, 56, 128, 3, 2, 1)
    verify_conv2d_hwnc(8, 64, 56, 64, 1, 2, 0)
    verify_conv2d_hwnc(8, 128, 28, 128, 3, 1, 1)
    verify_conv2d_hwnc(8, 128, 28, 256, 3, 2, 1)
    verify_conv2d_hwnc(8, 128, 28, 256, 1, 2, 0)
    verify_conv2d_hwnc(8, 256, 14, 256, 3, 1, 1)
    verify_conv2d_hwnc(8, 256, 14, 512, 3, 2, 1)
    verify_conv2d_hwnc(8, 256, 14, 512, 1, 2, 0)
    verify_conv2d_hwnc(8, 512, 9, 512, 3, 1, 1)
    verify_feature_length()


if __name__ == "__main__":
    test_conv2d_hwnc_tensorcore()
