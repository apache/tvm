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
""" Support level2 dynamic operator test cases.
"""

import numpy as np
import tvm
from tvm import relay
from tvm import te
from tvm.relay.testing import enabled_targets
import random
from test_dynamic_op_level3 import verify_func
import tvm.topi.testing
from tvm.relay.testing import run_infer_type

executor_kind = tvm.testing.parameter("debug", "vm")


@tvm.testing.uses_gpu
def test_dyn_upsampling_run(executor_kind):
    def verify_upsampling(dshape, scale_h, scale_w, layout, method, align_corners=False):

        if layout == "NCHW":
            (n, c, h, w) = dshape
            x_data = np.random.uniform(size=(n, c, h, w)).astype("float32")

        elif layout == "NHWC":
            (n, h, w, c) = dshape
            x_data = np.random.uniform(size=(n, h, w, c)).astype("float32")

        ref_res = tvm.topi.testing.resize2d_python(
            x_data,
            (scale_h, scale_w),
            layout,
            method[2:] if method[0:2] == "bi" else method,
            "align_corners" if align_corners else "asymmetric",
        )
        x = relay.Var("x", relay.TensorType(dshape, "float32"))
        scale_h_var = relay.var("scale_h", relay.TensorType((), "float32"))
        scale_w_var = relay.var("scale_h", relay.TensorType((), "float32"))

        z = relay.nn.upsampling(
            x, scale_h_var, scale_w_var, method=method, layout=layout, align_corners=align_corners
        )
        zz = run_infer_type(z)
        func = relay.Function([x, scale_h_var, scale_w_var], z)

        for target, dev in tvm.testing.enabled_targets():
            mod = tvm.ir.IRModule.from_expr(func)
            op_res = relay.create_executor(
                executor_kind, mod=mod, device=dev, target=target
            ).evaluate()(
                x_data, np.array(scale_h).astype("float32"), np.array(scale_w).astype("float32")
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-4, atol=1e-6)

    verify_upsampling((1, 16, 32, 32), 3, 2.0, "NCHW", "nearest_neighbor")
    verify_upsampling((1, 16, 32, 32), 5, 2.0, "NCHW", "bilinear", True)
    verify_upsampling((1, 16, 32, 32), 2.0, 6, "NHWC", "nearest_neighbor")
    verify_upsampling((1, 16, 32, 32), 2.0, 2.0, "NHWC", "bilinear", True)


# tests upsampling type inference with scale_h passed in as a constant and scale_w as a variable
@tvm.testing.uses_gpu
def test_dyn_upsampling_infer_type_const():
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")

    data = relay.var("data", relay.TensorType((n, c, h, w), "int8"))
    scale_w = relay.Var("scale_w", relay.TensorType((), "float32"))

    z = relay.nn.upsampling(data, 2.0, scale_w)
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((n, c, relay.Any(), relay.Any()), "int8")


@tvm.testing.uses_gpu
def test_dyn_upsampling3d_run(executor_kind):
    def verify_upsampling3d(
        dshape, scale_d, scale_h, scale_w, layout, method, coord_trans="asymmetric"
    ):

        if layout == "NCDHW":
            (n, c, d, h, w) = dshape
            x_data = np.random.uniform(size=(n, c, d, h, w)).astype("float32")

        elif layout == "NDHWC":
            (n, d, h, w, c) = dshape
            x_data = np.random.uniform(size=(n, d, h, w, c)).astype("float32")

        ref_res = tvm.topi.testing.resize3d_python(
            x_data,
            (scale_d, scale_h, scale_w),
            layout,
            method[3:] if method[0:3] == "tri" else method,
            coord_trans,
        )

        x = relay.Var("x", relay.TensorType(dshape, "float32"))
        scale_d_var = relay.var("scale_d", relay.TensorType((), "float32"))
        scale_h_var = relay.var("scale_h", relay.TensorType((), "float32"))
        scale_w_var = relay.var("scale_h", relay.TensorType((), "float32"))

        z = relay.nn.upsampling3d(
            x,
            scale_d_var,
            scale_h_var,
            scale_w_var,
            method=method,
            layout=layout,
            coordinate_transformation_mode=coord_trans,
        )
        zz = run_infer_type(z)
        func = relay.Function([x, scale_d_var, scale_h_var, scale_w_var], z)

        for target, dev in enabled_targets():
            mod = tvm.ir.IRModule.from_expr(func)
            op_res = relay.create_executor(
                executor_kind, mod=mod, device=dev, target=target
            ).evaluate()(
                x_data,
                np.array(scale_d).astype("float32"),
                np.array(scale_h).astype("float32"),
                np.array(scale_w).astype("float32"),
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-4, atol=1e-6)

    verify_upsampling3d((1, 1, 1, 1, 1), 2, 3, 4, "NCDHW", "nearest_neighbor")
    verify_upsampling3d((1, 8, 16, 16, 16), 2.0, 3.0, 4.0, "NCDHW", "nearest_neighbor")
    verify_upsampling3d((1, 8, 16, 16, 16), 2.0, 5.0, 1.0, "NCDHW", "trilinear", "align_corners")
    verify_upsampling3d((1, 20, 3, 4, 16), 2.0, 2.0, 2.0, "NDHWC", "nearest_neighbor")
    verify_upsampling3d((1, 8, 4, 16, 15), 2.0, 2.0, 2.0, "NDHWC", "trilinear", "align_corners")


# tests upsampling type inference with scale_h passed in as a constant and scale_w as a variable
def test_dyn_upsampling3d_infer_type_const():
    n, c, d, h, w = (
        te.size_var("n"),
        te.size_var("c"),
        te.size_var("d"),
        te.size_var("h"),
        te.size_var("w"),
    )

    data = relay.var("data", relay.TensorType((n, c, d, h, w), "int8"))
    scale_d = relay.Var("scale_h", relay.TensorType((), "float32"))
    scale_w = relay.Var("scale_w", relay.TensorType((), "float32"))

    z = relay.nn.upsampling3d(data, scale_d, 2.0, scale_w, layout="NCDHW", method="trilinear")
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType(
        (n, c, relay.Any(), relay.Any(), relay.Any()), "int8"
    )


@tvm.testing.uses_gpu
def test_dyn_pad(executor_kind):
    def verify_pad(dshape, pad_width, pad_val, dtype):
        x = relay.var("x", relay.TensorType(dshape, dtype))
        ndim = len(dshape)
        pad_width_var = relay.var("pad_width_var", relay.TensorType((ndim, 2), "int64"))
        pad_val_var = relay.var("pad_val_var", relay.TensorType((), dtype))
        y = relay.nn.pad(x, pad_width_var, pad_val_var)
        yy = run_infer_type(y)

        assert yy.checked_type == relay.ty.TensorType((relay.Any(),) * ndim, dtype)
        func = relay.Function([x, pad_width_var, pad_val_var], y)
        data = np.random.uniform(size=dshape).astype(dtype)
        ref_res = np.pad(data, pad_width, "constant", constant_values=(((pad_val,) * 2),) * ndim)
        pad_width = np.array(pad_width).astype("int64")

        verify_func(
            executor_kind, func, [data, pad_width, np.array(pad_val).astype(dtype)], ref_res
        )

    def verify_pad_default_fill(dshape, pad_width, dtype):
        x = relay.var("x", relay.TensorType(dshape, dtype))
        ndim = len(dshape)
        pad_width_var = relay.var("pad_width_var", relay.TensorType((ndim, 2), "int64"))
        y = relay.nn.pad(x, pad_width_var)
        yy = run_infer_type(y)

        assert yy.checked_type == relay.ty.TensorType((relay.Any(),) * ndim, dtype)
        func = relay.Function([x, pad_width_var], y)
        data = np.random.uniform(size=dshape).astype(dtype)
        ref_res = np.pad(data, pad_width)
        pad_width = np.array(pad_width).astype("int64")

        verify_func(executor_kind, func, [data, pad_width], ref_res)

    verify_pad((4, 10, 7, 7), ((1, 1), (2, 2), (3, 3), (4, 4)), 2.0, "int32")
    verify_pad((2, 7), ((1, 4), (2, 2)), 4.0, "float64")
    verify_pad_default_fill((4, 10, 7, 7), ((1, 1), (2, 2), (3, 3), (4, 4)), "float64")
    verify_pad_default_fill((2, 7), ((1, 4), (2, 2)), "int32")


if __name__ == "__main__":
    tvm.testing.main()
