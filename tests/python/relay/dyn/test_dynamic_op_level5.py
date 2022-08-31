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
""" Support level5 operator test cases.
"""
import math
import numpy as np
import tvm
from tvm import te
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_infer_type
import tvm.topi.testing
import tvm.testing

executor_kind = tvm.testing.parameter("debug", "vm")


def test_resize2d_infer_type():
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "int8"))
    size = relay.var("size", relay.TensorType((2,), "int8"))
    z = relay.image.resize2d(x, size)
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((n, c, relay.Any(), relay.Any()), "int8")


@tvm.testing.uses_gpu
def test_resize2d(executor_kind):
    def verify_resize2d(dshape, scale, method, layout):
        if layout == "NHWC":
            size = (dshape[1] * scale, dshape[2] * scale)
        else:
            size = (dshape[2] * scale, dshape[3] * scale)
        size = np.array(size).astype("int64")
        x_data = np.random.uniform(size=dshape).astype("float32")

        x = relay.var("x", relay.TensorType(dshape, "float32"))
        size_var = relay.var("size", relay.TensorType((2,), "int64"))

        coord_trans = "asymmetric" if method == "nearest_neighbor" else "align_corners"
        z = relay.image.resize2d(
            x, size_var, None, layout, method, coordinate_transformation_mode=coord_trans
        )

        zz = run_infer_type(z)
        func = relay.Function([x, size_var], z)

        ref_res = tvm.topi.testing.resize2d_python(
            x_data, (scale, scale), layout, method, coord_trans
        )

        for target, dev in tvm.testing.enabled_targets():
            mod = tvm.ir.IRModule.from_expr(func)
            op_res = relay.create_executor(
                executor_kind, mod=mod, device=dev, target=target
            ).evaluate()(x_data, size)
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-4, atol=1e-6)

    for method in ["linear", "nearest_neighbor"]:
        for layout in ["NCHW", "NHWC"]:
            verify_resize2d((1, 4, 4, 4), 2, method, layout)
            verify_resize2d((2, 8, 17, 20), 7, method, layout)


if __name__ == "__main__":
    test_resize2d_infer_type()
    test_resize2d()
