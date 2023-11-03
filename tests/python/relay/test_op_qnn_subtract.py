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
import numpy as np
from tvm import relay


def qnn_subtract_driver(x_datas, y_datas, golden_outputs, scale_and_zp, data_dtype="uint8"):
    # all x, y and golden outputs should be of the same length
    assert len(x_datas) == len(y_datas)
    assert len(y_datas) == len(golden_outputs)

    x = relay.var("x", shape=(1, 4), dtype=data_dtype)
    y = relay.var("y", shape=(1, 4), dtype=data_dtype)
    lhs_scale = relay.const(scale_and_zp["lhs_scale"], "float32")
    lhs_zp = relay.const(scale_and_zp["lhs_zp"], "int32")
    rhs_scale = relay.const(scale_and_zp["rhs_scale"], "float32")
    rhs_zp = relay.const(scale_and_zp["rhs_zp"], "int32")
    output_scale = relay.const(scale_and_zp["output_scale"], "float32")
    output_zp = relay.const(scale_and_zp["output_zp"], "int32")
    z = relay.qnn.subtract(
        lhs=x,
        rhs=y,
        lhs_scale=lhs_scale,
        lhs_zero_point=lhs_zp,
        rhs_scale=rhs_scale,
        rhs_zero_point=rhs_zp,
        output_scale=output_scale,
        output_zero_point=output_zp,
    )
    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]
    for i in range(0, len(x_datas)):
        x_data = x_datas[i]
        y_data = y_datas[i]
        golden_output = golden_outputs[i]
        op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
            x_data, y_data
        )
        np.testing.assert_equal(op_res.numpy(), golden_output)


def test_tflite_same_io_qnn_params():
    scale_and_zp = {
        "lhs_scale": 0.00784314,
        "lhs_zp": 127,
        "rhs_scale": 0.00784314,
        "rhs_zp": 127,
        "output_scale": 0.00784314,
        "output_zp": 127,
    }
    x_datas = [
        np.array((140, 153, 165, 178)).reshape((1, 4)),
        np.array((25, 153, 178, 216)).reshape((1, 4)),
        np.array((25, 153, 216, 165)).reshape((1, 4)),
    ]
    y_datas = [
        np.array((204, 178, 165, 140)).reshape((1, 4)),
        np.array((204, 178, 191, 25)).reshape((1, 4)),
        np.array((204, 178, 25, 191)).reshape((1, 4)),
    ]
    golden_outputs = [
        np.array((63, 102, 127, 165)).reshape((1, 4)),
        np.array((0, 102, 114, 255)).reshape((1, 4)),
        np.array((0, 102, 255, 101)).reshape((1, 4)),
    ]
    qnn_subtract_driver(x_datas, y_datas, golden_outputs, scale_and_zp)


def test_tflite_different_io_qnn_params():
    scale_and_zp = {
        "lhs_scale": 0.0156863,
        "lhs_zp": 127,
        "rhs_scale": 0.0117647,
        "rhs_zp": 85,
        "output_scale": 0.0235294,
        "output_zp": 128,
    }
    x_datas = [
        np.array((76, 140, 153, 172)).reshape((1, 4)),
        np.array((133, 140, 146, 153)).reshape((1, 4)),
        np.array((76, 140, 172, 146)).reshape((1, 4)),
    ]
    y_datas = [
        np.array((136, 119, 128, 17)).reshape((1, 4)),
        np.array((136, 119, 111, 94)).reshape((1, 4)),
        np.array((136, 119, 17, 128)).reshape((1, 4)),
    ]
    golden_outputs = [
        np.array((68, 120, 123, 192)).reshape((1, 4)),
        np.array((106, 120, 128, 140)).reshape((1, 4)),
        np.array((68, 120, 192, 119)).reshape((1, 4)),
    ]
    qnn_subtract_driver(x_datas, y_datas, golden_outputs, scale_and_zp)


def test_saturation():
    # Same params
    scale_and_zp = {
        "lhs_scale": 0.125,
        "lhs_zp": 0,
        "rhs_scale": 0.125,
        "rhs_zp": 0,
        "output_scale": 0.125,
        "output_zp": 0,
    }
    x_data = [np.array((255, 1, 1, 0)).reshape((1, 4))]
    y_data = [np.array((255, 255, 128, 0)).reshape((1, 4))]
    golden_output = [np.array((0, 0, 0, 0)).reshape((1, 4))]
    qnn_subtract_driver(x_data, y_data, golden_output, scale_and_zp)

    # Same params, different scale
    scale_and_zp = {
        "lhs_scale": 0.125,
        "lhs_zp": 0,
        "rhs_scale": 0.125,
        "rhs_zp": 0,
        "output_scale": 0.25,
        "output_zp": 0,
    }
    x_data = [np.array((255, 1, 200, 0)).reshape((1, 4))]
    y_data = [np.array((255, 255, 127, 0)).reshape((1, 4))]
    golden_output = [np.array((0, 0, 36, 0)).reshape((1, 4))]
    qnn_subtract_driver(x_data, y_data, golden_output, scale_and_zp)

    # All params different
    scale_and_zp = {
        "lhs_scale": 0.5,
        "lhs_zp": 0,
        "rhs_scale": 0.25,
        "rhs_zp": 0,
        "output_scale": 0.125,
        "output_zp": 0,
    }
    x_data = [np.array((255, 0, 1, 0)).reshape((1, 4))]
    y_data = [np.array((0, 128, 64, 0)).reshape((1, 4))]
    golden_output = [np.array((255, 0, 0, 0)).reshape((1, 4))]
    qnn_subtract_driver(x_data, y_data, golden_output, scale_and_zp)


if __name__ == "__main__":
    test_tflite_same_io_qnn_params()
    test_tflite_different_io_qnn_params()
    test_saturation()
