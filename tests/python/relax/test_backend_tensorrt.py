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

import tvm
from tvm import relax
from tvm.relax.backend.contrib.tensorrt import partition_for_tensorrt


def _make_resize2d_module(
    input_shape=(1, 3, 8, 8),
    input_dtype="float32",
    size=(16, 16),
    *,
    dynamic_size=False,
    layout="NCHW",
    method="linear",
    coordinate_transformation_mode="half_pixel",
    rounding_method="round",
    out_dtype=None,
):
    builder = relax.BlockBuilder()
    data = relax.Var("data", relax.TensorType(input_shape, input_dtype))
    params = [data]
    if dynamic_size:
        size_expr = relax.Var("size", relax.ShapeType(ndim=2))
        params.append(size_expr)
    else:
        size_expr = relax.ShapeExpr(size)

    with builder.function("main", params):
        with builder.dataflow():
            output = builder.emit(
                relax.op.image.resize2d(
                    data,
                    size=size_expr,
                    layout=layout,
                    method=method,
                    coordinate_transformation_mode=coordinate_transformation_mode,
                    rounding_method=rounding_method,
                    out_dtype=out_dtype,
                )
            )
            output = builder.emit_output(output)
        builder.emit_func_output(output)
    return builder.get()


def _functions_with_attr(mod, attr_name, attr_value):
    return [
        func
        for func in mod.functions.values()
        if isinstance(func, relax.Function)
        and func.attrs is not None
        and func.attrs.get(attr_name) == attr_value
    ]


def _collect_call_ops(expr):
    op_names = []

    def visit(node):
        if isinstance(node, relax.Call) and isinstance(node.op, tvm.ir.Op):
            op_names.append(node.op.name)

    relax.analysis.post_order_visit(expr, visit)
    return op_names


def test_resize2d_partition_supported():
    mod = _make_resize2d_module(
        size=(13, 11),
        method="nearest_neighbor",
        coordinate_transformation_mode="asymmetric",
        rounding_method="floor",
    )
    partitioned = partition_for_tensorrt(mod)

    regions = _functions_with_attr(partitioned, "Codegen", "tensorrt")
    assert len(regions) == 1
    assert "relax.image.resize2d" in _collect_call_ops(regions[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"input_shape": (1, 8, 8, 3), "layout": "NHWC"}, id="unsupported-layout"),
        pytest.param({"dynamic_size": True}, id="dynamic-size"),
        pytest.param({"input_dtype": "float64"}, id="unsupported-input-dtype"),
        pytest.param({"out_dtype": "float16"}, id="different-output-dtype"),
        pytest.param(
            {
                "method": "nearest_neighbor",
                "coordinate_transformation_mode": "tf_half_pixel_for_nn",
                "rounding_method": "floor",
            },
            id="unsupported-coordinate-mode",
        ),
        pytest.param(
            {
                "method": "nearest_neighbor",
                "coordinate_transformation_mode": "asymmetric",
                "rounding_method": "round",
            },
            id="ties-to-even-rounding",
        ),
    ],
)
def test_resize2d_partition_fallback(kwargs):
    partitioned = partition_for_tensorrt(_make_resize2d_module(**kwargs))
    assert not _functions_with_attr(partitioned, "Codegen", "tensorrt")
