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
from unittest.mock import MagicMock

import tvm
from tvm import relay
from tvm.relay import testing
from tvm.relay.expr_functor import ExprMutator
from tvm.ir.instrument import pass_instrument
from tvm.driver.tvmc.transform import apply_graph_transforms
from tvm.driver.tvmc.model import TVMCException


def test_layout_transform_fold_constant(relay_conv2d):
    """
    Test layout is correctly transformed and constant folding is applied.
    """
    desired_layout = "NHWC"

    @pass_instrument
    class CollectPassNames:
        def __init__(self):
            self.names = []

        def run_after_pass(self, _, info):
            self.names.append(info.name)

    pass_names = CollectPassNames()
    with tvm.transform.PassContext(opt_level=3, instruments=[pass_names]):
        apply_graph_transforms(relay_conv2d, {"desired_layout": [desired_layout]})

    names = pass_names.names
    assert "ConvertLayout" in names
    assert "FoldConstant" in names
    assert names.index("ConvertLayout") < names.index("FoldConstant")


def test_layout_transform_convert_layout_pass_args(relay_conv2d, monkeypatch):
    """
    Check the convert layout desired layouts arugment is what is expected when
    a desired layout is provided.
    """
    desired_layout = "NHWC"

    mock_convert_layout = MagicMock()
    mock_convert_layout.return_value = relay.transform.ConvertLayout({})
    monkeypatch.setattr(relay.transform, "ConvertLayout", mock_convert_layout)

    with tvm.transform.PassContext(opt_level=3):
        apply_graph_transforms(relay_conv2d, {"desired_layout": [desired_layout]})

    mock_convert_layout.assert_called_once_with(
        {
            "nn.conv2d": ["NHWC", "default"],
            "nn.conv2d_transpose": ["NHWC", "default"],
            "qnn.conv2d": ["NHWC", "default"],
        }
    )


def test_layout_transform_convert_kernel_layout_pass_args(relay_conv2d, monkeypatch):
    """
    Check the convert layout desired layouts arugment is what is expected when
    a non-default kernel layout is provided.
    """
    desired_layout = "NHWC:HWIO"
    desired_layout_ops = ["nn.conv2d"]

    mock_convert_layout = MagicMock()
    mock_convert_layout.return_value = relay.transform.ConvertLayout({})
    monkeypatch.setattr(relay.transform, "ConvertLayout", mock_convert_layout)

    with tvm.transform.PassContext(opt_level=3):
        apply_graph_transforms(
            relay_conv2d,
            {"desired_layout": [desired_layout], "desired_layout_ops": desired_layout_ops},
        )

    mock_convert_layout.assert_called_once_with(
        {
            "nn.conv2d": ["NHWC", "HWIO"],
        }
    )


def test_layout_transform_convert_layout_pass_args_multiple(relay_conv2d, monkeypatch):
    """
    Check the convert layout desired layouts arugment is what is expected when
    a multiple desired layouts are provided.
    """
    desired_layout = ["NHWC", "NCHW"]
    desired_layout_ops = ["nn.max_pool2d", "qnn.conv2d"]

    mock_convert_layout = MagicMock()
    mock_convert_layout.return_value = relay.transform.ConvertLayout({})
    monkeypatch.setattr(relay.transform, "ConvertLayout", mock_convert_layout)

    with tvm.transform.PassContext(opt_level=3):
        apply_graph_transforms(
            relay_conv2d,
            {"desired_layout": desired_layout, "desired_layout_ops": desired_layout_ops},
        )

    mock_convert_layout.assert_called_once_with(
        {
            "nn.max_pool2d": ["NHWC", "default"],
            "qnn.conv2d": ["NCHW", "default"],
        }
    )


@pytest.mark.parametrize(
    "desired",
    [
        (["NHWC", "NCHW"], ["nn.max_pool2d"]),
        (["NHWC", "NCHW"], None),
    ],
)
def test_layout_transform_convert_layout_pass_args_multiple_invalid(
    relay_conv2d,
    monkeypatch,
    desired,
):
    """
    Check invalid cases when passing multiple values to the desired layouts argument.
    """
    desired_layout, desired_layout_ops = desired

    mock_convert_layout = MagicMock()
    mock_convert_layout.return_value = relay.transform.ConvertLayout({})
    monkeypatch.setattr(relay.transform, "ConvertLayout", mock_convert_layout)

    with pytest.raises(TVMCException):
        with tvm.transform.PassContext(opt_level=3):
            apply_graph_transforms(
                relay_conv2d,
                {"desired_layout": desired_layout, "desired_layout_ops": desired_layout_ops},
            )


def test_layout_transform_to_mixed_precision_pass_args_mock(relay_conv2d, monkeypatch):
    """
    Check the mixed precision arugments which are expected when
    mixed precision arguments are provided.
    """
    mock_mixed_precision = MagicMock()
    mock_mixed_precision.return_value = tvm.driver.tvmc.transform.MixedPrecision([], "")
    monkeypatch.setattr(tvm.driver.tvmc.transform, "MixedPrecision", mock_mixed_precision)

    with tvm.transform.PassContext(opt_level=3):
        apply_graph_transforms(
            relay_conv2d,
            {
                "mixed_precision": True,
                "mixed_precision_ops": ["nn.conv2d"],
                "mixed_precision_calculation_type": "float16",
                "mixed_precision_acc_type": "float16",
            },
        )
        mock_mixed_precision.assert_called_with(["nn.conv2d"], "float16")

        apply_graph_transforms(
            relay_conv2d,
            {
                "mixed_precision": True,
                "mixed_precision_ops": ["nn.conv2d", "nn.dense"],
                "mixed_precision_calculation_type": "float16",
                "mixed_precision_acc_type": "float32",
            },
        )
        mock_mixed_precision.assert_called_with(["nn.conv2d", "nn.dense"], "float32")


def test_layout_transform_to_mixed_precision_pass_args_graph():
    """
    Check the mixed precision arugments application with in a graph.
    """

    mod, params = testing.mobilenet.get_workload(batch_size=1, dtype="float32")

    class CheckOpMutator(ExprMutator):
        """Inspect Ops According to expected types."""

        def __init__(self, calculation_type, acc_type, op):
            self.calculation_type = calculation_type
            self.acc_type = acc_type
            self.op = op
            self.is_expected = True
            super().__init__()

        def visit_call(self, call):
            visit = super().visit(call.args[0])
            if call.op == relay.op.get(self.op):
                if self.is_expected:
                    self.is_expected = (
                        call.checked_type.dtype == self.acc_type
                        or call.args[0].checked_type.dtype == self.calculation_type
                    )
            return call

        def check(self, func):
            self.visit(func)
            return self.is_expected

    mod = apply_graph_transforms(
        mod,
        {
            "mixed_precision": True,
            "mixed_precision_ops": ["nn.conv2d", "nn.dense"],
            "mixed_precision_calculation_type": "float16",
            "mixed_precision_acc_type": "float16",
        },
    )
    ret = CheckOpMutator("float16", "float16", "nn.conv2d").check(mod["main"])
    assert ret
    ret = CheckOpMutator("float16", "float16", "nn.dense").check(mod["main"])
    assert ret

    mod = apply_graph_transforms(
        mod,
        {
            "mixed_precision": True,
            "mixed_precision_ops": ["nn.conv2d", "nn.dense"],
            "mixed_precision_calculation_type": "float16",
            "mixed_precision_acc_type": "float32",
        },
    )
    ret = CheckOpMutator("float16", "float32", "nn.conv2d").check(mod["main"])
    assert ret
    ret = CheckOpMutator("float16", "float32", "nn.dense").check(mod["main"])
    assert ret


if __name__ == "__main__":
    tvm.testing.main()
