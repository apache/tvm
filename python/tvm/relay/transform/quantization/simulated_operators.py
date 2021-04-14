from typing import Callable, List, NamedTuple, Optional, Tuple

import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    _DFPatternCallback,
    is_constant,
    is_op,
    wildcard,
)
from tvm.relay.transform.quantization import calibration_rewrite_utils


def conv2d_pad_input(
    input_tensor: tvm.relay.Expr, zero_point: tvm.relay.Constant, attrs: "relay.attrs.Conv2dAttrs"
) -> tvm.relay.Expr:
    pad_top, pad_left, pad_bottom, pad_right = attrs.padding

    if pad_top == pad_left == pad_bottom == pad_right == 0:
        relay.
        pass 




"""
  // 1) Pad the input data
  auto padded_data = data;
  auto pad_top_value = get_const_int(param->padding[0]);
  auto pad_left_value = get_const_int(param->padding[1]);
  auto pad_bottom_value = get_const_int(param->padding[2]);
  auto pad_right_value = get_const_int(param->padding[3]);
  bool do_pad =
      pad_top_value != 0 || pad_left_value != 0 || pad_bottom_value != 0 || pad_right_value != 0;
  if (do_pad) {
    Array<IndexExpr> pad_n({0, 0});
    Array<IndexExpr> pad_c({0, 0});
    Array<IndexExpr> pad_h({param->padding[0], param->padding[2]});
    Array<IndexExpr> pad_w({param->padding[1], param->padding[3]});

    Array<Array<IndexExpr>> pad_width;
    if (param->data_layout == "NCHW") {
      pad_width = {pad_n, pad_c, pad_h, pad_w};
    } else if (param->data_layout == "NHWC") {
      pad_width = {pad_n, pad_h, pad_w, pad_c};
    } else {
      LOG(FATAL) << "qnn.conv2d does not support " << param->data_layout << " layout";
    }
    auto pad_value = GetScalarFromConstant<int>(input_zero_point);
    padded_data = Pad(data, pad_width, pad_value, "constant");
  }
  return padded_data;
"""


def get_simulated_conv2d(
    input_tensor: tvm.relay.Expr, weight_tensor: tvm.relay.Expr
) -> tvm.relay.Expr:
    pass
