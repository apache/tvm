/*!
 *  Copyright (c) 2018 by Contributors.
 *
 * \file tvm/relay/pass/pattern_util.h
 * \brief Header of internal operator functions
 *  These can be used for writing passes.
 */
#ifndef TVM_RELAY_PASS_PATTERN_UTIL_H_
#define TVM_RELAY_PASS_PATTERN_UTIL_H_

#include <tvm/relay/op.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/attrs/transform.h>
#include "../op/nn/layout.h"

namespace tvm {
namespace relay {

/*!
 * \brief Try to match lhs and rhs via broadcasting rule, such that:
 *
 * rhs matches the dimension of lhs specified by lhs_axes
 * rhs's value equals 1 on rest of dimensions.
 *
 * \param tlhs The type of left operand (data)
 * \param trhs The type right operand (bias)
 * \param lhs_axes The axes on lhs to match.
 * \param rhs_value A squeezed version of rhs which only contains matched dimension.
 * \return Whether match is successful.
 */
inline bool MatchBroadcastToLeftAxes(const TensorTypeNode* tlhs,
                                     const TensorTypeNode* trhs,
                                     const Array<Integer>& lhs_axes,
                                     Expr* rhs_value = nullptr) {
  if (tlhs->shape.size() < trhs->shape.size()) return false;
  AttrsEqual equal;
  size_t base = tlhs->shape.size() - trhs->shape.size();
  size_t j = 0;

  NodePtr<SqueezeAttrs> squeeze_attrs;
  if (rhs_value != nullptr) {
    squeeze_attrs = make_node<SqueezeAttrs>();
  }

  for (size_t i = 0; i < tlhs->shape.size(); ++i) {
    if (j < lhs_axes.size() && i == static_cast<size_t>(lhs_axes[j]->value)) {
      if (i < base || !equal(tlhs->shape[i], trhs->shape[i - base])) {
        return false;
      }
      ++j;
    } else if (i >= base) {
      if (!is_const_int(trhs->shape[i - base], 1)) {
        return false;
      }
      if (rhs_value != nullptr) {
        squeeze_attrs->axis.push_back(static_cast<int>(i - base));
      }
    }
  }
  if (rhs_value != nullptr && squeeze_attrs->axis.size() != 0) {
    static const Op& squeeze_op = Op::Get("squeeze");
    *rhs_value = CallNode::make(squeeze_op, {rhs_value[0]}, Attrs(squeeze_attrs), {});
  }
  return true;
}

/*!
 * \brief Expand 1D Tensor to match axis.
 *
 * The result bias can be used to add or multiply to
 * the target Tensor on the specified axis via broadcasting rule.
 *
 * \param bias The bias.
 * \param target_ndim target dimension.
 * \param axes The axis on the output we want to match on.
 */
inline Expr ExpandBiasToMatchAxis(Expr bias,
                                  int target_ndim,
                                  const Array<Integer>& axes) {
  static const Op& expand_dims = Op::Get("expand_dims");
  for (size_t i = axes.size(); i != 0; --i) {
    if (i == axes.size()) {
      int64_t num_pad_axis = target_ndim - axes[i - 1]->value - 1;
      if (num_pad_axis > 0) {
        auto attrs = make_node<ExpandDimsAttrs>();
        attrs->axis = i;
        attrs->num_newaxis = static_cast<int>(num_pad_axis);
        bias = CallNode::make(expand_dims, {bias}, Attrs(attrs), {});
      }
    } else {
      int64_t diff = axes[i]->value - axes[i - 1]->value;
      CHECK_GE(diff, 0L);
      if (diff > 0) {
        auto attrs = make_node<ExpandDimsAttrs>();
        attrs->axis = i;
        attrs->num_newaxis = static_cast<int>(diff);
        bias = CallNode::make(expand_dims, {bias}, Attrs(attrs), {});
      }
    }
  }
  return bias;
}

/*!
 * \brief Check if the call is depthwise conv2d.
 *
 * \param call The conv2d call.
 * \param param The conv2d attributes.
 * \return Whether it is depthwise_conv2d.
 */
inline bool IsDepthwiseConv2D(const Call& call,
                              const Conv2DAttrs* param,
                              const Layout& weight_layout) {
  static const Layout kOIHW("OIHW");
  auto wshape = ConvertLayout(
      call->args[1]->type_as<TensorTypeNode>()->shape,
      weight_layout, kOIHW);
  return is_const_int(wshape[0], param->groups) &&
      is_const_int(wshape[1], 1);
}


/*!
 * \brief Create a Constant with a scalar
 *
 * \param dtype The data type.
 * \param value The value of the scalar.
 * \return A Constant.
 */
template<typename T>
inline Constant MakeConstantScalar(DataType dtype, T value) {
  CHECK_EQ(sizeof(T) * 8, dtype.bits()) << "data type mismatch";
  runtime::NDArray arr = runtime::NDArray::Empty({}, Type2TVMType(dtype), {kDLCPU, 0});
  *static_cast<T*>(arr->data) = value;
  return ConstantNode::make(arr);
}


inline Expr Negative(Expr x) {
  static const Op& op = Op::Get("negative");
  return CallNode::make(op, {x}, Attrs(), {});
}


inline Expr Sqrt(Expr x) {
  static const Op& op = Op::Get("sqrt");
  return CallNode::make(op, {x}, Attrs(), {});
}


inline Expr Add(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("add");
  return CallNode::make(op, {lhs, rhs}, Attrs(), {});
}


inline Expr Multiply(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("multiply");
  return CallNode::make(op, {lhs, rhs}, Attrs(), {});
}


inline Expr Divide(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("divide");
  return CallNode::make(op, {lhs, rhs}, Attrs(), {});
}


inline Expr ReshapeLike(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("reshape_like");
  return CallNode::make(op, {lhs, rhs}, Attrs(), {});
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_PATTERN_UTIL_H_
