/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2018 by Contributors.
 *
 * \file tvm/relay/pass/pattern_util.h
 * \brief Header of internal operator functions
 *  These can be used for writing passes.
 */
#ifndef TVM_RELAY_PASS_PATTERN_UTIL_H_
#define TVM_RELAY_PASS_PATTERN_UTIL_H_

#include <builtin_fp16.h>
#include <tvm/data_layout.h>
#include <tvm/relay/op.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/attrs/reduce.h>
#include <string>
#include <utility>


namespace tvm {
namespace relay {

/*!
 * \brief Dispatch DataType to the C++ data type
 *  during runtime.
 */
#define TVM_DTYPE_DISPATCH(type, DType, ...)            \
  if (type == Float(64)) {                              \
    typedef double DType;                               \
    {__VA_ARGS__}                                       \
  } else if (type == Float(32)) {                       \
    typedef float DType;                                \
    {__VA_ARGS__}                                       \
  } else if (type == Float(16)) {                       \
    typedef uint16_t DType;                             \
    {__VA_ARGS__}                                       \
  } else if (type == Int(64)) {                         \
    typedef int64_t DType;                              \
    {__VA_ARGS__}                                       \
  } else if (type == Int(32)) {                         \
    typedef int32_t DType;                              \
    {__VA_ARGS__}                                       \
  } else if (type == Int(16)) {                         \
    typedef int16_t DType;                              \
    {__VA_ARGS__}                                       \
  } else if (type == Int(8)) {                          \
    typedef int8_t DType;                               \
    {__VA_ARGS__}                                       \
  } else if (type == UInt(64)) {                        \
    typedef uint64_t DType;                             \
    {__VA_ARGS__}                                       \
  } else if (type == UInt(32)) {                        \
    typedef uint32_t DType;                             \
    {__VA_ARGS__}                                       \
  } else if (type == UInt(16)) {                        \
    typedef uint16_t DType;                             \
    {__VA_ARGS__}                                       \
  } else if (type == UInt(8)) {                         \
    typedef uint8_t DType;                              \
    {__VA_ARGS__}                                       \
  } else {                                              \
    LOG(FATAL) << "unknown data type " << type;         \
  }

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
 * \param target_ndim Target dimension.
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
                              const Layout& kernel_layout) {
  static const Layout kOIHW("OIHW");
  const auto bilayout = BijectiveLayoutNode::make(kernel_layout, kOIHW);
  auto wshape = bilayout.ForwardShape(call->args[1]->type_as<TensorTypeNode>()->shape);
  return is_const_int(wshape[0], param->groups) &&
      is_const_int(wshape[1], 1);
}

/*!
 * \brief Get super-dimension of output channels of conv2d
 * \param call The conv2d call.
 * \return Super-dimension size of output channels of conv2d.
 */
inline int64_t GetConv2DSuperChannelsDim(const CallNode* call) {
    auto param = call->attrs.as<Conv2DAttrs>();
    auto tweight = call->args[1]->type_as<TensorTypeNode>();
    auto index = param->kernel_layout.find('O');
    CHECK_NE(index, std::string::npos);
    auto channels = as_const_int(tweight->shape[index]);
    return *channels;
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
  runtime::NDArray arr = runtime::NDArray::Empty({}, Type2TVMType(dtype), {kDLCPU, 0});
  TVM_DTYPE_DISPATCH(dtype, DType, {
    if (dtype == Float(16)) {
      // convert to float16
      // storage is uint16_t
      *static_cast<DType*>(arr->data) =
        __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(static_cast<float>(value));
    } else {
      *static_cast<DType*>(arr->data) = value;
    }
  })
  return ConstantNode::make(arr);
}

/*!
 * \brief Check if two expressions are equal scalars.
 * \param a The expression to be checked.
 * \param b The expression to be checked
 * \return Whether two expressions are equal scalars.
 */
inline bool IsEqualScalar(const Expr& a, const Expr& b) {
  const auto* constant_a = a.as<ConstantNode>();
  const auto* constant_b = b.as<ConstantNode>();
  if (!constant_a || !constant_b || !constant_a->is_scalar() || !constant_b->is_scalar()) {
    return false;
  }
  return AlphaEqual(a, b);
}

inline Expr GetField(Expr t, size_t i) {
  return TupleGetItemNode::make(t, i);
}

inline Expr Pair(Expr l, Expr r) {
  return TupleNode::make({l, r});
}

inline Expr Exp(Expr e) {
  static const Op& op = Op::Get("exp");
  return CallNode::make(op, {e});
}

inline Expr Log(Expr e) {
  static const Op& op = Op::Get("log");
  return CallNode::make(op, {e});
}
/*!
 * \brief Get an immediate scalar from a Constant expr.
 *
 * \param expr The Constant expr.
 * \return A scalar with type T.
 */
template <typename T>
T GetScalarFromConstant(Expr expr) {
  const auto* n = expr.as<ConstantNode>();
  CHECK(n->is_scalar());
  return static_cast<T*>(n->data->data)[0];
}

inline Expr Cast(Expr x, DataType dtype) {
  static const Op& op = Op::Get("cast");
  auto attrs = make_node<CastAttrs>();
  attrs->dtype = dtype;
  return CallNode::make(op, {x}, Attrs(attrs), {});
}

inline Expr Negative(Expr x) {
  static const Op& op = Op::Get("negative");
  return CallNode::make(op, {x}, Attrs(), {});
}


inline Expr Sqrt(Expr x) {
  static const Op& op = Op::Get("sqrt");
  return CallNode::make(op, {x}, Attrs(), {});
}


inline Expr Relu(Expr x) {
  static const Op& op = Op::Get("nn.relu");
  return CallNode::make(op, {x}, Attrs(), {});
}


inline Expr Round(Expr x) {
  static const Op& op = Op::Get("round");
  return CallNode::make(op, {x}, Attrs(), {});
}


inline Expr Clip(Expr x, double a_min, double a_max) {
  static const Op& op = Op::Get("clip");
  auto attrs = make_node<ClipAttrs>();
  attrs->a_min = a_min;
  attrs->a_max = a_max;
  return CallNode::make(op, {x}, Attrs(attrs), {});
}


inline Expr Add(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("add");
  return CallNode::make(op, {lhs, rhs}, Attrs(), {});
}


inline Expr Subtract(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("subtract");
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

inline Expr ZerosLike(Expr e) {
  static const Op& op = Op::Get("zeros_like");
  return CallNode::make(op, {e});
}

inline Expr Zeros(Array<IndexExpr> shape, DataType dtype) {
  auto attrs = make_node<InitOpAttrs>();
  attrs->shape = std::move(shape);
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("zeros");
  return CallNode::make(op, {}, Attrs(attrs), {});
}

inline Expr OnesLike(Expr e) {
  static const Op& op = Op::Get("ones_like");
  return CallNode::make(op, {e});
}

inline Expr CollapseSumLike(Expr e) {
  static const Op& op = Op::Get("collapse_sum_like");
  return CallNode::make(op, {e});
}

inline Expr Power(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("power");
  return CallNode::make(op, {lhs, rhs}, Attrs(), {});
}


inline Expr RightShift(Expr x, Expr nbit) {
  static const Op& op = Op::Get("right_shift");
  return CallNode::make(op, {x, nbit}, Attrs(), {});
}


inline Expr LeftShift(Expr x, Expr nbit) {
  static const Op& op = Op::Get("left_shift");
  return CallNode::make(op, {x, nbit}, Attrs(), {});
}


inline Expr ReshapeLike(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("reshape_like");
  return CallNode::make(op, {lhs, rhs}, Attrs(), {});
}


inline Expr Copy(Expr data) {
  static const Op& op = Op::Get("copy");
  return CallNode::make(op, {data}, Attrs(), {});
}


inline Expr Mean(Expr data, Array<Integer> axis, bool keepdims, bool exclude) {
  auto attrs = make_node<ReduceAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  attrs->exclude = exclude;
  static const Op& op = Op::Get("mean");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

inline Expr Variance(Expr data, Expr mean, Array<Integer> axis, bool keepdims, bool exclude) {
  auto attrs = make_node<ReduceAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  attrs->exclude = exclude;
  static const Op& op = Op::Get("variance");
  return CallNode::make(op, {data, mean}, Attrs(attrs), {});
}


static inline Expr Where(const Expr& condition, const Expr& x, const Expr& y) {
  static const Op& op = Op::Get("where");
  return CallNode::make(op, {condition, x, y});
}

static inline Expr GreaterEqual(const Expr& lhs, const Expr& rhs) {
  static const Op& op = Op::Get("greater_equal");
  return CallNode::make(op, {lhs, rhs}, Attrs(), {});
}

static inline Expr Full(Expr fill_value,
                        Array<IndexExpr> shape,
                        DataType dtype) {
  auto attrs = make_node<InitOpAttrs>();
  attrs->shape = std::move(shape);
  attrs->dtype = std::move(dtype);
  static const Op& op = Op::Get("full");
  return CallNode::make(op, {fill_value}, Attrs(attrs), {});
}

static inline Expr Conv2D(Expr data, Expr weight, Array<IndexExpr> strides,
                          Array<IndexExpr> padding, Array<IndexExpr> dilation, int groups,
                          IndexExpr channels, Array<IndexExpr> kernel_size, std::string data_layout,
                          std::string kernel_layout, std::string out_layout, DataType out_dtype) {
  auto attrs = make_node<Conv2DAttrs>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->dilation = std::move(dilation);
  attrs->groups = groups;
  attrs->channels = std::move(channels);
  attrs->kernel_size = std::move(kernel_size);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);
  static const Op& op = Op::Get("nn.conv2d");
  return CallNode::make(op, {data, weight}, Attrs(attrs), {});
}

static inline Expr Dense(Expr data,
                         Expr weight,
                         IndexExpr units,
                         DataType out_dtype) {
  auto attrs = make_node<DenseAttrs>();
  attrs->units = units;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("nn.dense");
  return CallNode::make(op, {data, weight}, Attrs(attrs), {});
}

static inline Expr Sum(Expr data, Array<Integer> axis, bool keepdims, bool exclude) {
  auto attrs = make_node<ReduceAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  attrs->exclude = exclude;
  static const Op& op = Op::Get("sum");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

static inline Expr Reshape(Expr data, Array<Integer> newshape) {
  auto attrs = make_node<ReshapeAttrs>();
  attrs->newshape = std::move(newshape);
  attrs->reverse = false;
  static const Op& op = Op::Get("reshape");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

static inline Expr AvgPool2D(Expr data, Array<IndexExpr> pool_size, Array<IndexExpr> strides,
                             Array<IndexExpr> padding, std::string layout, bool ceil_mode,
                             bool count_include_pad) {
  auto attrs = make_node<AvgPool2DAttrs>();
  attrs->pool_size = std::move(pool_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->layout = std::move(layout);
  attrs->ceil_mode = ceil_mode;
  attrs->count_include_pad = count_include_pad;
  static const Op& op = Op::Get("nn.avg_pool2d");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

static inline Expr Pad(Expr data, Array<Array<IndexExpr>> pad_width, double pad_value,
                       std::string pad_mode) {
  auto attrs = make_node<PadAttrs>();
  attrs->pad_value = pad_value;
  attrs->pad_width = std::move(pad_width);
  attrs->pad_mode = std::move(pad_mode);
  static const Op& op = Op::Get("nn.pad");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

static inline Expr Tile(Expr data, Array<Integer> reps) {
  auto attrs = make_node<TileAttrs>();
  attrs->reps = reps;
  static const Op& op = Op::Get("tile");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

Expr MakeConcatenate(Expr data, int axis);

Expr MakeStridedSlice(Expr data, Array<Integer> begin, Array<Integer> end, Array<Integer> strides);

Expr MakeStack(Expr data, int axis);

Expr MakeSplit(Expr data, NodeRef indices_or_sections, int axis);

Expr MakeSqueeze(Expr data, Array<Integer> axis);

Expr MakeExpandDims(Expr data, int axis, int num_newaxis);

Expr MakeLayoutTransform(Expr data, std::string src_layout, std::string dst_layout);

Expr StopFusion(Expr data);

Expr CastHint(Expr data, DataType dtype);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_PATTERN_UTIL_H_
