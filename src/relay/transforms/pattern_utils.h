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
 *
 * \file tvm/relay/transforms/pattern_utils.h
 * \brief Header of internal operator functions
 *  These can be used for writing passes.
 */
#ifndef TVM_RELAY_TRANSFORMS_PATTERN_UTILS_H_
#define TVM_RELAY_TRANSFORMS_PATTERN_UTILS_H_

#include <builtin_fp16.h>
#include <dmlc/optional.h>
#include <tvm/node/structural_equal.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/data_layout.h>

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "../op/make_op.h"

namespace tvm {
namespace relay {

/*!
 * \brief Dispatch DataType to the C++ data type
 *  during runtime.
 */
#define TVM_DTYPE_DISPATCH(type, DType, ...)                                          \
  if (type == DataType::Float(64)) {                                                  \
    typedef double DType;                                                             \
    { __VA_ARGS__ }                                                                   \
  } else if (type == DataType::Float(32)) {                                           \
    typedef float DType;                                                              \
    { __VA_ARGS__ }                                                                   \
  } else if (type == DataType::Float(16)) {                                           \
    typedef uint16_t DType;                                                           \
    { __VA_ARGS__ }                                                                   \
  } else if (type == DataType::Int(64)) {                                             \
    typedef int64_t DType;                                                            \
    { __VA_ARGS__ }                                                                   \
  } else if (type == DataType::Int(32)) {                                             \
    typedef int32_t DType;                                                            \
    { __VA_ARGS__ }                                                                   \
  } else if (type == DataType::Int(16)) {                                             \
    typedef int16_t DType;                                                            \
    { __VA_ARGS__ }                                                                   \
  } else if (type == DataType::Int(8)) {                                              \
    typedef int8_t DType;                                                             \
    { __VA_ARGS__ }                                                                   \
  } else if (type == DataType::UInt(64)) {                                            \
    typedef uint64_t DType;                                                           \
    { __VA_ARGS__ }                                                                   \
  } else if (type == DataType::UInt(32)) {                                            \
    typedef uint32_t DType;                                                           \
    { __VA_ARGS__ }                                                                   \
  } else if (type == DataType::UInt(16)) {                                            \
    typedef uint16_t DType;                                                           \
    { __VA_ARGS__ }                                                                   \
  } else if (type == DataType::UInt(8)) {                                             \
    typedef uint8_t DType;                                                            \
    { __VA_ARGS__ }                                                                   \
  } else if (type == DataType::Bool()) {                                              \
    typedef bool DType;                                                               \
    { __VA_ARGS__ }                                                                   \
  } else if ((*tvm::runtime::Registry::Get("runtime._datatype_get_type_registered"))( \
                 static_cast<uint8_t>(type.code()))) {                                \
    typedef double DType;                                                             \
    { __VA_ARGS__ }                                                                   \
  } else {                                                                            \
    LOG(FATAL) << "unknown data type " << type;                                       \
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
inline bool MatchBroadcastToLeftAxes(const TensorTypeNode* tlhs, const TensorTypeNode* trhs,
                                     const Array<Integer>& lhs_axes, Expr* rhs_value = nullptr) {
  if (tlhs->shape.size() < trhs->shape.size()) return false;
  StructuralEqual equal;
  size_t base = tlhs->shape.size() - trhs->shape.size();
  size_t j = 0;

  ObjectPtr<SqueezeAttrs> squeeze_attrs;
  if (rhs_value != nullptr) {
    squeeze_attrs = make_object<SqueezeAttrs>();
  }

  for (size_t i = 0; i < tlhs->shape.size(); ++i) {
    if (j < lhs_axes.size() && i == static_cast<size_t>(lhs_axes[j]->value)) {
      if (i < base || !equal(tlhs->shape[i], trhs->shape[i - base])) {
        return false;
      }
      ++j;
    } else if (i >= base) {
      if (!tir::is_const_int(trhs->shape[i - base], 1)) {
        return false;
      }
      if (rhs_value != nullptr) {
        squeeze_attrs->axis.push_back(static_cast<int>(i - base));
      }
    }
  }
  if (rhs_value != nullptr && squeeze_attrs->axis.size() != 0) {
    static const Op& squeeze_op = Op::Get("squeeze");
    *rhs_value = Call(squeeze_op, {rhs_value[0]}, Attrs(squeeze_attrs), {});
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
inline Expr ExpandBiasToMatchAxis(Expr bias, int target_ndim, const Array<Integer>& axes) {
  static const Op& expand_dims = Op::Get("expand_dims");
  for (size_t i = axes.size(); i != 0; --i) {
    if (i == axes.size()) {
      int64_t num_pad_axis = target_ndim - axes[i - 1]->value - 1;
      if (num_pad_axis > 0) {
        auto attrs = make_object<ExpandDimsAttrs>();
        attrs->axis = i;
        attrs->num_newaxis = static_cast<int>(num_pad_axis);
        bias = Call(expand_dims, {bias}, Attrs(attrs), {});
      }
    } else {
      int64_t diff = axes[i]->value - axes[i - 1]->value;
      ICHECK_GE(diff, 0L);
      if (diff > 0) {
        auto attrs = make_object<ExpandDimsAttrs>();
        attrs->axis = i;
        attrs->num_newaxis = static_cast<int>(diff);
        bias = Call(expand_dims, {bias}, Attrs(attrs), {});
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
inline bool IsDepthwiseConv2D(const Call& call, const Conv2DAttrs* param,
                              const Layout& kernel_layout) {
  static const Layout kOIHW("OIHW");
  const auto bilayout = tir::BijectiveLayout(kernel_layout, kOIHW);
  auto wshape = bilayout.ForwardShape(call->args[1]->type_as<TensorTypeNode>()->shape);
  return tir::is_const_int(wshape[0], param->groups) && tir::is_const_int(wshape[1], 1);
}

/*!
 * \brief Get super-dimension of output channels of conv2d
 * \param call The conv2d call.
 * \return Super-dimension size of output channels of conv2d.
 */
inline int64_t GetConv2DSuperChannelsDim(const CallNode* call) {
  auto param = call->attrs.as<Conv2DAttrs>();
  auto tweight = call->args[1]->type_as<TensorTypeNode>();
  auto index = param->kernel_layout.operator std::string().find('O');
  ICHECK_NE(index, std::string::npos);
  auto channels = tir::as_const_int(tweight->shape[index]);
  return *channels;
}

/*!
 * \brief Is single value tensor (scalar).
 * \param expr The expr.
 * \return True if single value tensor.
 */
inline bool IsScalar(const Expr& expr) {
  if (auto tensor_type = expr->checked_type().as<TensorTypeNode>()) {
    for (auto dim_index_expr : tensor_type->shape) {
      if (auto dim_index = dim_index_expr.as<IntImmNode>()) {
        if (dim_index->value != 1) {
          return false;
        }
      } else {
        return false;
      }
    }
  } else {
    return false;
  }
  return true;
}

/*!
 * \brief Check if expr is a const scalar.
 * \param expr The expr.
 * \return True if const scalar.
 */
inline bool IsConstScalar(const Expr& expr) {
  const auto* const_expr = expr.as<ConstantNode>();
  if (const_expr) {
    return const_expr->is_scalar();
  }
  return false;
}

/*!
 * \brief Create a Constant with a scalar
 *
 * \param dtype The data type.
 * \param value The value of the scalar.
 * \return A Constant.
 */
template <typename T>
inline Constant MakeConstantScalar(DataType dtype, T value) {
  runtime::NDArray arr = runtime::NDArray::Empty({}, dtype, {kDLCPU, 0});
  TVM_DTYPE_DISPATCH(dtype, DType, {
    if (dtype == DataType::Float(16)) {
      // convert to float16
      // storage is uint16_t
      *static_cast<DType*>(arr->data) =
          __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(static_cast<float>(value));
    } else {
      *static_cast<DType*>(arr->data) = value;
    }
  })
  return Constant(arr);
}

/*!
 * \brief Create a Constant with a tensor.
 *
 * \param dtype The data type.
 * \param value The vector of the tensor values.
 * \return A Constant.
 */
template <typename T>
static inline Constant MakeConstantTensor(DataType dtype, std::vector<int64_t> shape,
                                          std::vector<T> value) {
  runtime::NDArray arr = runtime::NDArray::Empty(shape, dtype, {kDLCPU, 0});
  TVM_DTYPE_DISPATCH(dtype, DType, {
    for (size_t i = 0; i < value.size(); i++) {
      if (dtype == DataType::Float(16)) {
        // convert to float16
        // storage is uint16_t
        // Similar handling as that in MakeConstantScalar
        *(static_cast<DType*>(arr->data) + i) =
            __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(
                static_cast<float>(value[i]));
      } else {
        *(static_cast<DType*>(arr->data) + i) = value[i];
      }
    }
  })
  return Constant(arr);
}

/*!
 * \brief Create a Constant with a tensor.
 *
 * \param dtype The data type.
 * \param value The array of the tensor values.
 * \return A Constant.
 */
template <typename T>
static inline Constant MakeConstantTensor(DataType dtype, std::vector<int64_t> shape,
                                          Array<T> value) {
  runtime::NDArray arr = runtime::NDArray::Empty(shape, dtype, {kDLCPU, 0});
  TVM_DTYPE_DISPATCH(dtype, DType, {
    for (size_t i = 0; i < value.size(); i++) {
      if (dtype == DataType::Float(16)) {
        // convert to float16
        // storage is uint16_t
        // Similar handling as that in MakeConstantScalar
        *(static_cast<DType*>(arr->data) + i) =
            __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(
                static_cast<float>(value[i]));
      } else {
        *(static_cast<DType*>(arr->data) + i) = value[i];
      }
    }
  })
  return Constant(arr);
}

/*!
 * \brief Check whether a shape is static and create corresponding Constant.
 Eventually this will be removed and replaced with CheckConstantShapeArrayInteger
 *
 * \param shape The Array of the shape values.
 * \return A Constant.
 */
static inline Constant CheckConstantShape(const Array<IndexExpr>& shape) {
  auto shape_array =
      runtime::NDArray::Empty({int64_t(shape.size())}, DataType::Int(64), {kDLCPU, 0});
  auto* shape_data = static_cast<int64_t*>(shape_array->data);
  for (size_t i = 0; i < shape.size(); ++i) {
    const auto& dim_val = shape[i].as<IntImmNode>();
    ICHECK(dim_val) << "Do not support symbolic shape for "
                       "Array format. Pass shape as Expr instead.";
    shape_data[i] = dim_val->value;
  }
  return Constant(shape_array);
}

/*!
 * \brief Check whether a shape is static and create corresponding Array<Integer>. Will replace
 * CheckConstantShape after dynamic refactorization is complete
 *
 * \param shape The Array of the shape values.
 * \return A Constant.
 */
static inline Array<Integer> CheckConstantShapeArrayInteger(const Array<IndexExpr>& shape) {
  Array<Integer> constShape;

  for (size_t i = 0; i < shape.size(); ++i) {
    const auto& dim_val = shape[i].as<IntImmNode>();
    ICHECK(dim_val) << "Do not support symbolic shape for "
                       "Array format. Pass shape as Expr instead.";

    constShape.push_back(dim_val->value);
  }
  return constShape;
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
  return tvm::StructuralEqual()(a, b);
}

/*!
 * \brief Convert an element of a NDArray with type int or float to scalar.
 * \param array Input NDArray
 * \param i element index
 * \return Converted scalar value, or None if conversion failed
 */
static inline dmlc::optional<long double> TryToScalar(const runtime::NDArray& array, size_t i = 0) {
  if (array->dtype.code == kDLInt) {
    if (array->dtype.bits == 8) {
      return dmlc::optional<long double>(reinterpret_cast<int8_t*>(array->data)[i]);
    } else if (array->dtype.bits == 16) {
      return dmlc::optional<long double>(reinterpret_cast<int16_t*>(array->data)[i]);
    } else if (array->dtype.bits == 32) {
      return dmlc::optional<long double>(reinterpret_cast<int32_t*>(array->data)[i]);
    } else if (array->dtype.bits == 64) {
      return dmlc::optional<long double>(reinterpret_cast<int64_t*>(array->data)[i]);
    }
  } else if (array->dtype.code == kDLUInt) {
    if (array->dtype.bits == 1) {  // bool
      return dmlc::optional<long double>(reinterpret_cast<uint8_t*>(array->data)[i]);
    } else if (array->dtype.bits == 8) {
      return dmlc::optional<long double>(reinterpret_cast<uint8_t*>(array->data)[i]);
    } else if (array->dtype.bits == 16) {
      return dmlc::optional<long double>(reinterpret_cast<uint16_t*>(array->data)[i]);
    } else if (array->dtype.bits == 32) {
      return dmlc::optional<long double>(reinterpret_cast<uint32_t*>(array->data)[i]);
    } else if (array->dtype.bits == 64) {
      return dmlc::optional<long double>(reinterpret_cast<uint64_t*>(array->data)[i]);
    }
  } else if (array->dtype.code == kDLFloat) {
    if (array->dtype.bits == 16) {
      return dmlc::optional<long double>(
          __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(
              reinterpret_cast<uint16_t*>(array->data)[i]));
    }
    if (array->dtype.bits == 32) {
      return dmlc::optional<long double>(reinterpret_cast<float*>(array->data)[i]);
    } else if (array->dtype.bits == 64) {
      return dmlc::optional<long double>(reinterpret_cast<double*>(array->data)[i]);
    }
  }
  return dmlc::optional<long double>();
}

/*!
 * \brief Convert an element of a NDArray with type int or float to scalar.
 * \param array Input NDArray
 * \param i element index
 * \return Converted scalar value
 */
static inline long double ToScalar(const runtime::NDArray& array, size_t i = 0) {
  auto try_value = TryToScalar(array, i);
  ICHECK(try_value) << "Unknown data type: " << tvm::runtime::DLDataType2String(array->dtype);
  return try_value.value();
}

/*!
 * \brief Convert a NDArray with type int or float to Array<Integer>.
 * \param array Input NDArray
 * \return Converted Array.
 */
static inline Array<Integer> ToVector(const runtime::NDArray& array) {
  size_t ndim = array.Shape().size();
  ICHECK_EQ(ndim, 1) << "This function should only be used for 1D NDArrays";
  size_t len = array.Shape().front();
  Array<Integer> out;
  for (size_t i = 0; i < len; ++i) {
    long double elem_val = ToScalar(array, i);
    out.push_back(Integer(IntImm(DataType::Int(32), static_cast<int64_t>(elem_val))));
  }
  return out;
}

/*!
 * \brief Convert a NDArray with type int or float to Array<Array<Integer>>.
 * \param array Input NDArray
 * \return Converted Array.
 */
static inline Array<Array<Integer>> ToMatrix(const runtime::NDArray& array) {
  size_t ndim = array.Shape().size();
  ICHECK_EQ(ndim, 2) << "This function should only used for 2D NDArrays";
  size_t dim1 = array.Shape().at(0);
  size_t dim2 = array.Shape().at(1);

  Array<Array<Integer>> out;

  for (size_t i = 0; i < dim1; ++i) {
    Array<Integer> inner_out;
    for (size_t j = 0; j < dim2; ++j) {
      double elem_val = ToScalar(array, i * dim2 + j);
      inner_out.push_back(Integer(static_cast<int>(elem_val)));
    }
    out.push_back(inner_out);
  }
  return out;
}

inline Expr GetField(Expr t, size_t i) { return TupleGetItem(t, i); }

inline Expr Pair(Expr l, Expr r) { return Tuple({l, r}); }

inline Expr Exp(Expr e) {
  static const Op& op = Op::Get("exp");
  return Call(op, {e});
}

inline Expr FastExp(Expr e) {
  static const Op& op = Op::Get("fast_exp");
  return Call(op, {e});
}

inline Expr FastErf(Expr e) {
  static const Op& op = Op::Get("fast_erf");
  return Call(op, {e});
}

inline Expr FastTanh(Expr e) {
  static const Op& op = Op::Get("fast_tanh");
  return Call(op, {e});
}

inline Expr FastSoftmax(Expr e, tvm::Attrs attr) {
  static const Op& op = Op::Get("nn.fast_softmax");
  return Call(op, {e}, attr);
}

inline Expr Log(Expr e) {
  static const Op& op = Op::Get("log");
  return Call(op, {e});
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
  ICHECK(n) << "Expr must be a constant expr - " << AsText(expr, false);
  ICHECK(n->is_scalar());
  return static_cast<T*>(n->data->data)[0];
}

inline Expr Cast(Expr x, DataType dtype) { return MakeCast(x, dtype); }

inline Expr Negative(Expr x) {
  static const Op& op = Op::Get("negative");
  return Call(op, {x}, Attrs(), {});
}

inline Expr Sqrt(Expr x) {
  static const Op& op = Op::Get("sqrt");
  return Call(op, {x}, Attrs(), {});
}

inline Expr Relu(Expr x) {
  static const Op& op = Op::Get("nn.relu");
  return Call(op, {x}, Attrs(), {});
}

inline Expr Round(Expr x) {
  static const Op& op = Op::Get("round");
  return Call(op, {x}, Attrs(), {});
}

inline Expr Clip(Expr x, double a_min, double a_max) { return MakeClip(x, a_min, a_max); }

inline Expr FixedPointMultiply(Expr x, int32_t multiplier, int32_t shift) {
  static const Op& op = Op::Get("fixed_point_multiply");
  auto attrs = make_object<FixedPointMultiplyAttrs>();
  attrs->multiplier = multiplier;
  attrs->shift = shift;
  return Call(op, {x}, Attrs(attrs), {});
}

inline Expr Add(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("add");
  return Call(op, {lhs, rhs}, Attrs(), {});
}

inline Expr Subtract(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("subtract");
  return Call(op, {lhs, rhs}, Attrs(), {});
}

inline Expr Multiply(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("multiply");
  return Call(op, {lhs, rhs}, Attrs(), {});
}

inline Expr Divide(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("divide");
  return Call(op, {lhs, rhs}, Attrs(), {});
}

inline Expr Maximum(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("maximum");
  return Call(op, {lhs, rhs}, Attrs(), {});
}

inline Expr ZerosLike(Expr e) {
  static const Op& op = Op::Get("zeros_like");
  return Call(op, {e});
}

inline Expr Zeros(Array<IndexExpr> shape, DataType dtype) {
  return MakeZeros(CheckConstantShapeArrayInteger(shape), dtype);
}

inline Expr OnesLike(Expr e) {
  static const Op& op = Op::Get("ones_like");
  return Call(op, {e});
}

inline Expr Ones(Array<IndexExpr> shape, DataType dtype) {
  return MakeOnes(CheckConstantShapeArrayInteger(shape), dtype);
}

inline Expr CollapseSumLike(Expr e) {
  static const Op& op = Op::Get("collapse_sum_like");
  return Call(op, {e});
}

inline Expr Power(Expr lhs, Expr rhs) {
  static const Op& op = Op::Get("power");
  return Call(op, {lhs, rhs}, Attrs(), {});
}

inline Expr RightShift(Expr x, Expr nbit) {
  static const Op& op = Op::Get("right_shift");
  return Call(op, {x, nbit}, Attrs(), {});
}

inline Expr LeftShift(Expr x, Expr nbit) {
  static const Op& op = Op::Get("left_shift");
  return Call(op, {x, nbit}, Attrs(), {});
}

inline Expr ReshapeLike(Expr lhs, Expr rhs, int lhs_begin, Integer lhs_end, int rhs_begin,
                        Integer rhs_end) {
  return MakeReshapeLike(lhs, rhs, lhs_begin, lhs_end, rhs_begin, rhs_end);
}

inline Expr Copy(Expr data) {
  static const Op& op = Op::Get("copy");
  return Call(op, {data}, Attrs(), {});
}

inline Expr Mean(Expr data, Array<Integer> axis, bool keepdims, bool exclude) {
  return MakeReduce(data, axis, keepdims, exclude, "mean");
}

inline Expr Variance(Expr data, Expr mean, Array<Integer> axis, bool keepdims, bool exclude,
                     bool unbiased = false) {
  return MakeVariance(data, mean, axis, keepdims, exclude, unbiased);
}

static inline Expr Where(const Expr& condition, const Expr& x, const Expr& y) {
  static const Op& op = Op::Get("where");
  return Call(op, {condition, x, y});
}

static inline Expr GreaterEqual(const Expr& lhs, const Expr& rhs) {
  static const Op& op = Op::Get("greater_equal");
  return Call(op, {lhs, rhs}, Attrs(), {});
}

static inline Expr Full(Expr fill_value, Array<IndexExpr> shape, DataType dtype) {
  return MakeFull(fill_value, CheckConstantShapeArrayInteger(shape), dtype);
}

static inline Expr Conv2D(Expr data, Expr weight, Array<IndexExpr> strides,
                          Array<IndexExpr> padding, Array<IndexExpr> dilation, int groups,
                          IndexExpr channels, Array<IndexExpr> kernel_size, std::string data_layout,
                          std::string kernel_layout, std::string out_layout, DataType out_dtype) {
  return MakeConv<Conv2DAttrs>(data, weight, strides, padding, dilation, groups, channels,
                               kernel_size, data_layout, kernel_layout, out_layout, out_dtype,
                               "nn.conv2d");
}

static inline Expr Dense(Expr data, Expr weight, IndexExpr units, DataType out_dtype) {
  return MakeDense(data, weight, units, out_dtype);
}

static inline Expr Sum(Expr data, Array<Integer> axis, bool keepdims, bool exclude) {
  return MakeReduce(data, axis, keepdims, exclude, "sum");
}

static inline Expr Prod(Expr data, Array<Integer> axis, bool keepdims, bool exclude) {
  return MakeReduce(data, axis, keepdims, exclude, "prod");
}

static inline Expr Reshape(Expr data, Array<Integer> newshape) {
  return MakeReshape(data, newshape);
}

static inline Expr AvgPool2D(Expr data, Array<IndexExpr> pool_size, Array<IndexExpr> strides,
                             Array<IndexExpr> dilation, Array<IndexExpr> padding,
                             std::string layout, bool ceil_mode, bool count_include_pad) {
  return MakeAvgPool<AvgPool2DAttrs>(data, pool_size, strides, dilation, padding, layout, ceil_mode,
                                     count_include_pad, "nn.avg_pool2d");
}

static inline Expr Pad(Expr data, Array<Array<IndexExpr>> pad_width, Expr pad_value,
                       std::string pad_mode) {
  Array<Array<Integer>> pad_width_int;
  for (size_t i = 0; i < pad_width.size(); ++i) {
    pad_width_int.push_back(CheckConstantShapeArrayInteger(pad_width[i]));
  }
  return MakePad(data, pad_width_int, pad_value, pad_mode);
}

static inline Expr Tile(Expr data, Array<Integer> reps) { return MakeTile(data, reps); }

static inline Expr BroadCastTo(Expr data, Array<IndexExpr> shape) {
  return MakeBroadCastTo(data, CheckConstantShapeArrayInteger(shape));
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_TRANSFORMS_PATTERN_UTILS_H_
