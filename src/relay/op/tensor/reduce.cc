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
 * \file reduce.cc
 * \brief Reduction operators.
 */
#include <tvm/relay/attrs/reduce.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/reduction.h>

#include <limits>
#include <numeric>

#include "../make_op.h"
#include "../op_common.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(ReduceAttrs);
TVM_REGISTER_NODE_TYPE(ArgReduceAttrs);
TVM_REGISTER_NODE_TYPE(VarianceAttrs);

/*!
 * \brief GetReduceAxes, get the new axis from indim and other arguments
 * \param indim Number of dimensions of input data.
 * \param axis The input axis vector.
 * \param exclude Whether 'axis' input given is the excluded axis.
 * \return r_axes The new reduced axes of the output.
 */
inline std::vector<int64_t> GetReduceAxes(const uint32_t indim, const Array<Integer>& inaxis,
                                          bool exclude) {
  if (!inaxis.defined() || inaxis.empty()) {
    std::vector<int64_t> r_axes(indim);
    std::iota(r_axes.begin(), r_axes.end(), 0);
    return r_axes;
  }

  std::vector<int64_t> in_axes;
  for (auto i : inaxis) {
    int64_t axis = i->value;
    if (axis < 0) {
      axis = axis + indim;
    }

    // Check out of bounds error
    ICHECK(axis >= 0) << "Axis out of bounds in reduce operator.";
    ICHECK(axis < indim) << "Axis out of bounds in reduce operator.";
    in_axes.push_back(axis);
  }

  ICHECK(in_axes[in_axes.size() - 1] < indim)
      << "Reduction axis " << in_axes[in_axes.size() - 1] << " exceeds input dimensions " << indim;

  std::sort(in_axes.begin(), in_axes.end());

  if (!exclude) {
    return in_axes;
  }

  auto r_size = indim - in_axes.size();
  std::vector<int64_t> r_axes(r_size);
  for (uint32_t i = 0, j = 0, k = 0; i < indim; ++i) {
    if (j < in_axes.size() && in_axes[j] == i) {
      ++j;
      continue;
    }
    r_axes[k++] = i;
  }
  return r_axes;
}

// Get axis under exclude condition.
Array<Integer> GetExcludeAxes(size_t indim, const Array<Integer>& inaxis) {
  ICHECK(inaxis.defined()) << "Cannot set exclude when axis=None";
  std::vector<bool> axis_flag(indim, true);
  for (auto i : inaxis) {
    int64_t axis = i->value;
    if (axis < 0) {
      axis = axis + static_cast<int64_t>(indim);
    }
    // Check out of bounds error
    ICHECK_GE(axis, 0) << "Axis out of bounds in reduce operator.";
    ICHECK_LT(axis, static_cast<int64_t>(indim)) << "Axis out of bounds in reduce operator.";
    axis_flag[axis] = false;
  }

  Array<Integer> r_axes;

  for (size_t i = 0; i < axis_flag.size(); ++i) {
    if (axis_flag[i]) {
      r_axes.push_back(static_cast<int>(i));
    }
  }
  return r_axes;
}

// Return the modified layout for AlterOpLayout pass.
template <typename T>
InferCorrectLayoutOutput ReduceInferCorrectLayout(const Attrs& attrs,
                                                  const Array<Layout>& new_in_layouts,
                                                  const Array<Layout>& old_in_layouts,
                                                  const Array<tvm::relay::Type>& old_in_types) {
  const auto* attrs_ptr = attrs.as<T>();
  ICHECK(attrs_ptr);
  ObjectPtr<T> params = make_object<T>(*attrs_ptr);

  // Get the reduce axes.
  Array<Array<IndexExpr>> old_in_shapes;
  for (auto old_in_t : old_in_types) {
    ICHECK(old_in_t.as<TensorTypeNode>());
    old_in_shapes.push_back(old_in_t.as<TensorTypeNode>()->shape);
  }
  uint32_t indim = old_in_shapes[0].size();
  auto r_axes = GetReduceAxes(indim, params->axis, params->exclude);

  Layout inferred_in = Layout::Undef();
  Layout inferred_out = Layout::Undef();

  // Infer [in_layout, out_layout, new_r_axes] from old_in_layout or new_in_layout
  auto infer = [&](const Layout& layout) {
    // 1) Collect the original axes
    std::unordered_set<std::string> old_r_dims;
    for (auto r_axis : r_axes) {
      old_r_dims.emplace(old_in_layouts[0][r_axis].name());
    }

    // 2) Collect the new axes by walking new_layout.
    tvm::Array<tvm::Integer> new_r_axes;
    std::string inferred_in_string = "";
    std::string inferred_out_string = "";
    auto push_new_axis = [&](const std::string& layout_dim, int axis) {
      if ((old_r_dims.count(layout_dim) && !params->exclude) ||
          (!old_r_dims.count(layout_dim) && params->exclude)) {
        new_r_axes.push_back(tvm::Integer(axis));
        return true;
      }
      return false;
    };
    for (size_t axis_index = 0; axis_index < layout->axes.size(); ++axis_index) {
      const auto& layout_axis = LayoutAxis::Get(layout->axes[axis_index]);
      const std::string& layout_dim = layout_axis.name();
      if (layout_axis.IsPrimal()) {
        push_new_axis(layout_dim, axis_index);
        inferred_in_string += layout_dim;
        if (!old_r_dims.count(layout_dim) || params->keepdims) {
          inferred_out_string += layout_dim;
        }
      } else {
        // For example, if the original layout is NCHW, the new layout is NCHW8c, and the original
        // reduce axes is [1], the new reduce axes become [1, 4].
        auto primal_dim = layout_axis.ToPrimal().name();
        auto packed_dim = std::to_string(layout.FactorOf(layout_axis)) + layout_dim;
        inferred_in_string += packed_dim;
        if (push_new_axis(primal_dim, axis_index)) {
          if (params->exclude) {
            // The primal axis is not reduced, so keep the input packed dim.
            inferred_out_string += packed_dim;
          } else if (params->keepdims) {
            // If the primal axis is part of reduce axes in the original layout, the inner dim
            // becomes 1 after reduction.
            inferred_out_string += "1" + layout_dim;
          }
        } else {
          inferred_out_string += packed_dim;
        }
      }
    }

    // 3) Set the new axis and layout.
    return std::make_tuple(Layout(inferred_in_string), Layout(inferred_out_string), new_r_axes);
  };

  std::string new_layout_string;
  Array<Integer> new_r_axes;
  Array<Layout> new_input_layouts;

  auto check_num_input_layouts = [](Array<Layout> in_layouts) {
    // The second case is for variance op
    ICHECK(in_layouts.size() == 1 || in_layouts.size() == 2);
  };

  if (new_in_layouts.defined() && r_axes.size()) {
    // Adapt to new layout. The axis has to change. Record original reduce axes. Convert to the
    // modified layout axes.
    check_num_input_layouts(new_in_layouts);
    check_num_input_layouts(old_in_layouts);

    // Get inferred_in and inferred_out from new_in_layout.
    std::tie(inferred_in, inferred_out, new_r_axes) = infer(new_in_layouts[0]);
    params->axis = new_r_axes;
  } else if (old_in_layouts.defined()) {
    check_num_input_layouts(old_in_layouts);

    // If the new layout is undefined, get inferred_in and inferred_out from old_in_layout.
    if (old_in_layouts[0].defined()) {
      std::tie(inferred_in, inferred_out, std::ignore) = infer(old_in_layouts[0]);
    }
  }

  new_input_layouts.push_back(inferred_in);

  if (old_in_layouts.size() == 2) {
    new_input_layouts.push_back(inferred_in);
  }

  return InferCorrectLayoutOutput(new_input_layouts, {inferred_out}, Attrs(params));
}

template <typename F>
Array<te::Tensor> ReduceCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type, F f) {
  const ReduceAttrs* param = attrs.as<ReduceAttrs>();
  ICHECK(param != nullptr);
  if (inputs[0]->shape.size() == 0) {
    return {topi::identity(inputs[0])};
  }
  auto axes = param->axis;
  if (param->exclude) {
    axes = GetExcludeAxes(inputs[0]->shape.size(), param->axis);
    if (axes.size() == 0) {
      return {topi::identity(inputs[0])};
    }
  }

  return {f(inputs[0], axes, param->keepdims, false)};
}

template <typename F>
Array<te::Tensor> ArgReduceCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                   const Type& out_type, F f) {
  const ArgReduceAttrs* param = attrs.as<ArgReduceAttrs>();
  ICHECK(param != nullptr);
  if (inputs[0]->shape.size() == 0) {
    return {topi::identity(inputs[0])};
  }
  auto axes = param->axis;
  if (param->exclude) {
    axes = GetExcludeAxes(inputs[0]->shape.size(), param->axis);
    if (axes.size() == 0) {
      return {topi::identity(inputs[0])};
    }
  }

  return {f(inputs[0], axes, param->keepdims, false, param->select_last_index)};
}

/*!
 * \brief ReduceShapeImpl get the outshape for the reduction operator
 * \param in_shape Shape of input data.
 * \param param Attrs details.
 * \param reporter The reporter to report solution to.
 * \return oshape Output shape inferred.
 * \tparam AttrsType The attribute type.
 */
template <typename AttrsType>
inline std::vector<IndexExpr> ReduceShapeImpl(const std::vector<IndexExpr>& in_shape,
                                              const AttrsType* param,
                                              const TypeReporter& reporter) {
  uint32_t indim = in_shape.size();
  auto r_axes = GetReduceAxes(indim, param->axis, param->exclude);
  if (!r_axes.size()) {
    return in_shape;
  }

  auto max_shape = tir::make_const(DataType::Int(64), 1);
  bool is_dynamic_input = false;
  for (int64_t axis : r_axes) {
    if (in_shape[axis].as<IntImmNode>()) {
      max_shape *= in_shape[axis];
    } else {
      is_dynamic_input = true;
      break;
    }
  }

  if (is_dynamic_input) {
    ICHECK(reporter->Assert(
        max_shape < tir::make_const(DataType::Int(64), std::numeric_limits<int32_t>::max())))
        << "The maximum possible index of reduced shape cannot be more than int32 max.";
  }

  if (param->keepdims) {
    std::vector<IndexExpr> oshape(in_shape);
    for (unsigned i = 0, j = 0; i < indim; ++i) {
      if (j >= r_axes.size() || !(r_axes[j] == i)) {
        continue;
      }
      oshape[i] = 1;
      ++j;
    }
    return oshape;
  } else {
    auto osize = indim - r_axes.size();
    std::vector<IndexExpr> oshape(osize);
    for (unsigned i = 0, j = 0, k = 0; i < indim; ++i) {
      if (j < r_axes.size() && (r_axes[j] == i)) {
        ++j;
        continue;
      }
      oshape[k++] = in_shape[i];
    }
    return oshape;
  }
}

template <class T>
bool GenericReduceRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  ICHECK(static_cast<int>(data->shape.size()) != 0);
  std::vector<IndexExpr> in_shape(data->shape.begin(), data->shape.end());

  const T* param = attrs.as<T>();
  ICHECK(param != nullptr);

  // assign output type and shape
  auto oshape = ReduceShapeImpl(in_shape, param, reporter);
  reporter->Assign(types[1], TensorType(oshape, data->shape[0].dtype()));
  return true;
}
/*!
 * \brief ArgReduceRel Output type and shape relation evaluation function.
 * \param num_inputs Number of input types in the args.
 * \param attrs The additional attributes of the operator.
 * \param reporter The reporter to report solution to.
 * \return false if This relation cannot be resolved. true if this relation has been resolved.
 */
bool ArgReduceRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  return GenericReduceRel<ReduceAttrs>(types, num_inputs, attrs, reporter);
}

/*!
 * \brief ReduceRel Output type and shape relation evaluation function.
 * \param num_inputs Number of input types in the args.
 * \param attrs The additional attributes of the operator.
 * \param reporter The reporter to report solution to.
 * \return false if This relation cannot be resolved. true if this relation has been resolved.
 */
bool ReduceRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  std::vector<IndexExpr> in_shape(data->shape.begin(), data->shape.end());

  const ReduceAttrs* param = attrs.as<ReduceAttrs>();
  ICHECK(param != nullptr);

  // assign output type and shape
  auto oshape = ReduceShapeImpl(in_shape, param, reporter);
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

Expr MakeReduce(Expr data, Array<Integer> axis, bool keepdims, bool exclude, String op_name) {
  auto attrs = make_object<ReduceAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  attrs->exclude = exclude;
  return Call(Op::Get(op_name), {data}, Attrs(attrs), {});
}

Expr MakeOneElementReduce(Expr data, Array<Integer> axis, bool keepdims, bool exclude,
                          bool select_last_index, String op_name) {
  auto attrs = make_object<ArgReduceAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  attrs->exclude = exclude;
  attrs->select_last_index = select_last_index;
  return Call(Op::Get(op_name), {data}, Attrs(attrs), {});
}

#define RELAY_REGISTER_REDUCE_OP(OpName)                                                \
  TVM_REGISTER_GLOBAL("relay.op._make." OpName)                                         \
      .set_body_typed([](Expr data, Array<Integer> axis, bool keepdims, bool exclude) { \
        return MakeReduce(data, axis, keepdims, exclude, OpName);                       \
      });                                                                               \
  RELAY_REGISTER_OP(OpName).set_num_inputs(1).add_argument("data", "Tensor", "The input tensor.")

#define RELAY_REGISTER_ONE_ELEMENT_REDUCE_OP(OpName)                                           \
  TVM_REGISTER_GLOBAL("relay.op._make." OpName)                                                \
      .set_body_typed([](Expr data, Array<Integer> axis, bool keepdims, bool exclude,          \
                         bool select_last_index) {                                             \
        return MakeOneElementReduce(data, axis, keepdims, exclude, select_last_index, OpName); \
      });                                                                                      \
  RELAY_REGISTER_OP(OpName).set_num_inputs(1).add_argument("data", "Tensor", "The input tensor.")

Array<te::Tensor> ArgMaxCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  return ArgReduceCompute(attrs, inputs, out_type, topi::argmax);
}

RELAY_REGISTER_ONE_ELEMENT_REDUCE_OP("argmax")
    .describe(R"code(Creates an operation that finds the indices of the maximum
values over a given axis.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ArgReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("ArgReduce", GenericReduceRel<ArgReduceAttrs>)
    .set_attr<FTVMCompute>("FTVMCompute", ArgMaxCompute)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<ArgReduceAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<te::Tensor> ArgMinCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  return ArgReduceCompute(attrs, inputs, out_type, topi::argmin);
}

RELAY_REGISTER_ONE_ELEMENT_REDUCE_OP("argmin")
    .describe(R"code(Creates an operation that finds the indices of the minimum
values over a given axis.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ArgReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("ArgReduce", GenericReduceRel<ArgReduceAttrs>)
    .set_attr<FTVMCompute>("FTVMCompute", ArgMinCompute)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<ArgReduceAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<te::Tensor> SumCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                             const Type& out_type) {
  return ReduceCompute(attrs, inputs, out_type, topi::sum);
}

RELAY_REGISTER_REDUCE_OP("sum")
    .describe(R"code(Computes the sum of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  sum(data, axis=1)
  [[  4.   8.]
   [ 10.   9.]
   [ 21.   6.]]

  sum(data, axis=[1,2])
  [ 12.  19.  27.]

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("Reduce", ReduceRel)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<ReduceAttrs>)
    .set_attr<FTVMCompute>("FTVMCompute", SumCompute)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<te::Tensor> AllCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                             const Type& out_type) {
  return ReduceCompute(attrs, inputs, out_type, topi::all);
}

RELAY_REGISTER_REDUCE_OP("all")
    .describe(R"code(Computes the logical AND of boolean array elements over given axes.

Example::

  data = [[[ True,  True,  True],
           [ True,  True,  True],
           [False,  True, False]],
          [[ True, False, False],
           [ True,  True, False],
           [False,  True,  True]]]

  all(data, axis=1)
  [[False,  True, False],
   [False, False, False]]

  all(data, axis=0)
  [[ True, False, False],
   [ True,  True, False],
   [False,  True, False]]

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("Reduce", ReduceRel)
    .set_attr<FTVMCompute>("FTVMCompute", AllCompute)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<ReduceAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<te::Tensor> AnyCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                             const Type& out_type) {
  return ReduceCompute(attrs, inputs, out_type, topi::any);
}

RELAY_REGISTER_REDUCE_OP("any")
    .describe(R"code(Computes the logical OR of boolean array elements over given axes.

Example::

  data = [[[ True,  True,  True],
           [ True,  True,  True],
           [False,  True, False]],
          [[ True, False, False],
           [ True,  True, False],
           [False,  True,  True]]]

  any(data, axis=1)
  [[True,  True, True],
   [True,  True, True]]

  any(data, axis=0)
  [[ True,  True, True],
   [ True,  True, True],
   [False,  True, True]]

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("Reduce", ReduceRel)
    .set_attr<FTVMCompute>("FTVMCompute", AnyCompute)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<te::Tensor> MaxCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                             const Type& out_type) {
  return ReduceCompute(attrs, inputs, out_type, topi::max);
}

RELAY_REGISTER_REDUCE_OP("max")
    .describe(R"code(Computes the max of array elements over given axes.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("Reduce", ReduceRel)
    .set_attr<FTVMCompute>("FTVMCompute", MaxCompute)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<ReduceAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<te::Tensor> MinCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                             const Type& out_type) {
  return ReduceCompute(attrs, inputs, out_type, topi::min);
}

RELAY_REGISTER_REDUCE_OP("min")
    .describe(R"code(Computes the min of array elements over given axes.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("Reduce", ReduceRel)
    .set_attr<FTVMCompute>("FTVMCompute", MinCompute)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<ReduceAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<te::Tensor> ProdCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_type) {
  return ReduceCompute(attrs, inputs, out_type, topi::prod);
}

TVM_REGISTER_GLOBAL("relay.op._make.prod").set_body_typed(Prod);

RELAY_REGISTER_OP("prod")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .describe(R"code(Computes the products of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  prod(data, axis=1)
  [35562240]

  prod(data, axis=[1,2])
  [ 36  480  2058]

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("Reduce", ReduceRel)
    .set_attr<FTVMCompute>("FTVMCompute", ProdCompute)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<ReduceAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<te::Tensor> MeanCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_type) {
  auto data = inputs[0];
  IndexExpr count = tir::make_const(DataType::Int(64), 1);
  const ReduceAttrs* param = attrs.as<ReduceAttrs>();
  ICHECK(param != nullptr);
  auto axes = param->axis;
  for (int64_t i : GetReduceAxes(inputs[0]->shape.size(), param->axis, param->exclude)) {
    count *= inputs[0]->shape[i];
  }
  // Check the datatype of input data. If it's fp16, we'll have trouble representing all
  // indices and summation needed so we instead just cast to fp32.
  bool recast_fp16 = false;
  if (data->dtype.is_float16()) {
    recast_fp16 = true;
    data = topi::cast(data, DataType::Float(32));
  }
  count = cast(data->dtype, count);
  auto res = ReduceCompute(attrs, {data}, out_type, topi::sum);
  auto output = topi::divide(res[0], count);
  // Set the output back to the appropriate fp16 type if needed.
  if (recast_fp16) {
    output = topi::cast(output, DataType::Float(16));
  }
  return {output};
}

RELAY_REGISTER_REDUCE_OP("mean")
    .describe(R"code(Computes the mean of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  mean(data)
  [3.22]

  mean(data, axis=[1,2])
  [ 2.  3.16666667  4.5]

)code" TVM_ADD_FILELINE)
    .set_attrs_type<ReduceAttrs>()
    .set_support_level(4)
    .add_type_rel("Reduce", ReduceRel)
    .set_attr<FTVMCompute>("FTVMCompute", MeanCompute)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<ReduceAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

bool VarianceRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                 const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  ICHECK(static_cast<int>(data->shape.size()) != 0);
  const auto* mean = types[1].as<TensorTypeNode>();
  if (mean == nullptr) return false;

  std::vector<IndexExpr> in_shape(data->shape.begin(), data->shape.end());
  std::vector<IndexExpr> mean_shape(mean->shape.begin(), mean->shape.end());
  ICHECK_EQ(in_shape.size(), mean_shape.size());

  const VarianceAttrs* param = attrs.as<VarianceAttrs>();
  ICHECK(param != nullptr);

  // assign output type and shape
  auto oshape = ReduceShapeImpl(in_shape, param, reporter);
  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}

Array<te::Tensor> VarianceCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                  const Type& out_type) {
  IndexExpr count = tir::make_const(DataType::Int(64), 1);
  const VarianceAttrs* param = attrs.as<VarianceAttrs>();
  ICHECK(param != nullptr);
  auto axes = param->axis;
  bool unbiased = param->unbiased;
  auto data = inputs[0];
  auto mean = inputs[1];
  for (int64_t i : GetReduceAxes(data->shape.size(), param->axis, param->exclude)) {
    count *= data->shape[i];
  }
  if (unbiased) {
    count -= 1;
  }
  std::vector<Integer> expand_shape;
  auto diff = topi::subtract(data, mean);
  auto sq_diff = topi::multiply(diff, diff);
  if (param->exclude) {
    axes = GetExcludeAxes(sq_diff->shape.size(), param->axis);
    ICHECK_NE(axes.size(), 0);
  }
  // If the input is fp16, we might have trouble representing the full sum of
  // indices or values. We recast to fp32 to avoid this issue.
  bool recast_fp16 = false;
  if (data->dtype.is_float16()) {
    recast_fp16 = true;
    sq_diff = topi::cast(sq_diff, DataType::Float(32));
  }
  auto var = topi::divide(topi::sum(sq_diff, axes, param->keepdims, false), count);

  // Recast back to fp16 if needed.
  if (recast_fp16) {
    var = topi::cast(var, DataType::Float(16));
  }

  return {var};
}

Expr MakeVariance(Expr data, Expr mean, Array<Integer> axis, bool keepdims, bool exclude,
                  bool unbiased = false) {
  auto attrs = make_object<VarianceAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  attrs->exclude = exclude;
  attrs->unbiased = unbiased;
  static const Op& op = Op::Get("variance");
  return Call(op, {data, mean}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op._make._variance").set_body_typed(MakeVariance);

RELAY_REGISTER_OP("variance")
    .describe(R"code(Computes the variance of array elements over given axes.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<VarianceAttrs>()
    .set_support_level(4)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("mean", "Tensor", "The mean tensor.")
    .add_type_rel("Variance", VarianceRel)
    .set_attr<FTVMCompute>("FTVMCompute", VarianceCompute)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ReduceInferCorrectLayout<VarianceAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kCommReduce);

}  // namespace relay
}  // namespace tvm
