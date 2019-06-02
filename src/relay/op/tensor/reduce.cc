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
 *  Copyright (c) 2018 by Contributors
 * \file reduce.cc
 * \brief Reduction operators.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <topi/elemwise.h>
#include <topi/reduction.h>
#include <numeric>
#include <limits>
#include "../op_common.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {

/*! \brief Attributes for Reduce operators */
struct ReduceAttrs : public tvm::AttrsNode<ReduceAttrs> {
  Array<Integer> axis;
  bool keepdims;
  bool exclude;

  TVM_DECLARE_ATTRS(ReduceAttrs, "relay.attrs.ReduceAttrs") {
    TVM_ATTR_FIELD(axis).set_default(NullValue<Array<Integer>>())
        .describe(R"code(The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.)code");

    TVM_ATTR_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
    TVM_ATTR_FIELD(exclude).set_default(false)
      .describe("Whether to perform reduction on axis that are NOT in axis instead.");
  }
};

/*!
* \brief GetReduceAxes, get the new axis from indim and other arguments
* \param indim Number of dimensions of input data.
* \param axis The input axis vector.
* \param exclude Whether 'axis' input given is the excluded axis.
* \return r_axes The new reduced axes of the output.
*/
inline std::vector<int64_t> GetReduceAxes(const uint32_t indim,
                                          const Array<Integer>& inaxis,
                                          bool exclude) {
  if (!inaxis.defined()) {
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
    CHECK(axis >= 0)
      << "Axis out of bounds in reduce operator.";
    CHECK(axis < indim)
      << "Axis out of bounds in reduce operator.";
    in_axes.push_back(axis);
  }

  CHECK(in_axes[in_axes.size() - 1] < indim)
    << "Reduction axis " << in_axes[in_axes.size() - 1]
    << " exceeds input dimensions " << indim;

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
Array<Integer> GetExcludeAxes(size_t indim,
                              const Array<Integer>& inaxis) {
  CHECK(inaxis.defined()) << "Cannot set exclude when axis=None";
  std::vector<bool> axis_flag(indim, true);
  for (auto i : inaxis) {
    int64_t axis = i->value;
    if (axis < 0) {
      axis = axis + static_cast<int64_t>(indim);
    }
    // Check out of bounds error
    CHECK_GE(axis, 0)
      << "Axis out of bounds in reduce operator.";
    CHECK_LT(axis, static_cast<int64_t>(indim))
      << "Axis out of bounds in reduce operator.";
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


template<typename F>
Array<Tensor> ReduceCompute(const Attrs& attrs,
                            const Array<Tensor>& inputs,
                            const Type& out_type,
                            const Target& target,
                            F f) {
  const ReduceAttrs* param = attrs.as<ReduceAttrs>();
  CHECK(param != nullptr);
  if (inputs[0]->shape.size() == 0) {
    return { topi::identity(inputs[0]) };
  }
  auto axes = param->axis;
  if (param->exclude) {
    axes = GetExcludeAxes(inputs[0]->shape.size(), param->axis);
    if (axes.size() == 0) {
      return { topi::identity(inputs[0]) };
    }
  }
  return { f(inputs[0], axes, param->keepdims, false) };
}

/*!
* \brief ReduceShapeImpl get the outshape for the reduction operator
* \param in_shape Shape of input data.
* \param param ReduceAttrs details.
* \param reporter The reporter to report solution to.
* \return oshape Output shape inferred.
*/
inline std::vector<IndexExpr> ReduceShapeImpl(const std::vector<IndexExpr> &in_shape,
                                              const ReduceAttrs* param,
                                              const TypeReporter& reporter) {
  uint32_t indim = in_shape.size();
  auto r_axes = GetReduceAxes(indim, param->axis, param->exclude);
  if (!r_axes.size()) {
    return in_shape;
  }

  auto max_shape = make_const(Int(64), 1);
  for (int64_t axis : r_axes) {
    max_shape *= in_shape[axis];
  }
  CHECK(reporter->Assert(max_shape < make_const(Int(64), std::numeric_limits<int32_t>::max())))
    << "The maximum possible index of reduced shape cannot be more than int32 max.";

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

/*!
* \brief ArgReduceRel Output type and shape relation evaluation function.
* \param num_inputs Number of input types in the args.
* \param attrs The additional attributes of the operator.
* \param reporter The reporter to report solution to.
* \return false if This relation cannot be resolved. true if this relation has been resolved.
*/
bool ArgReduceRel(const Array<Type>& types,
               int num_inputs,
               const Attrs& attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  CHECK(static_cast<int>(data->shape.size()) != 0);
  std::vector<IndexExpr>&& in_shape = AsVector(data->shape);

  const ReduceAttrs* param = attrs.as<ReduceAttrs>();
  CHECK(param != nullptr);

  // assign output type and shape
  auto oshape = ReduceShapeImpl(in_shape, param, reporter);
  reporter->Assign(types[1], TensorTypeNode::make(oshape, Int(32)));
  return true;
}

/*!
* \brief ReduceRel Output type and shape relation evaluation function.
* \param num_inputs Number of input types in the args.
* \param attrs The additional attributes of the operator.
* \param reporter The reporter to report solution to.
* \return false if This relation cannot be resolved. true if this relation has been resolved.
*/
bool ReduceRel(const Array<Type>& types,
               int num_inputs,
               const Attrs& attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  std::vector<IndexExpr>&& in_shape = AsVector(data->shape);

  const ReduceAttrs* param = attrs.as<ReduceAttrs>();
  CHECK(param != nullptr);

  // assign output type and shape
  auto oshape = ReduceShapeImpl(in_shape, param, reporter);
  reporter->Assign(types[1], TensorTypeNode::make(oshape, data->dtype));
  return true;
}

#define RELAY_REGISTER_REDUCE_OP(OpName)                           \
  TVM_REGISTER_API("relay.op._make." OpName)                       \
  .set_body_typed<Call(Expr, Array<Integer>, bool, bool)>([](      \
                        Expr data,                                 \
                        Array<Integer> axis,                       \
                        bool keepdims,                             \
                        bool exclude) {                            \
      auto attrs = make_node<ReduceAttrs>();                       \
      attrs->axis = std::move(axis);                               \
      attrs->keepdims = keepdims;                                  \
      attrs->exclude = exclude;                                    \
      static const Op& op = Op::Get(OpName);                       \
      return CallNode::make(op, {data}, Attrs(attrs), {});         \
    });                                                            \
  RELAY_REGISTER_OP(OpName)                                        \
  .set_num_inputs(1)                                               \
  .add_argument("data", "Tensor", "The input tensor.")


Array<Tensor> ArgMaxCompute(const Attrs& attrs,
                            const Array<Tensor>& inputs,
                            const Type& out_type,
                            const Target& target) {
  return ReduceCompute(attrs, inputs, out_type, target, topi::argmax);
}


RELAY_REGISTER_REDUCE_OP("argmax")
.describe(R"code(Creates an operation that finds the indices of the maximum
values over a given axis.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.ReduceAttrs")
.set_support_level(4)
.add_type_rel("ArgReduce", ArgReduceRel)
.set_attr<FTVMCompute>("FTVMCompute", ArgMaxCompute)
.set_attr<TOpPattern>("TOpPattern", kCommReduce);


Array<Tensor> ArgMinCompute(const Attrs& attrs,
                            const Array<Tensor>& inputs,
                            const Type& out_type,
                            const Target& target) {
  return ReduceCompute(attrs, inputs, out_type, target, topi::argmin);
}

RELAY_REGISTER_REDUCE_OP("argmin")
.describe(R"code(Creates an operation that finds the indices of the minimum
values over a given axis.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.ReduceAttrs")
.set_support_level(4)
.add_type_rel("ArgReduce", ArgReduceRel)
.set_attr<FTVMCompute>("FTVMCompute", ArgMinCompute)
.set_attr<TOpPattern>("TOpPattern", kCommReduce);

Array<Tensor> SumCompute(const Attrs& attrs,
                         const Array<Tensor>& inputs,
                         const Type& out_type,
                         const Target& target) {
  return ReduceCompute(attrs, inputs, out_type, target, topi::sum);
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
.set_attrs_type_key("relay.attrs.ReduceAttrs")
.set_support_level(4)
.add_type_rel("Reduce", ReduceRel)
.set_attr<FTVMCompute>("FTVMCompute", SumCompute)
.set_attr<TOpPattern>("TOpPattern", kCommReduce);


Array<Tensor> AllCompute(const Attrs& attrs,
                         const Array<Tensor>& inputs,
                         const Type& out_type,
                         const Target& target) {
  return ReduceCompute(attrs, inputs, out_type, target, topi::all);
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
.set_attrs_type_key("relay.attrs.ReduceAttrs")
.set_support_level(4)
.add_type_rel("Reduce", ReduceRel)
.set_attr<FTVMCompute>("FTVMCompute", AllCompute)
.set_attr<TOpPattern>("TOpPattern", kCommReduce);


Array<Tensor> MaxCompute(const Attrs& attrs,
                         const Array<Tensor>& inputs,
                         const Type& out_type,
                         const Target& target) {
  return ReduceCompute(attrs, inputs, out_type, target, topi::max);
}

RELAY_REGISTER_REDUCE_OP("max")
.describe(R"code(Computes the max of array elements over given axes.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.ReduceAttrs")
.set_support_level(4)
.add_type_rel("Reduce", ReduceRel)
.set_attr<FTVMCompute>("FTVMCompute", MaxCompute)
.set_attr<TOpPattern>("TOpPattern", kCommReduce);


Array<Tensor> MinCompute(const Attrs& attrs,
                         const Array<Tensor>& inputs,
                         const Type& out_type,
                         const Target& target) {
  return ReduceCompute(attrs, inputs, out_type, target, topi::min);
}


RELAY_REGISTER_REDUCE_OP("min")
.describe(R"code(Computes the min of array elements over given axes.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.ReduceAttrs")
.set_support_level(4)
.add_type_rel("Reduce", ReduceRel)
.set_attr<FTVMCompute>("FTVMCompute", MinCompute)
.set_attr<TOpPattern>("TOpPattern", kCommReduce);


Array<Tensor> ProdCompute(const Attrs& attrs,
                          const Array<Tensor>& inputs,
                          const Type& out_type,
                          const Target& target) {
  return ReduceCompute(attrs, inputs, out_type, target, topi::prod);
}

RELAY_REGISTER_REDUCE_OP("prod")
.describe(R"code(Computes the products of array elements over given axes.

Example::

  data = [[[1,2],[2,3],[1,3]],
          [[1,4],[4,3],[5,2]],
          [[7,1],[7,2],[7,3]]]

  mean(data, axis=1)
  [35562240]

  mean(data, axis=[1,2])
  [ 36  480  2058]

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.ReduceAttrs")
.set_support_level(4)
.add_type_rel("Reduce", ReduceRel)
.set_attr<FTVMCompute>("FTVMCompute", ProdCompute)
.set_attr<TOpPattern>("TOpPattern", kCommReduce);


Array<Tensor> MeanCompute(const Attrs& attrs,
                          const Array<Tensor>& inputs,
                          const Type& out_type,
                          const Target& target) {
  IndexExpr count = make_const(inputs[0]->dtype, 1);
  const ReduceAttrs* param = attrs.as<ReduceAttrs>();
  CHECK(param != nullptr);
  auto axes = param->axis;
  for (int64_t i : GetReduceAxes(inputs[0]->shape.size(),
                                 param->axis,
                                 param->exclude)) {
    count *= inputs[0]->shape[i];
  }
  auto res = ReduceCompute(attrs, inputs, out_type, target, topi::sum);
  return {topi::divide(res[0], count)};
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
.set_attrs_type_key("relay.attrs.ReduceAttrs")
.set_support_level(4)
.add_type_rel("Reduce", ReduceRel)
.set_attr<FTVMCompute>("FTVMCompute", MeanCompute)
.set_attr<TOpPattern>("TOpPattern", kCommReduce);
}  // namespace relay
}  // namespace tvm
