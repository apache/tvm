/*!
 *  Copyright (c) 2018 by Contributors
 * \file reduce.cc
 * \brief Reduction operators.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <numeric>
#include <limits>
#include "../type_relations.h"

namespace tvm {
namespace relay {

/*! \brief Attributes for Reduce operators */
struct ReduceAttrs : public tvm::AttrsNode<ReduceAttrs> {
  Array<IndexExpr> axis;
  bool keepdims;
  bool exclude;

  TVM_DECLARE_ATTRS(ReduceAttrs, "relay.attrs.ReduceAttrs") {
    TVM_ATTR_FIELD(axis).set_default(Array<IndexExpr>({}))
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
                                          const Array<IndexExpr>& inaxis,
                                          bool exclude) {
  if (!inaxis.defined()) {
    std::vector<int64_t> r_axes(indim);
    std::iota(r_axes.begin(), r_axes.end(), 0);
    return r_axes;
  }

  std::vector<int64_t> in_axes;
  for (auto i : inaxis) {
    const int64_t* k = as_const_int(i);
    CHECK(k != nullptr) << "Reduce axis need to be constant, cannot be symbolic";
    int64_t axis = k[0];
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
  std::vector<IndexExpr> in_shape;
  for (auto i : data->shape) {
    in_shape.push_back(i);
  }

  const ReduceAttrs* param = attrs.as<ReduceAttrs>();
  CHECK(param != nullptr);

  // assign output type and shape
  auto oshape = ReduceShapeImpl(in_shape, param, reporter);
  reporter->Assign(types[1], TensorTypeNode::make(oshape, Int(32)));
  return true;
}


#define RELAY_REGISTER_REDUCE_OP(OpName)                           \
  TVM_REGISTER_API("relay.op._make." OpName)                       \
  .set_body([](const TVMArgs& args, TVMRetValue* rv) {             \
    auto make_func = [](Expr data,                                 \
                        Array<IndexExpr> axis,                     \
                        bool keepdims,                             \
                        bool exclude) {                            \
      auto attrs = make_node<ReduceAttrs>();                       \
      attrs->axis = std::move(axis);                               \
      attrs->keepdims = keepdims;                                  \
      attrs->exclude = exclude;                                    \
      static const Op& op = Op::Get(OpName);                       \
      return CallNode::make(op, {data}, Attrs(attrs), {});         \
    };                                                             \
    runtime::detail::unpack_call<Expr, 4>(make_func, args, rv);    \
    });                                                            \
  RELAY_REGISTER_OP(OpName)                                        \
  .set_num_inputs(1)                                               \
  .add_argument("data", "Tensor", "The input tensor.")


RELAY_REGISTER_REDUCE_OP("argmax")
.describe(R"code(Creates an operation that finds the indices of the maximum
values over a given axis.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_support_level(4)
.add_type_rel("ArgReduce", ArgReduceRel);


RELAY_REGISTER_REDUCE_OP("argmin")
.describe(R"code(Creates an operation that finds the indices of the minimum
values over a given axis.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_support_level(4)
.add_type_rel("ArgReduce", ArgReduceRel);

}  // namespace relay
}  // namespace tvm
