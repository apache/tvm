/*!
 *  Copyright (c) 2018 by Contributors
 * \file reduce.cc
 * \brief Reduction operators.
 */
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
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

inline std::vector<IndexExpr> GetReduceAxes(const uint32_t indim,
                                            const std::vector<IndexExpr>& axis,
                                            bool exclude,
                                            const TypeReporter& reporter) {
  if (axis.size() == 0) {
    std::vector<IndexExpr> r_axes;
    r_axes.push_back(make_const(tvm::Int(64), 0));
    return r_axes;
  }
  auto in_dim = make_const(tvm::Int(64), indim);
  CHECK(reporter->Assert(axis[axis.size() - 1] < in_dim))
    << "Reduction axis " << axis[axis.size() - 1]
    << " exceeds input dimensions " << indim;

  std::vector<IndexExpr> in_axis = axis;
  auto zero = make_zero(tvm::Int(64));
  for (auto &i : in_axis) {
    if (reporter->Assert(i < zero)) {
      i = i + in_dim;
    }
    CHECK(reporter->Assert(i >= zero))
      << "axis out of bounds in reduce operator";
    CHECK(reporter->Assert(i < make_const(tvm::Int(64), indim)))
      << "axis out of bounds in reduce operator";
  }

  std::sort(in_axis.begin(),
            in_axis.end(),
            [](IndexExpr lhs, IndexExpr rhs){
                if (const int64_t* pdiff = as_const_int(lhs - rhs)) {
                  return(pdiff[0] < 0);
                }
                return false;
              });

  if (!exclude) {
    return in_axis;
  }

  auto r_size = indim - in_axis.size();
  std::vector<IndexExpr> r_axis(r_size);

  for (uint32_t i = 0, j = 0, k = 0; i < indim; ++i) {
    auto val = make_const(tvm::Int(64), i);
    if (j < in_axis.size() && reporter->AssertEQ(in_axis[j], val)) {
        ++j;
        continue;
    }
    r_axis[k++] = val;
  }
  return r_axis;
}

inline std::vector<IndexExpr> ReduceShapeImpl(const std::vector<IndexExpr> &in_shape,
                                              const std::vector<IndexExpr> &in_axis,
                                              bool keepdims,
                                              bool exclude,
                                              const TypeReporter& reporter) {
  uint32_t indim = in_shape.size();
  auto r_axes = GetReduceAxes(indim, in_axis, exclude, reporter);
  if (!r_axes.size()) return in_shape;
  if (r_axes.size() == indim) {
    auto dim = 1;
    if (keepdims) {
      dim = indim;
      }
    return std::vector<IndexExpr>(dim);
  }
  CHECK(r_axes.size() < indim);
  if (keepdims) {
    std::vector<IndexExpr> oshape(in_shape);
    for (unsigned i = 0, j = 0; i < indim; ++i) {
      auto val = make_const(tvm::Int(64), i);
      if (j >= r_axes.size() || !(reporter->AssertEQ(r_axes[j], val))) continue;
      oshape[i] = 1;
      ++j;
    }
    return oshape;
  }

  auto osize = indim - r_axes.size();
  std::vector<IndexExpr> oshape(osize);
  for (unsigned i = 0, j = 0, k = 0; i < indim; ++i) {
    auto val = make_const(tvm::Int(64), i);
    if (j < r_axes.size() && (reporter->AssertEQ(r_axes[j], val))) {
      ++j;
      continue;
    }
    oshape[k++] = in_shape[i];
  }
  return oshape;
}

bool ReduceRel(const Array<Type>& types,
               int num_inputs,
               const Attrs& attrs,
               const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  CHECK(static_cast<int>(data->shape.size()) != 0);

  const ReduceAttrs* param = attrs.as<ReduceAttrs>();
  CHECK(param != nullptr);

  std::vector<IndexExpr> in_shape;
  for (auto i : data->shape) {
    in_shape.push_back(i);
  }

  std::vector<IndexExpr> in_axis;
  for (auto i : param->axis) {
    in_axis.push_back(i);
  }

  // assign output type
  auto oshape = ReduceShapeImpl(in_shape,
                                in_axis,
                                param->keepdims,
                                param->exclude,
                                reporter);
  reporter->Assign(types[1], TensorTypeNode::make(oshape, data->dtype));

  return true;
}

// Positional relay function to create reduce operator used by frontend FFI.
Expr MakeReduce(Expr data,
                Array<IndexExpr> axis,
                bool keepdims,
                bool exclude) {
  auto attrs = make_node<ReduceAttrs>();
  attrs->axis = std::move(axis);
  attrs->keepdims = keepdims;
  attrs->exclude = exclude;
  static const Op& op = Op::Get("argmax");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

#define RELAY_REGISTER_REDUCE_OP(OpName)                           \
  TVM_REGISTER_API("relay.op._make." OpName)                       \
  .set_body([](const TVMArgs& args, TVMRetValue* rv) {             \
      runtime::detail::unpack_call<Expr, 4>(MakeReduce, args, rv); \
    });                                                            \
  RELAY_REGISTER_OP(OpName)                                        \
  .set_num_inputs(1)                                              \
  .add_argument("data", "Tensor", "The input tensor.")            \
  .add_type_rel("Reduce", ReduceRel)


RELAY_REGISTER_REDUCE_OP("argmax")
.describe(R"code(Creates an operation that finds the indices of the maximum
values over a given axis.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_support_level(4);


RELAY_REGISTER_REDUCE_OP("argmin")
.describe(R"code(Creates an operation that finds the indices of the minimum
values over a given axis.

)code" TVM_ADD_FILELINE)
.set_num_inputs(1)
.set_support_level(4);

}  // namespace relay
}  // namespace tvm
