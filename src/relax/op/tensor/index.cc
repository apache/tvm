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
 * \file index.cc
 * \brief indexing operators.
 */

#include "index.h"

#include <tvm/relax/analysis.h>
#include <tvm/topi/transform.h>

#include <algorithm>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

/* relax.take */
TVM_REGISTER_NODE_TYPE(TakeAttrs);

Expr take(Expr x, Expr indices, Optional<Integer> axis) {
  ObjectPtr<TakeAttrs> attrs = make_object<TakeAttrs>();
  attrs->axis = std::move(axis);

  static const Op& op = Op::Get("relax.take");
  return Call(op, {std::move(x), std::move(indices)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.take").set_body_typed(take);

StructInfo InferStructInfoTake(const Call& call, const BlockBuilder& ctx) {
  CheckNumArguments(call, ctx);
  TensorStructInfo data_sinfo = GetInputTensorStructInfo(call, 0, ctx);

  // StructInfo inference when the index is a PrimValue is equivalent
  // to that of a scalar (0-d) tensor.
  TensorStructInfo indices_sinfo = [&]() {
    auto arg = call->args[1];
    auto sinfo = GetStructInfo(arg);
    if (auto tensor_sinfo = sinfo.as<TensorStructInfo>()) {
      return tensor_sinfo.value();
    } else if (auto prim_sinfo = sinfo.as<PrimStructInfoNode>()) {
      return TensorStructInfo(ShapeExpr(Array<PrimExpr>{}), prim_sinfo->dtype);
    } else {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "Operator " << call->op << " requires the indices argument to be "
                       << "either a tensor or a scalar value.  "
                       << "However, argument " << arg << " has struct info " << sinfo);
      // Unreachable, but [[noreturn]] attribute on virtual function
      // `ReportFatal` is insufficient to silence -Wreturn-type, as
      // child class might not be [[noreturn]].
      return TensorStructInfo();
    }
  }();

  if (indices_sinfo->IsUnknownDtype()) {
    // TODO(tvm-team): Do we have an equivalent of `ctx->ReportFatal` for warning?
    LOG(WARNING) << "Data type of indice has not been specified. Assume it has an integer type.";
  } else if (!(indices_sinfo->dtype.is_int() || indices_sinfo->dtype.is_uint())) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Take op requires the input indices to have integer dtype. However, the "
                        "given indices dtype is "
                     << indices_sinfo->dtype);
  }

  const auto* attrs = call->attrs.as<TakeAttrs>();
  if (!attrs->axis.defined() && data_sinfo->ndim != 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Take op expects the input data to be 1-dimensional tensor when the axis "
                        "is not specified. However, the given data tensor has ndim "
                     << data_sinfo->ndim);
  }
  if (data_sinfo->IsUnknownNdim() || indices_sinfo->IsUnknownNdim()) {
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim, data_sinfo->vdevice);
  }

  int axis = attrs->axis.defined()
                 ? NormalizeAxis(call, ctx, data_sinfo->ndim, attrs->axis.value()->value)
                 : 0;
  const auto* data_shape = data_sinfo->shape.as<ShapeExprNode>();
  const auto* indices_shape = indices_sinfo->shape.as<ShapeExprNode>();
  if (data_shape == nullptr || indices_shape == nullptr) {
    return TensorStructInfo(data_sinfo->dtype, indices_sinfo->ndim + data_sinfo->ndim - 1,
                            data_sinfo->vdevice);
  }

  Array<PrimExpr> output_shape;
  for (int i = 0; i < data_sinfo->ndim; i++) {
    if (i == axis) {
      for (int j = 0; j < indices_sinfo->ndim; j++)
        output_shape.push_back(indices_shape->values[j]);
    } else {
      output_shape.push_back(data_shape->values[i]);
    }
  }
  return TensorStructInfo(ShapeExpr(output_shape), data_sinfo->dtype, data_sinfo->vdevice);
}

TVM_REGISTER_OP("relax.take")
    .set_attrs_type<TakeAttrs>()
    .set_num_inputs(2)
    .add_argument("x", "Tensor", "The source tensor.")
    .add_argument("indices", "Tensor", "The indices of the values to extract.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoTake)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.strided_slice */
TVM_REGISTER_NODE_TYPE(StridedSliceAttrs);

Expr strided_slice(Expr x, Expr axes, Expr begin, Expr end, Optional<Expr> strides,
                   bool assume_inbound) {
  // Initial validation of the arguments.  A more complete validation
  // will be done when inferring the StructInfo, but that requires the
  // StructInfo of all arguments to be populated.

  std::optional<std::tuple<const char*, size_t, Expr>> known_length;
  auto check_tuple = [&known_length](const char* name, Expr expr) {
    if (const auto* tuple = expr.as<TupleNode>()) {
      size_t length = tuple->fields.size();
      if (known_length.has_value()) {
        const auto& prev = known_length.value();
        CHECK_EQ(length, std::get<size_t>(prev))
            << "The strided_slice operator requires that "
            << "the axes, begin, end, and strides tuples are all the same length.  "
            << "However, the " << std::get<const char*>(prev) << " argument ("
            << std::get<Expr>(prev) << ") has " << std::get<size_t>(prev) << " elements, while the "
            << name << " argument (" << expr << ") has " << length << " elements.";
      } else {
        known_length = std::tuple{name, length, expr};
      }
    }
  };
  check_tuple("axes", axes);
  check_tuple("begin", begin);
  check_tuple("end", end);
  if (strides.defined()) check_tuple("strides", strides.value());

  ObjectPtr<StridedSliceAttrs> attrs = make_object<StridedSliceAttrs>();
  attrs->assume_inbound = assume_inbound;

  Array<Expr> args = {x, axes, begin, end};
  if (strides.defined()) {
    args.push_back(strides.value());
  }

  static const Op& op = Op::Get("relax.strided_slice");
  auto call = Call(op, args, Attrs(attrs));

  return call;
}

TVM_REGISTER_GLOBAL("relax.op.strided_slice").set_body_typed(strided_slice);

/* \brief Helper function to unpack a relax::Tuple
 *
 * A `relax::Tuple` may be provided to an operator as an in-line
 * expression, as a variable bound to known tuple within the current
 * function, as a function argument, etc.  The StructInfo of the tuple
 * tracks the known values of any `PrimValue` elements, but it can be
 * tedious to extract.  This utility extracts the `PrimExpr` contents
 * of a `relax::Tuple`.
 *
 * If the StructInfo cannot contain a tuple of the type specified,
 * this function will throw an exception.  (e.g. Attempting to extract
 * a tuple from a `TensorStructInfo`.)
 *
 * \tparam PrimType The subtype of PrimExpr to extract.  For example,
 *     extracting an `Array<Integer>`
 *
 * \param sinfo The StructInfo to inspect
 *
 * \returns An array of the `PrimType`, if it can be extracted.
 *     Otherwise, `NullOpt`.
 */
template <typename PrimType = PrimExpr,
          typename = std::enable_if_t<std::is_base_of_v<PrimExpr, PrimType>>>
Optional<Array<PrimType>> UnpackTupleOfPrimValue(Optional<StructInfo> sinfo) {
  if (!sinfo) return NullOpt;

  // An ObjectStructInfo may contain a tuple of the desired type, but
  // it isn't yet known whether it does.  Return early, as we cannot
  // provide a known `Array<PrimType>` to the caller.
  if (sinfo.as<ObjectStructInfoNode>()) return NullOpt;

  auto tuple = sinfo.as<TupleStructInfoNode>();
  CHECK(tuple) << "TypeError: "
               << "The struct info " << sinfo << " cannot contain a tuple whose elements are "
               << PrimType::ContainerType::_type_key;

  Array<PrimType> output;
  for (size_t i = 0; i < tuple->fields.size(); i++) {
    auto field = tuple->fields[i];

    if (field.as<ObjectStructInfoNode>()) return NullOpt;

    auto prim_sinfo = field.as<PrimStructInfoNode>();
    CHECK(prim_sinfo) << "TypeError: "
                      << "The struct info " << sinfo
                      << " cannot contain a tuple whose elements are "
                      << PrimType::ContainerType::_type_key << ", because element " << i
                      << " has struct info " << field;

    if (!prim_sinfo->value.defined()) return NullOpt;

    Optional<PrimType> element = prim_sinfo->value.as<PrimType>();
    if (!element) return NullOpt;

    output.push_back(element.value());
  }
  return output;
}

/* \brief Helper function to unpack a relax::Tuple
 *
 * A `relax::Tuple` may be provided to an operator as an in-line
 * expression, as a variable bound to known tuple within the current
 * function, as a function argument, etc.  The StructInfo of the tuple
 * tracks the known values of any `PrimValue` elements, but it can be
 * tedious to extract.  This utility extracts the `PrimExpr` contents
 * of a `relax::Tuple`.
 *
 * If the StructInfo cannot contain a tuple of the type specified,
 * this function will throw an exception.  (e.g. Attempting to extract
 * a tuple from a `TensorStructInfo`.)
 *
 * \tparam PrimType The subtype of PrimExpr to extract.  For example,
 *     extracting an `Array<Integer>`
 *
 * \param expr The `relax::Expr` to inspect
 *
 * \returns An array of the `PrimType`, if it can be extracted.
 *     Otherwise, `NullOpt`.
 */
template <typename PrimType = PrimExpr,
          typename = std::enable_if_t<std::is_base_of_v<PrimExpr, PrimType>>>
Optional<Array<PrimType>> UnpackTupleOfPrimValue(Optional<Expr> expr) {
  if (expr) {
    return UnpackTupleOfPrimValue<PrimType>(GetStructInfo(expr.value()));
  } else {
    return NullOpt;
  }
}

StructInfo InferStructInfoStridedSlice(const Call& call, const BlockBuilder& ctx) {
  size_t n_args = call->args.size();
  CHECK(4 <= n_args && n_args <= 5)
      << "Operator " << call->op << " accepts either three arguments (data, axes, begin, end) "
      << " or four arguments (data, axes, begin, end, strides), "
      << "but received " << n_args << " in expression " << call;

  Expr data = call->args[0];
  Expr axes = call->args[1];
  Expr begin = call->args[2];
  Expr end = call->args[3];
  Optional<Expr> strides = [&]() -> Optional<Expr> {
    if (n_args > 4) {
      return call->args[4];
    } else {
      return NullOpt;
    }
  }();

  auto axes_sinfo = GetStructInfo(call->args[1]);
  auto begin_sinfo = GetStructInfo(call->args[2]);
  auto end_sinfo = GetStructInfo(call->args[3]);
  auto strides_sinfo = [&]() -> Optional<StructInfo> {
    if (n_args > 4) {
      return GetStructInfo(call->args[4]);
    } else {
      return NullOpt;
    }
  }();

  CHECK(IsBaseOf(relax::TensorStructInfo(DataType::Void(), kUnknownNDim), GetStructInfo(data)))
      << "Operator " << call->op << " requires the first argument to be a tensor.  "
      << "However, in expression " << call << ", the first argument " << data << " has struct info "
      << GetStructInfo(data);

  // TODO(Lunderberg): Implement this check using `IsBaseOf`.  Doing
  // so will require a way to represent a `relax::TupleStructInfo` of
  // unknown length, where each element has the same `StructInfo`.
  auto is_base_of_tuple_of_int64 = [&](const StructInfo& sinfo) -> bool {
    if (sinfo.as<ObjectStructInfoNode>()) {
      return true;
    }

    const auto* tuple = sinfo.as<TupleStructInfoNode>();
    if (!tuple) return false;

    return std::all_of(tuple->fields.begin(), tuple->fields.end(), [](const StructInfo& field) {
      return IsBaseOf(relax::PrimStructInfo(DataType::Int(64)), field);
    });
  };
  auto check_tuple = [&](const char* name, Expr expr) {
    auto sinfo = GetStructInfo(expr);

    CHECK(is_base_of_tuple_of_int64(sinfo)) << "Operator " << call->op << " requires the " << name
                                            << " argument to be a tuple of int64 PrimValues.  "
                                            << "However, in expression " << call << ", the " << name
                                            << " argument " << expr << " has struct info " << sinfo;
  };
  check_tuple("axes", call->args[1]);
  check_tuple("begin", call->args[2]);
  check_tuple("end", call->args[3]);
  if (call->args.size() > 4) {
    check_tuple("strides", call->args[4]);
  }

  const auto* data_sinfo = data->struct_info_.as<TensorStructInfoNode>();

  DataType dtype = DataType::Void();
  Optional<VDevice> vdevice = NullOpt;
  int ndim = kUnknownNDim;
  if (data_sinfo) {
    dtype = data_sinfo->dtype;
    vdevice = data_sinfo->vdevice;
    ndim = data_sinfo->ndim;
  }

  Optional<Expr> shape = [&]() -> Optional<Expr> {
    if (!data_sinfo) return NullOpt;
    if (!data_sinfo->shape) return NullOpt;

    auto opt_axes_tuple = UnpackTupleOfPrimValue<Integer>(axes);
    if (!opt_axes_tuple) return NullOpt;
    auto axes_tuple = opt_axes_tuple.value();

    auto opt_begin_tuple = UnpackTupleOfPrimValue(begin);
    if (!opt_begin_tuple) return NullOpt;
    auto begin_tuple = opt_begin_tuple.value();

    CHECK_EQ(axes_tuple.size(), begin_tuple.size())
        << "For operator " << call->op << ", "
        << "the number of axes provided must match the number of 'begin' indices.  "
        << "However, there are " << axes_tuple.size() << " axes specified (" << axes_tuple
        << ") and " << begin_tuple.size() << " 'begin' indices specified (" << begin_tuple << ")";

    auto opt_end_tuple = UnpackTupleOfPrimValue(end);
    if (!opt_end_tuple) return NullOpt;
    auto end_tuple = opt_end_tuple.value();

    CHECK_EQ(axes_tuple.size(), end_tuple.size())
        << "For operator " << call->op << ", "
        << "the number of axes provided must match the number of 'end' indices.  "
        << "However, there are " << axes_tuple.size() << " axes specified (" << axes_tuple
        << ") and " << end_tuple.size() << " 'end' indices specified (" << end_tuple << ")";

    Array<PrimExpr> strides_tuple;
    if (strides.defined()) {
      auto opt_strides_tuple = UnpackTupleOfPrimValue(strides);
      if (!opt_strides_tuple) return NullOpt;

      strides_tuple = opt_strides_tuple.value();
    } else {
      strides_tuple = Array<PrimExpr>(axes_tuple.size(), IntImm(DataType::Int(64), 1));
    }

    CHECK_EQ(axes_tuple.size(), strides_tuple.size())
        << "For operator " << call->op << ", "
        << "when the optional 'strides' argument is provided, "
        << "the number of axes provided must match the number of strides provided.  "
        << "However, there are " << axes_tuple.size() << " axes specified (" << axes_tuple
        << ") and " << strides_tuple.size() << " strides specified (" << strides_tuple << ")";

    auto opt_data_shape = data_sinfo->GetShape();

    if (axes_tuple.empty() && !opt_data_shape.defined()) {
      return data_sinfo->shape.value();
    } else if (!opt_data_shape.defined()) {
      return NullOpt;
    }

    std::vector<int> axes = NormalizeAxes(call, ctx, data_sinfo->ndim, axes_tuple);
    auto attrs = call->attrs.as<StridedSliceAttrs>();

    Array<PrimExpr> output_shape = data_sinfo->GetShape().value();
    for (size_t i = 0; i < axes.size(); i++) {
      size_t axis = axes[i];
      PrimExpr input_dim = output_shape[axis];
      PrimExpr begin = begin_tuple[i];
      PrimExpr end = end_tuple[i];

      PrimExpr output_dim =
          topi::GetLength(begin, end, strides_tuple[i], input_dim, attrs->assume_inbound);

      arith::Analyzer* analyzer = ctx->GetAnalyzer();
      std::optional<With<arith::ConstraintContext>> context;
      if (attrs->assume_inbound) {
        context.emplace(analyzer, 0 <= begin && begin <= input_dim && 0 <= end && end <= input_dim);
      }

      output_dim = analyzer->Simplify(output_dim);

      output_shape.Set(axis, output_dim);
    }
    return ShapeExpr(output_shape);
  }();

  if (shape.defined()) {
    return TensorStructInfo(shape.value(), dtype, vdevice);
  } else {
    return TensorStructInfo(dtype, ndim, vdevice);
  }
}

InferLayoutOutput InferLayoutStridedSlice(const Call& call,
                                          const Map<String, Array<String>>& desired_layouts,
                                          const VarLayoutMap& var_layout_map) {
  ICHECK(NoDesiredLayout(call, desired_layouts));

  const auto* attrs = call->attrs.as<StridedSliceAttrs>();
  ICHECK(attrs != nullptr) << "Invalid Call";

  const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  CHECK(tensor_sinfo) << "Invalid Call";
  CHECK(!tensor_sinfo->IsUnknownNdim()) << "Layout inference only supports known dimensionality, "
                                        << "but expression " << call << " has argument "
                                        << call->args[0] << " of unknown dimensionality.";
  LayoutDecision existing_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  // Can't handle sub indexed layouts.
  if (existing_layout->layout.ndim() != existing_layout->layout.ndim_primal()) {
    existing_layout = LayoutDecision(InitialLayout(tensor_sinfo->ndim));
  }

  auto opt_axes_tuple = UnpackTupleOfPrimValue<Integer>(GetStructInfo(call->args[1]));
  CHECK(opt_axes_tuple) << "Layout inference of " << call->op
                        << " requires slices to be along static axes.  "
                        << "However, expression " << call << " slices along non-static axes "
                        << call->args[1];
  Array<Integer> axes_tuple = opt_axes_tuple.value();

  Array<Expr> new_axes;
  for (const auto& axis : axes_tuple) {
    int new_axis = FindAxis(existing_layout->layout, axis->value);
    new_axes.push_back(relax::PrimValue::Int64(new_axis));
  }

  return InferLayoutOutput({existing_layout}, {existing_layout}, call->attrs,
                           {{1, relax::Tuple(new_axes)}});
}

TVM_REGISTER_OP("relax.strided_slice")
    .set_attrs_type<StridedSliceAttrs>()
    .set_num_inputs(1)
    .add_argument("x", "Tensor", "The source tensor to be sliced.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoStridedSlice)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutStridedSlice)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow)
    .set_attr<Bool>("FPurity", Bool(true));

/* relax.dynamic_strided_slice */
Expr dynamic_strided_slice(Expr x,      //
                           Expr begin,  //
                           Expr end,    //
                           Expr strides) {
  static const Op& op = Op::Get("relax.dynamic_strided_slice");
  return Call(op, {std::move(x), std::move(begin), std::move(end), std::move(strides)}, {});
}

TVM_REGISTER_GLOBAL("relax.op.dynamic_strided_slice").set_body_typed(dynamic_strided_slice);

StructInfo InferStructInfoDynStridedSlice(const Call& call, const BlockBuilder& ctx) {
  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* begin_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);
  const auto* end_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[2]);
  const auto* strides_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[3]);

  ICHECK(data_sinfo);
  if (data_sinfo->IsUnknownNdim()) {
    LOG(WARNING) << "When data rank is unknown, dynamic strided slice assumes begin/end/strides "
                    "tensors are well-formed. It could produce runtime error when this assumption "
                    "turns out to be wrong.";
    return TensorStructInfo(data_sinfo->dtype, kUnknownNDim, data_sinfo->vdevice);
  }
  if (data_sinfo->IsUnknownDtype()) {
    LOG(WARNING) << "When data type is unknown, dynamic strided slice assumes to have a valid "
                    "dtype. It could produce runtime error when this assumption "
                    "turns out to be wrong.";
  }

  int n_axis = data_sinfo->ndim;
  auto diag_def = [&](const TensorStructInfoNode* sinfo, String name) {
    ICHECK(sinfo) << "Dynamic strided slice requires the input " << name
                  << " to be have the struct info. Please try normalizing the inputs.";
    CHECK_EQ(sinfo->ndim, 1) << "Dynamic strided slice requires " << name
                             << " to be 1d tensor (list of values).";
    const auto* shape = sinfo->shape.as<ShapeExprNode>();
    ICHECK(shape) << "Dynamic strided slice requires the input " << name
                  << " to have well-defined shape.";
    // NOTE(tvm-team): This strong restriction seems necessary for now until we have a generic
    // solution in converting 1d Tensor with unknown num_elem to Array<PrimExpr>.
    const auto* num_elem = shape->values[0].as<IntImmNode>();
    ICHECK(num_elem) << "Dynamic strided slice requires the input " << name
                     << " to have a known integer shape value.";
    CHECK_EQ(num_elem->value, n_axis) << "Dynamic strided slice requires the number of indices in "
                                      << name << " to equal the number of axes.";
    if (sinfo->IsUnknownDtype()) {
      LOG(WARNING) << "Dynamic strided slice assumes " << name
                   << " to be int64 when it is not specified.";
    } else {
      CHECK(sinfo->dtype == DataType::Int(64))
          << "Dynamic strided_slice expects the input " << name
          << "values to be all int64. However, " << name << " has dtype " << sinfo->dtype << ".";
    }
  };
  diag_def(begin_sinfo, "begin");
  diag_def(end_sinfo, "end");
  diag_def(strides_sinfo, "strides");

  // The output shape will depend on the runtime value in begin/end/strides tensors.
  // TODO(tvm-team): Currently, it is unable to express partially-static shape. Revisit when
  // PrimValue lands.
  return TensorStructInfo(data_sinfo->dtype, n_axis, data_sinfo->vdevice);
}

// TODO(tvm-team): Register FRelaxInferLayout, TMixedPrecisionPolicy
TVM_REGISTER_OP("relax.dynamic_strided_slice")
    .set_num_inputs(4)
    .add_argument("x", "Tensor", "The source tensor to be sliced.")
    .add_argument("begin", "Tensor", "The indices to begin with in the slicing.")
    .add_argument("end", "Tensor", "Indices indicating end of the slice.")
    .add_argument("strides", "Tensor", "The stride values.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoDynStridedSlice)
    .set_attr<Bool>("FPurity", Bool(true));

}  // namespace relax
}  // namespace tvm
