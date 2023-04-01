#include <tvm/relax/op/manipulate.h>

#include "../arg2relax.h"
#include "../op_common.h"
#include "../relax2te.h"

namespace tvm {
namespace relax {

StructInfo BroadcastToSInfo(const Call& call, const BlockBuilder& ctx) {
  ICHECK_EQ(call->args.size(), 2);
  const auto* data_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  const auto* tgt_shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(call->args[1]);
  if (data_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "broadcast_to requires the input data to be Tensor. However, the given one is "
        << call->args[0]->struct_info_);
  }
  if (tgt_shape_sinfo == nullptr) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << "broadcast_to requires the input new shape to be Shape. However, the given one is "
        << call->args[1]->struct_info_);
  }
  if (!data_sinfo->IsUnknownNdim() && !tgt_shape_sinfo->IsUnknownNdim() &&
      tgt_shape_sinfo->ndim < data_sinfo->ndim) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "broadcast_to expects the input shape to have the number of ndim at least "
                        "as the input tensor's. However, the given tensor has ndim "
                     << data_sinfo->ndim << " while the target shape has ndim "
                     << tgt_shape_sinfo->ndim);
  }
  // Trust the input target shape when there is no possibility to do any compile-time check.
  if (!data_sinfo->shape.defined()) {
    return TensorStructInfo(/*shape=*/call->args[1], data_sinfo->dtype);
  }
  ShapeStructInfo shape_sinfo = Downcast<ShapeStructInfo>(data_sinfo->shape.value()->struct_info_);
  if (!shape_sinfo->values.defined() || !tgt_shape_sinfo->values.defined()) {
    return TensorStructInfo(/*shape=*/call->args[1], data_sinfo->dtype);
  }
  arith::Analyzer* analyzer = ctx->GetAnalyzer();
  Array<PrimExpr> old_shape_value = shape_sinfo->values.value();
  Array<PrimExpr> tgt_shape_value = tgt_shape_sinfo->values.value();
  int old_ndim = old_shape_value.size();
  int tgt_ndim = tgt_shape_value.size();
  for (int i = 0; i < old_ndim; ++i) {
    PrimExpr old_len = old_shape_value[old_ndim - i - 1];
    PrimExpr tgt_len = tgt_shape_value[tgt_ndim - i - 1];
    const auto* old_len_int = old_len.as<IntImmNode>();
    if (old_len_int != nullptr && old_len_int->value == 1) {
      continue;
    } else if (analyzer->CanProve(old_len != tgt_len)) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "broadcast_to failed. Input shape: " << old_shape_value
                       << ". Target shape: " << tgt_shape_value);
    }
    // TODO(relax-team): revisit here for better check on if the tensor length
    // is consistent with the length in the given shape.
  }
  return TensorStructInfo(/*shape=*/call->args[1], data_sinfo->dtype);
}

StructInfo RepeatSInfoFallback(const Call& call, const BlockBuilder& bb) {
  TensorStructInfo a = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
  Optional<IntImm> axis =
      relax2te::OptionalFromOpaque(relax2te::Axis(kUnknownNDim, false, true))(call->args[2]);
  if (axis.defined()) {
    if (!a->IsUnknownNdim()) {
      int ndim = a->ndim;
      int axis_val = axis.value()->value;
      if (axis_val < 0) {
        axis_val += ndim;
      }
      CHECK_LT(axis_val, ndim) << "ValueError: axis out of range, expected in range of [" << -ndim
                               << ", " << ndim << "), but got " << axis.value()->value;
    }
    if (a->shape.defined()) {
      return TensorStructInfo(a->shape.value(), a->dtype);
    } else {
      return TensorStructInfo(a->dtype, a->ndim);
    }
  }
  return TensorStructInfo(a->dtype, 1);
}

StructInfo TileSInfoFallback(const Call& call, const BlockBuilder& bb) {
  TensorStructInfo a = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
  if (a->shape.defined()) {
    return TensorStructInfo(a->shape.value(), a->dtype);
  } else {
    return TensorStructInfo(a->dtype, a->ndim);
  }
}

StructInfo ExpandDimsSInfoFallback(const Call& call, const BlockBuilder& bb) {
  TensorStructInfo a = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
  Array<IntImm> axes = relax2te::Axes(a->ndim, true, !a->IsUnknownNdim())(call->args[1]);
  if (axes.empty() && a->shape.defined()) {
    return TensorStructInfo(a->shape.value(), a->dtype);
  } else {
    std::vector<int64_t> int_axes = topi::array_api::ArrayToVector(axes);
    if (std::set<int64_t>(int_axes.begin(), int_axes.end()).size() != int_axes.size()) {
      LOG(FATAL) << "ValueError: Duplicate axes are not allowed, but got: " << axes;
    }
  }
  return a->IsUnknownNdim() ? TensorStructInfo(a->dtype, kUnknownNDim)
                            : TensorStructInfo(a->dtype, a->ndim + axes.size());
}

StructInfo PermuteDimsSInfoFallback(const Call& call, const BlockBuilder& bb) {
  TensorStructInfo a = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
  Optional<Array<IntImm>> _axes =
      relax2te::OptionalFromOpaque(relax2te::Axes(a->ndim, false, false))(call->args[1]);
  if (_axes.defined()) {
    if (a->IsUnknownNdim()) {
      LOG(FATAL) << "ValueError: Cannot permute dimensions of a tensor with unknown ndim. "
                    "Please provide the target shape explicitly.";
    }
    std::vector<int64_t> axes = topi::array_api::ArrayToVector(_axes.value());
    int ndim = a->ndim;
    bool is_same = true;
    for (int i = 0; i < ndim; ++i) {
      int64_t axis = axes[i];
      CHECK(-ndim <= axis && axis < ndim) << "ValueError: axis must be in range [-ndim, ndim), "
                                             "where ndim is "
                                          << ndim << ", but got: " << _axes;
      if (axis < 0) {
        axes[i] += ndim;
      }
      if (axes[i] != i) {
        is_same = false;
      }
    }
    std::set<int64_t> axis_set(axes.begin(), axes.end());
    for (int i = 0; i < ndim; ++i) {
      CHECK(axis_set.count(i)) << "ValueError: axis must be permutation of [0, ..., ndim), "
                                  "where ndim is "
                               << ndim << ", but got: " << _axes;
    }
    if (is_same && a->shape.defined()) {
      return TensorStructInfo(a->shape.value(), a->dtype);
    }
  }
  return TensorStructInfo(a->dtype, a->ndim);
}

StructInfo SqueezeSInfoFallback(const Call& call, const BlockBuilder& bb) {
  TensorStructInfo a = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
  Optional<Array<IntImm>> axis =
      relax2te::OptionalFromOpaque(relax2te::Axes(a->ndim, false, true))(call->args[1]);
  if (axis.defined() && a->shape.defined() && axis.value().empty()) {
    return TensorStructInfo(a->shape.value(), a->dtype);
  }
  if (a->IsUnknownNdim()) {
    return TensorStructInfo(a->dtype, kUnknownNDim);
  }
  if (!axis.defined()) {
    return TensorStructInfo(a->dtype, kUnknownNDim);
  }
  return TensorStructInfo(a->dtype, a->ndim - axis.value().size());
}

StructInfo ReshapeSInfoFallback(const Call& call, const BlockBuilder& bb) {
  TensorStructInfo a = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
  try {
    Array<PrimExpr> shape = relax2te::ShapeArrayFromShape()(call->args[1]);
    for (const PrimExpr& e : shape) {
      if (const IntImmNode* imm = e.as<IntImmNode>()) {
        if (imm->value == -1) {
          return TensorStructInfo(a->dtype, shape.size());
        }
      }
    }
    return TensorStructInfo(ShapeExpr(shape), a->dtype);
  } catch (const relax2te::NotDerivable& e) {
    return TensorStructInfo(call->args[1], a->dtype);
  }
}

StructInfo FlattenSInfoFallback(const Call& call, const BlockBuilder& bb) {
  TensorStructInfo a = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
  IntImm start_dim = relax2te::Axis(a->ndim, false, true)(call->args[1]);
  IntImm end_dim = relax2te::Axis(a->ndim, false, true)(call->args[2]);
  if (a->IsUnknownNdim()) {
    return (start_dim->value == 0 && end_dim->value == -1)
               ? TensorStructInfo(a->dtype, 1)
               : TensorStructInfo(a->dtype, kUnknownNDim);
  }
  if (a->ndim == 0) {
    IntImm one = IntImm(DataType::Int(64), 1);
    return TensorStructInfo(ShapeExpr({one}), a->dtype);
  }
  if (a->ndim == 1 && a->shape.defined()) {
    return TensorStructInfo(a->shape.value(), a->dtype);
  }
  if (start_dim->value == 0 && end_dim->value == a->ndim - 1) {
    return TensorStructInfo(a->dtype, 1);
  }
  return TensorStructInfo(a->dtype, kUnknownNDim);
}

StructInfo LayoutTransformSInfo(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo x = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
  tir::IndexMap index_map = Downcast<tir::IndexMap>(Downcast<AttrExpr>(call->args[1])->value);
  Optional<PrimExpr> optional_pad_value = relax2te::OptionalFromOpaque(
      relax2te::PrimExprFromPrimValue<relax2te::DTypeFloat>())(call->args[2]);
  if (optional_pad_value.defined()) {
    PrimExpr padded_value = optional_pad_value.value();
    if (padded_value->dtype != x->dtype) {
      ctx->ReportFatal(Diagnostic::Error(call)
                       << "layout_transform pad_value dtype (" << padded_value->dtype
                       << ") and input dtype (" << x->dtype << ") must be the same");
    }
  }
  if (x->IsUnknownNdim()) {
    // Todo(relax-team): revisit here for better check on if the input tensor has desired ndim.
    return TensorStructInfo(x->dtype, /*ndim=*/index_map->final_indices.size());
  }
  if (index_map->initial_indices.size() != static_cast<size_t>(x->ndim)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "number of dimensions in input must match the number of source dimensions "
                        "in index map, but got "
                     << x->ndim << " != " << index_map->initial_indices.size());
  }
  if (!x->shape.defined()) {
    return TensorStructInfo(x->dtype, /*ndim=*/index_map->final_indices.size());
  }
  ShapeStructInfo shape_sinfo = Downcast<ShapeStructInfo>(x->shape.value()->struct_info_);
  if (!shape_sinfo->values.defined()) {
    return TensorStructInfo(x->dtype, /*ndim=*/index_map->final_indices.size());
  }
  Array<PrimExpr> output_shape = index_map->MapShape(shape_sinfo->values.value());
  return TensorStructInfo(ShapeExpr(output_shape), x->dtype);
}

StructInfo SplitSInfoFallback(const Call& call, const BlockBuilder& bb) {
  TensorStructInfo x = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
  ObjectRef indices_or_sections = relax2te::ObjectFromOpaque()(call->args[1]);
  int num_parts = 0;
  if (const auto* _num_parts = indices_or_sections.as<IntImmNode>()) {
    num_parts = _num_parts->value;
  } else {
    Array<PrimExpr> sections = Downcast<Array<PrimExpr>>(indices_or_sections);
    num_parts = sections.size() + 1;
  }
  Array<StructInfo> fields;
  if (num_parts == 1 && x->shape.defined()) {
    if (const auto* shape_expr = x->shape.as<ShapeExprNode>()) {
      fields.push_back(TensorStructInfo(ShapeExpr(shape_expr->values), x->dtype));
    } else {
      fields.push_back(TensorStructInfo(x->shape.value(), x->dtype));
    }
  } else {
    for (int i = 0; i < num_parts; ++i) {
      fields.push_back(TensorStructInfo(x->dtype, x->ndim));
    }
  }
  return TupleStructInfo(fields);
}

StructInfo ConcatSInfoFallback(const Call& call, const BlockBuilder& bb) {
  Array<TensorStructInfo> arrays;
  {
    TupleStructInfo tuple_sinfo = MatchStructInfo<TupleStructInfo>(call->args[0]).value();
    arrays.reserve(tuple_sinfo->fields.size());
    for (const StructInfo& field : tuple_sinfo->fields) {
      arrays.push_back(Downcast<TensorStructInfo>(field));
    }
  }
  CHECK_GT(arrays.size(), 0) << "ValueError: concat expects at least one input";
  // Determine `ndim` and `dtype`
  int n = arrays.size();
  int ndim = kUnknownNDim;
  DataType dtype = arrays[0]->dtype;
  for (const TensorStructInfo& a : arrays) {
    if (a->ndim != kUnknownNDim) {
      if (ndim == kUnknownNDim) {
        ndim = a->ndim;
      } else if (ndim != a->ndim) {
        LOG(FATAL) << "ValueError: concat tensors must have the same ndims, but got: " << arrays;
      }
    }
    if (!dtype.is_void() && !a->IsUnknownDtype()) {
      dtype = (te::placeholder(Array<PrimExpr>{}, dtype)(Array<PrimExpr>{}) +
               te::placeholder(Array<PrimExpr>{}, a->dtype)(Array<PrimExpr>{}))
                  ->dtype;
    } else {
      dtype = DataType::Void();
    }
  }
  if (ndim == kUnknownNDim) {
    return n == 1 ? arrays[0] : TensorStructInfo(dtype, kUnknownNDim);
  }
  int axis = relax2te::Axis(ndim, false, true)(call->args[1]).operator IntImm()->value;
  CHECK_LT(axis, ndim) << "ValueError: axis out of bounds for ndim = " << ndim
                       << ", but got axis = " << axis << ".";
  if (n == 1) {
    return arrays[0];
  }
  Optional<Array<PrimExpr>> lhs_shape;
  for (int i = 1; i < n; ++i) {
    TensorStructInfo a = arrays[i];
    Optional<Array<PrimExpr>> rhs_shape = a->GetShape();
    if (!rhs_shape.defined()) {
      continue;
    }
    if (!lhs_shape.defined()) {
      lhs_shape = rhs_shape;
      continue;
    }
    Array<PrimExpr> lhs = lhs_shape.value();
    Array<PrimExpr> rhs = rhs_shape.value();
    ICHECK_EQ(lhs.size(), ndim);
    ICHECK_EQ(rhs.size(), ndim);
    for (int dim = 0; dim < ndim; ++dim) {
      if (dim == axis) {
        continue;
      }
      arith::Analyzer* analyzer = bb->GetAnalyzer();
      if (analyzer->CanProve(lhs[dim] != rhs[dim])) {
        LOG(FATAL) << "ValueError: all the input array dimensions except for the concatenation "
                      "axis must match exactly, but got: "
                   << arrays;
      }
    }
  }
  return TensorStructInfo(dtype, ndim);
}

// (TVM-TOOL) cc_op begin def/manipulate/*
relax::Call broadcast_to(relax::Expr x, relax::Expr shape) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.broadcast_to");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(shape, shape, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.broadcast_to").set_body_typed(broadcast_to);
TVM_RELAX_REGISTER_OP("broadcast_to")
    .set_attr<FInferStructInfo>("FInferStructInfo", BroadcastToSInfo)
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.broadcast_to");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV x = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("x", DataType::Float(32), true),
                                                         {})(call->args[0]);
        _RV shape = ShapeArrayFromShape()(call->args[1]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, shape}));
        return _h.EmitTE(_bb, "broadcast_to", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call concat(Array<relax::Expr> x, int64_t axis) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.concat");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(ArrayToTuple(nullptr, {})(x), x, _args);
  TVM_RELAX_OP_ARG_CHECK(Axis()(axis), axis, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.concat").set_body_typed(concat);
TVM_RELAX_REGISTER_OP("concat")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
                                  using namespace relax::relax2te;
                                  using _RV = runtime::TVMRetValue;
                                  static const auto* _te =
                                      runtime::Registry::Get("topi.array_api.concat");
                                  ICHECK(_te != nullptr);
                                  TensorHandler _h;
                                  try {
                                    _RV x = ArrayFromTupleTensor(_h.TupleTensor(
                                        "x", DataType::Float(32), true))(call->args[0]);
                                    _RV axis =
                                        Axis(_TupleTensorNDim(x), false, true)(call->args[1]);
                                    return _h.AddOutput(CallGlobalFunc(_te, {x, axis}));
                                  } catch (const NotDerivable& e) {
                                    return ConcatSInfoFallback(call, _bb);
                                  }
                                })
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.concat");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV x = ArrayFromTupleTensor(_h.TupleTensor("x", DataType::Float(32), true))(call->args[0]);
        _RV axis = Axis(_TupleTensorNDim(x), false, true)(call->args[1]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, axis}));
        return _h.EmitTE(_bb, "concat", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call expand_dims(relax::Expr x, Array<IntImm> axis) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.expand_dims");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(Axes()(axis), axis, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.expand_dims").set_body_typed(expand_dims);
TVM_RELAX_REGISTER_OP("expand_dims")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
                                  using namespace relax::relax2te;
                                  using _RV = runtime::TVMRetValue;
                                  static const auto* _te =
                                      runtime::Registry::Get("topi.array_api.expand_dims");
                                  ICHECK(_te != nullptr);
                                  TensorHandler _h;
                                  try {
                                    _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                                        _h.Tensor("x", DataType::Float(32), true),
                                        {})(call->args[0]);
                                    _RV axis = Axes(_TensorNDim(x), true, false)(call->args[1]);
                                    return _h.AddOutput(CallGlobalFunc(_te, {x, axis}));
                                  } catch (const NotDerivable& e) {
                                    return ExpandDimsSInfoFallback(call, _bb);
                                  }
                                })
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.expand_dims");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV x = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("x", DataType::Float(32), true),
                                                         {})(call->args[0]);
        _RV axis = Axes(_TensorNDim(x), true, false)(call->args[1]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, axis}));
        return _h.EmitTE(_bb, "expand_dims", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call flatten(relax::Expr x, int64_t start_dim, int64_t end_dim) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.flatten");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(Axis()(start_dim), start_dim, _args);
  TVM_RELAX_OP_ARG_CHECK(Axis()(end_dim), end_dim, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.flatten").set_body_typed(flatten);
TVM_RELAX_REGISTER_OP("flatten")
    .set_attr<FInferStructInfo>(
        "FInferStructInfo",
        [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
          using namespace relax::relax2te;
          using _RV = runtime::TVMRetValue;
          static const auto* _te = runtime::Registry::Get("topi.array_api.flatten");
          ICHECK(_te != nullptr);
          TensorHandler _h;
          try {
            _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x", DataType::Float(32), true), {})(call->args[0]);
            _RV start_dim = Axis(_TensorNDim(x), false, true)(call->args[1]);
            _RV end_dim = Axis(_TensorNDim(x), false, true)(call->args[2]);
            return _h.AddOutput(CallGlobalFunc(_te, {x, start_dim, end_dim}));
          } catch (const NotDerivable& e) {
            return FlattenSInfoFallback(call, _bb);
          }
        })
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.flatten");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV x = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("x", DataType::Float(32), true),
                                                         {})(call->args[0]);
        _RV start_dim = Axis(_TensorNDim(x), false, true)(call->args[1]);
        _RV end_dim = Axis(_TensorNDim(x), false, true)(call->args[2]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, start_dim, end_dim}));
        return _h.EmitTE(_bb, "flatten", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call layout_transform(relax::Expr x, tir::IndexMap index_map, Optional<FloatImm> pad_value) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.layout_transform");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(IndexMapToOpaque()(index_map), index_map, _args);
  TVM_RELAX_OP_ARG_CHECK(OptionalToOpaque(ScalarToPrimValue(false))(pad_value), pad_value, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.layout_transform").set_body_typed(layout_transform);
TVM_RELAX_REGISTER_OP("layout_transform")
    .set_attr<FInferStructInfo>("FInferStructInfo", LayoutTransformSInfo);
relax::Call permute_dims(relax::Expr x, Optional<Array<IntImm>> axes) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.permute_dims");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(OptionalToOpaque(Axes())(axes), axes, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.permute_dims").set_body_typed(permute_dims);
TVM_RELAX_REGISTER_OP("permute_dims")
    .set_attr<FInferStructInfo>(
        "FInferStructInfo",
        [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
          using namespace relax::relax2te;
          using _RV = runtime::TVMRetValue;
          static const auto* _te = runtime::Registry::Get("topi.array_api.permute_dims");
          ICHECK(_te != nullptr);
          TensorHandler _h;
          try {
            _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x", DataType::Float(32), true), {})(call->args[0]);
            _RV axes = OptionalFromOpaque(Axes(_TensorNDim(x), false, false))(call->args[1]);
            return _h.AddOutput(CallGlobalFunc(_te, {x, axes}));
          } catch (const NotDerivable& e) {
            return PermuteDimsSInfoFallback(call, _bb);
          }
        })
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.permute_dims");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV x = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("x", DataType::Float(32), true),
                                                         {})(call->args[0]);
        _RV axes = OptionalFromOpaque(Axes(_TensorNDim(x), false, false))(call->args[1]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, axes}));
        return _h.EmitTE(_bb, "permute_dims", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call repeat(relax::Expr x, Array<PrimExpr> repeats, Optional<IntImm> axis) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.repeat");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(AttrExpr(repeats), repeats, _args);
  TVM_RELAX_OP_ARG_CHECK(OptionalToOpaque(Axis())(axis), axis, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.repeat").set_body_typed(repeat);
TVM_RELAX_REGISTER_OP("repeat")
    .set_attr<FInferStructInfo>(
        "FInferStructInfo",
        [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
          using namespace relax::relax2te;
          using _RV = runtime::TVMRetValue;
          static const auto* _te = runtime::Registry::Get("topi.array_api.repeat");
          ICHECK(_te != nullptr);
          TensorHandler _h;
          try {
            _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x", DataType::Float(32), true), {})(call->args[0]);
            _RV repeats = ArrayFromOpaque()(call->args[1]);
            _RV axis = OptionalFromOpaque(Axis(_TensorNDim(x), false, true))(call->args[2]);
            return _h.AddOutput(CallGlobalFunc(_te, {x, repeats, axis}));
          } catch (const NotDerivable& e) {
            return RepeatSInfoFallback(call, _bb);
          }
        })
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.repeat");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV x = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("x", DataType::Float(32), true),
                                                         {})(call->args[0]);
        _RV repeats = ArrayFromOpaque()(call->args[1]);
        _RV axis = OptionalFromOpaque(Axis(_TensorNDim(x), false, true))(call->args[2]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, repeats, axis}));
        return _h.EmitTE(_bb, "repeat", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call reshape(relax::Expr x, relax::Expr shape) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.reshape");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(shape, shape, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.reshape").set_body_typed(reshape);
TVM_RELAX_REGISTER_OP("reshape")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
                                  using namespace relax::relax2te;
                                  using _RV = runtime::TVMRetValue;
                                  static const auto* _te =
                                      runtime::Registry::Get("topi.array_api.reshape");
                                  ICHECK(_te != nullptr);
                                  TensorHandler _h;
                                  try {
                                    _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                                        _h.Tensor("x", DataType::Float(32), true),
                                        {})(call->args[0]);
                                    _RV shape = ShapeArrayFromShape()(call->args[1]);
                                    return _h.AddOutput(CallGlobalFunc(_te, {x, shape}));
                                  } catch (const NotDerivable& e) {
                                    return ReshapeSInfoFallback(call, _bb);
                                  }
                                })
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.reshape");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV x = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("x", DataType::Float(32), true),
                                                         {})(call->args[0]);
        _RV shape = ShapeArrayFromShape()(call->args[1]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, shape}));
        return _h.EmitTE(_bb, "reshape", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call split(relax::Expr x, ObjectRef indices_or_sections, int64_t axis) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.split");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(AttrExpr(indices_or_sections), indices_or_sections, _args);
  TVM_RELAX_OP_ARG_CHECK(Axis()(axis), axis, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.split").set_body_typed(split);
TVM_RELAX_REGISTER_OP("split")
    .set_attr<FInferStructInfo>(
        "FInferStructInfo",
        [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
          using namespace relax::relax2te;
          using _RV = runtime::TVMRetValue;
          static const auto* _te = runtime::Registry::Get("topi.array_api.split");
          ICHECK(_te != nullptr);
          TensorHandler _h;
          try {
            _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x", DataType::Float(32), true), {})(call->args[0]);
            _RV indices_or_sections = ObjectFromOpaque()(call->args[1]);
            _RV axis = Axis(_TensorNDim(x), false, true)(call->args[2]);
            return _h.AddOutput(CallGlobalFunc(_te, {x, indices_or_sections, axis}));
          } catch (const NotDerivable& e) {
            return SplitSInfoFallback(call, _bb);
          }
        })
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.split");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV x = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("x", DataType::Float(32), true),
                                                         {})(call->args[0]);
        _RV indices_or_sections = ObjectFromOpaque()(call->args[1]);
        _RV axis = Axis(_TensorNDim(x), false, true)(call->args[2]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, indices_or_sections, axis}));
        return _h.EmitTE(_bb, "split", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call squeeze(relax::Expr x, Optional<Array<IntImm>> axis) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.squeeze");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(OptionalToOpaque(Axes())(axis), axis, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.squeeze").set_body_typed(squeeze);
TVM_RELAX_REGISTER_OP("squeeze")
    .set_attr<FInferStructInfo>(
        "FInferStructInfo",
        [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
          using namespace relax::relax2te;
          using _RV = runtime::TVMRetValue;
          static const auto* _te = runtime::Registry::Get("topi.array_api.squeeze");
          ICHECK(_te != nullptr);
          TensorHandler _h;
          try {
            _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x", DataType::Float(32), false), {})(call->args[0]);
            _RV axis = OptionalFromOpaque(Axes(_TensorNDim(x), false, true))(call->args[1]);
            return _h.AddOutput(CallGlobalFunc(_te, {x, axis}));
          } catch (const NotDerivable& e) {
            return SqueezeSInfoFallback(call, _bb);
          }
        })
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.squeeze");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV x = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("x", DataType::Float(32), false),
                                                         {})(call->args[0]);
        _RV axis = OptionalFromOpaque(Axes(_TensorNDim(x), false, true))(call->args[1]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, axis}));
        return _h.EmitTE(_bb, "squeeze", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call tile(relax::Expr x, relax::Expr repeats) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.tile");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(repeats, repeats, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.tile").set_body_typed(tile);
TVM_RELAX_REGISTER_OP("tile")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
                                  using namespace relax::relax2te;
                                  using _RV = runtime::TVMRetValue;
                                  static const auto* _te =
                                      runtime::Registry::Get("topi.array_api.tile");
                                  ICHECK(_te != nullptr);
                                  TensorHandler _h;
                                  try {
                                    _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                                        _h.Tensor("x", DataType::Float(32), true),
                                        {})(call->args[0]);
                                    _RV repeats = ShapeArrayFromShape()(call->args[1]);
                                    return _h.AddOutput(CallGlobalFunc(_te, {x, repeats}));
                                  } catch (const NotDerivable& e) {
                                    return TileSInfoFallback(call, _bb);
                                  }
                                })
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.tile");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV x = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("x", DataType::Float(32), true),
                                                         {})(call->args[0]);
        _RV repeats = ShapeArrayFromShape()(call->args[1]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, repeats}));
        return _h.EmitTE(_bb, "tile", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
// (TVM-TOOL) cc_op end def/manipulate/*

}  // namespace relax
}  // namespace tvm
