#include "../arg2relax.h"
#include "../infer_struct_info.h"
#include "../op_common.h"
#include "../relax2te.h"

namespace tvm {
namespace relax {

FInferStructInfo UnarySInfo(std::string te_name, runtime::DataType force_dtype) {
  return [te_name, force_dtype](const Call& call, const BlockBuilder& bb) -> StructInfo {
    using namespace relax::relax2te;
    const auto* te_func = runtime::Registry::Get(te_name);
    try {
      TensorHandler h;
      te::Tensor a = TETensorFromRelaxTensor<DTypeAll, false>(
          h.Tensor("a", DataType::Void(), false), {})(call->args[0]);
      te::Tensor b = (*te_func)(a);
      return h.AddOutput(b);
    } catch (const NotDerivable& e) {
    }
    TensorStructInfo a = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
    runtime::DataType dtype = a->dtype;
    if (!force_dtype.is_void()) {
      dtype = force_dtype;
    }
    if (a->shape.defined()) {
      return TensorStructInfo(a->shape.value(), dtype);
    } else {
      return TensorStructInfo(dtype, a->ndim);
    }
  };
}

FInferStructInfo BinarySInfo(std::string te_name, runtime::DataType force_dtype) {
  return [te_name, force_dtype](const Call& call, const BlockBuilder& bb) -> StructInfo {
    using namespace relax::relax2te;
    const auto* te_func = runtime::Registry::Get(te_name);
    TensorStructInfo a = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
    TensorStructInfo b = MatchStructInfo<TensorStructInfo>(call->args[1]).value();
    // Step 1.Infer dtype
    runtime::DataType dtype = force_dtype;
    if (force_dtype.is_void() && !a->IsUnknownDtype() && !b->IsUnknownDtype()) {
      te::Tensor a_te = te::placeholder({}, a->dtype, "a");
      te::Tensor b_te = te::placeholder({}, b->dtype, "b");
      te::Tensor c_te = (*te_func)(a_te, b_te);
      dtype = c_te->dtype;
    }
    // Step 2. Quick return if a and b's shape are the same
    if (a->shape.defined() && b->shape.defined() && a->shape.same_as(b->shape)) {
      return TensorStructInfo(a->shape.value(), dtype);
    }
    // Step 3. Infer ndim
    if (a->IsUnknownNdim() || b->IsUnknownNdim()) {
      return TensorStructInfo(dtype, kUnknownNDim);
    }
    int ndim = std::max(a->ndim, b->ndim);
    // Step 4. Infer shape
    const auto* a_shape = a->shape.as<ShapeExprNode>();
    const auto* b_shape = b->shape.as<ShapeExprNode>();
    if (!a_shape || !b_shape) {
      return TensorStructInfo(dtype, ndim);
    }
    try {
      Array<PrimExpr> shape = topi::array_api::BroadcastShape(a_shape->values,  //
                                                              b_shape->values,  //
                                                              bb->GetAnalyzer());
      ICHECK_EQ(shape.size(), ndim);
      return TensorStructInfo(ShapeExpr(shape), dtype);
    } catch (const relax2te::NotDerivable& e) {
      return TensorStructInfo(dtype, ndim);
    }
  };
}

FInferStructInfo TernarySInfo(std::string te_name, runtime::DataType force_dtype) {
  return [te_name, force_dtype](const Call& call, const BlockBuilder& bb) -> StructInfo {
    using namespace relax::relax2te;
    const auto* te_func = runtime::Registry::Get(te_name);
    TensorStructInfo a = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
    TensorStructInfo b = MatchStructInfo<TensorStructInfo>(call->args[1]).value();
    TensorStructInfo c = MatchStructInfo<TensorStructInfo>(call->args[2]).value();
    // Step 1.Infer dtype
    runtime::DataType dtype = force_dtype;
    if (force_dtype.is_void() && !a->IsUnknownDtype() && !b->IsUnknownDtype() &&
        !c->IsUnknownDtype()) {
      te::Tensor a_te = te::placeholder({}, a->dtype, "a");
      te::Tensor b_te = te::placeholder({}, b->dtype, "b");
      te::Tensor c_te = te::placeholder({}, c->dtype, "c");
      te::Tensor d_te = (*te_func)(a_te, b_te, c_te);
      dtype = d_te->dtype;
    }
    // Step 2. Quick return if a/b/c's shape are the same
    if (a->shape.defined() && b->shape.defined() && c->shape.defined() &&
        a->shape.same_as(b->shape) && a->shape.same_as(c->shape)) {
      return TensorStructInfo(a->shape.value(), dtype);
    }
    // Step 3. Infer ndim
    if (a->IsUnknownNdim() || b->IsUnknownNdim() || c->IsUnknownNdim()) {
      return TensorStructInfo(dtype, kUnknownNDim);
    }
    int ndim = std::max(a->ndim, std::max(b->ndim, c->ndim));
    // Step 4. Infer shape
    const auto* a_shape = a->shape.as<ShapeExprNode>();
    const auto* b_shape = b->shape.as<ShapeExprNode>();
    const auto* c_shape = c->shape.as<ShapeExprNode>();
    if (!a_shape || !b_shape || !c_shape) {
      return TensorStructInfo(dtype, ndim);
    }
    try {
      Array<PrimExpr> shape = topi::array_api::BroadcastShape(a_shape->values,  //
                                                              b_shape->values,  //
                                                              c_shape->values,  //
                                                              bb->GetAnalyzer());
      ICHECK_EQ(shape.size(), ndim);
      return TensorStructInfo(ShapeExpr(shape), dtype);
    } catch (const relax2te::NotDerivable& e) {
      return TensorStructInfo(dtype, ndim);
    }
  };
}

FLegalize UnaryLegalize(std::string te_name, std::string name_hint) {
  return [te_name, name_hint](const BlockBuilder& bb, const Call& call) -> Call {
    using namespace relax::relax2te;
    const auto* te_func = runtime::Registry::Get(te_name);
    TensorHandler h;
    try {
      ObjectRef a = TETensorFromRelaxTensor<DTypeAll, true>(h.Tensor("a", DataType::Void(), false),
                                                            {})(call->args[0]);
      StructInfo sinfo = h.AddOutput((*te_func)(a));
      return h.EmitTE(bb, name_hint, sinfo);
    } catch (const NotDerivable& e) {
    }
    return call;
  };
}

FLegalize BinaryLegalize(std::string te_name, std::string name_hint) {
  return [te_name, name_hint](const BlockBuilder& bb, const Call& call) -> Call {
    using namespace relax::relax2te;
    const auto* te_func = runtime::Registry::Get(te_name);
    TensorHandler h;
    try {
      ObjectRef a = TETensorFromRelaxTensor<DTypeAll, true>(h.Tensor("a", DataType::Void(), false),
                                                            {})(call->args[0]);
      ObjectRef b = TETensorFromRelaxTensor<DTypeAll, true>(h.Tensor("b", DataType::Void(), false),
                                                            {})(call->args[1]);
      if (a->IsInstance<PrimExprNode>() && b->IsInstance<PrimExprNode>()) {
        h.Clear();
        a = TETensorFromRelaxTensor<DTypeAll, false>(h.Tensor("a", DataType::Void(), false),
                                                     {})(call->args[0]);
        b = TETensorFromRelaxTensor<DTypeAll, true>(h.Tensor("b", DataType::Void(), false),
                                                    {})(call->args[1]);
      }
      StructInfo sinfo = h.AddOutput((*te_func)(a, b));
      return h.EmitTE(bb, name_hint, sinfo);
    } catch (const NotDerivable& e) {
    }
    return call;
  };
}

// (TVM-TOOL) cc_op begin def/elementwise/*
relax::Call abs(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.abs");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.abs").set_body_typed(abs);
TVM_RELAX_REGISTER_OP("abs")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.abs", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.abs", "tir_abs"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call acos(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.acos");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.acos").set_body_typed(acos);
TVM_RELAX_REGISTER_OP("acos")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.acos", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.acos", "tir_acos"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call acosh(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.acosh");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.acosh").set_body_typed(acosh);
TVM_RELAX_REGISTER_OP("acosh")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.acosh", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.acosh", "tir_acosh"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call add(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.add");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.add").set_body_typed(add);
TVM_RELAX_REGISTER_OP("add")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                BinarySInfo("topi.array_api.add", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.add", "add"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call asin(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.asin");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.asin").set_body_typed(asin);
TVM_RELAX_REGISTER_OP("asin")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.asin", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.asin", "tir_asin"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call asinh(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.asinh");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.asinh").set_body_typed(asinh);
TVM_RELAX_REGISTER_OP("asinh")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.asinh", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.asinh", "tir_asinh"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call atan(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.atan");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.atan").set_body_typed(atan);
TVM_RELAX_REGISTER_OP("atan")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.atan", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.atan", "tir_atan"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call atan2(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.atan2");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.atan2").set_body_typed(atan2);
TVM_RELAX_REGISTER_OP("atan2")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                BinarySInfo("topi.array_api.atan2", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.atan2", "atan2"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call atanh(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.atanh");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.atanh").set_body_typed(atanh);
TVM_RELAX_REGISTER_OP("atanh")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.atanh", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.atanh", "tir_atanh"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call bitwise_and(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.bitwise_and");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.bitwise_and").set_body_typed(bitwise_and);
TVM_RELAX_REGISTER_OP("bitwise_and")
    .set_attr<FInferStructInfo>("FInferStructInfo", BinarySInfo("topi.array_api.bitwise_and",
                                                                runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.bitwise_and", "bitwise_and"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call bitwise_invert(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.bitwise_invert");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.bitwise_invert").set_body_typed(bitwise_invert);
TVM_RELAX_REGISTER_OP("bitwise_invert")
    .set_attr<FInferStructInfo>("FInferStructInfo", UnarySInfo("topi.array_api.bitwise_invert",
                                                               runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize",
                         UnaryLegalize("topi.array_api.bitwise_invert", "tir_bitwise_invert"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call bitwise_left_shift(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.bitwise_left_shift");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.bitwise_left_shift").set_body_typed(bitwise_left_shift);
TVM_RELAX_REGISTER_OP("bitwise_left_shift")
    .set_attr<FInferStructInfo>("FInferStructInfo", BinarySInfo("topi.array_api.bitwise_left_shift",
                                                                runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize",
                         BinaryLegalize("topi.array_api.bitwise_left_shift", "bitwise_left_shift"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call bitwise_or(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.bitwise_or");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.bitwise_or").set_body_typed(bitwise_or);
TVM_RELAX_REGISTER_OP("bitwise_or")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                BinarySInfo("topi.array_api.bitwise_or", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.bitwise_or", "bitwise_or"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call bitwise_right_shift(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.bitwise_right_shift");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.bitwise_right_shift").set_body_typed(bitwise_right_shift);
TVM_RELAX_REGISTER_OP("bitwise_right_shift")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                BinarySInfo("topi.array_api.bitwise_right_shift",
                                            runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.bitwise_right_shift",
                                                     "bitwise_right_shift"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call bitwise_xor(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.bitwise_xor");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.bitwise_xor").set_body_typed(bitwise_xor);
TVM_RELAX_REGISTER_OP("bitwise_xor")
    .set_attr<FInferStructInfo>("FInferStructInfo", BinarySInfo("topi.array_api.bitwise_xor",
                                                                runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.bitwise_xor", "bitwise_xor"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call ceil(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.ceil");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.ceil").set_body_typed(ceil);
TVM_RELAX_REGISTER_OP("ceil")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.ceil", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.ceil", "tir_ceil"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call cos(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.cos");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.cos").set_body_typed(cos);
TVM_RELAX_REGISTER_OP("cos")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.cos", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.cos", "tir_cos"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call cosh(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.cosh");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.cosh").set_body_typed(cosh);
TVM_RELAX_REGISTER_OP("cosh")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.cosh", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.cosh", "tir_cosh"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call divide(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.divide");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.divide").set_body_typed(divide);
TVM_RELAX_REGISTER_OP("divide")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                BinarySInfo("topi.array_api.divide", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.divide", "divide"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call equal(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.equal");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.equal").set_body_typed(equal);
TVM_RELAX_REGISTER_OP("equal")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                BinarySInfo("topi.array_api.equal", runtime::DataType::Bool()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.equal", "equal"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call exp(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.exp");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.exp").set_body_typed(exp);
TVM_RELAX_REGISTER_OP("exp")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.exp", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.exp", "tir_exp"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call floor(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.floor");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.floor").set_body_typed(floor);
TVM_RELAX_REGISTER_OP("floor")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.floor", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.floor", "tir_floor"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call floor_divide(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.floor_divide");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.floor_divide").set_body_typed(floor_divide);
TVM_RELAX_REGISTER_OP("floor_divide")
    .set_attr<FInferStructInfo>("FInferStructInfo", BinarySInfo("topi.array_api.floor_divide",
                                                                runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.floor_divide", "floor_divide"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call greater(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.greater");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.greater").set_body_typed(greater);
TVM_RELAX_REGISTER_OP("greater")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                BinarySInfo("topi.array_api.greater", runtime::DataType::Bool()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.greater", "greater"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call greater_equal(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.greater_equal");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.greater_equal").set_body_typed(greater_equal);
TVM_RELAX_REGISTER_OP("greater_equal")
    .set_attr<FInferStructInfo>("FInferStructInfo", BinarySInfo("topi.array_api.greater_equal",
                                                                runtime::DataType::Bool()))
    .set_attr<FLegalize>("FLegalize",
                         BinaryLegalize("topi.array_api.greater_equal", "greater_equal"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call isfinite(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.isfinite");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.isfinite").set_body_typed(isfinite);
TVM_RELAX_REGISTER_OP("isfinite")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.isfinite", runtime::DataType::Bool()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.isfinite", "tir_isfinite"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call isinf(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.isinf");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.isinf").set_body_typed(isinf);
TVM_RELAX_REGISTER_OP("isinf")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.isinf", runtime::DataType::Bool()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.isinf", "tir_isinf"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call isnan(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.isnan");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.isnan").set_body_typed(isnan);
TVM_RELAX_REGISTER_OP("isnan")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.isnan", runtime::DataType::Bool()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.isnan", "tir_isnan"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call less(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.less");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.less").set_body_typed(less);
TVM_RELAX_REGISTER_OP("less")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                BinarySInfo("topi.array_api.less", runtime::DataType::Bool()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.less", "less"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call less_equal(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.less_equal");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.less_equal").set_body_typed(less_equal);
TVM_RELAX_REGISTER_OP("less_equal")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                BinarySInfo("topi.array_api.less_equal", runtime::DataType::Bool()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.less_equal", "less_equal"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call log(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.log");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.log").set_body_typed(log);
TVM_RELAX_REGISTER_OP("log")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.log", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.log", "tir_log"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call log10(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.log10");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.log10").set_body_typed(log10);
TVM_RELAX_REGISTER_OP("log10")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.log10", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.log10", "tir_log10"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call log1p(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.log1p");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.log1p").set_body_typed(log1p);
TVM_RELAX_REGISTER_OP("log1p")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.log1p", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.log1p", "tir_log1p"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call log2(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.log2");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.log2").set_body_typed(log2);
TVM_RELAX_REGISTER_OP("log2")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.log2", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.log2", "tir_log2"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call logical_and(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.logical_and");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.logical_and").set_body_typed(logical_and);
TVM_RELAX_REGISTER_OP("logical_and")
    .set_attr<FInferStructInfo>("FInferStructInfo", BinarySInfo("topi.array_api.logical_and",
                                                                runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.logical_and", "logical_and"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call logical_not(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.logical_not");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.logical_not").set_body_typed(logical_not);
TVM_RELAX_REGISTER_OP("logical_not")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.logical_not", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize",
                         UnaryLegalize("topi.array_api.logical_not", "tir_logical_not"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call logical_or(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.logical_or");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.logical_or").set_body_typed(logical_or);
TVM_RELAX_REGISTER_OP("logical_or")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                BinarySInfo("topi.array_api.logical_or", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.logical_or", "logical_or"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call multiply(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.multiply");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.multiply").set_body_typed(multiply);
TVM_RELAX_REGISTER_OP("multiply")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                BinarySInfo("topi.array_api.multiply", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.multiply", "multiply"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call negative(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.negative");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.negative").set_body_typed(negative);
TVM_RELAX_REGISTER_OP("negative")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.negative", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.negative", "tir_negative"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call not_equal(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.not_equal");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.not_equal").set_body_typed(not_equal);
TVM_RELAX_REGISTER_OP("not_equal")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                BinarySInfo("topi.array_api.not_equal", runtime::DataType::Bool()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.not_equal", "not_equal"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call positive(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.positive");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.positive").set_body_typed(positive);
TVM_RELAX_REGISTER_OP("positive")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.positive", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.positive", "tir_positive"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call pow(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.pow");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.pow").set_body_typed(pow);
TVM_RELAX_REGISTER_OP("pow")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                BinarySInfo("topi.array_api.pow", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.pow", "pow"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call power(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.power");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.power").set_body_typed(power);
TVM_RELAX_REGISTER_OP("power")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                BinarySInfo("topi.array_api.power", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.power", "power"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call remainder(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.remainder");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.remainder").set_body_typed(remainder);
TVM_RELAX_REGISTER_OP("remainder")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                BinarySInfo("topi.array_api.remainder", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.remainder", "remainder"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call round(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.round");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.round").set_body_typed(round);
TVM_RELAX_REGISTER_OP("round")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.round", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.round", "tir_round"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call sin(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.sin");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.sin").set_body_typed(sin);
TVM_RELAX_REGISTER_OP("sin")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.sin", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.sin", "tir_sin"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call sinh(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.sinh");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.sinh").set_body_typed(sinh);
TVM_RELAX_REGISTER_OP("sinh")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.sinh", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.sinh", "tir_sinh"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call sqrt(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.sqrt");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.sqrt").set_body_typed(sqrt);
TVM_RELAX_REGISTER_OP("sqrt")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.sqrt", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.sqrt", "tir_sqrt"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call square(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.square");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.square").set_body_typed(square);
TVM_RELAX_REGISTER_OP("square")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.square", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.square", "tir_square"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call subtract(relax::Expr a, relax::Expr b) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.subtract");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(b, b, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.subtract").set_body_typed(subtract);
TVM_RELAX_REGISTER_OP("subtract")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                BinarySInfo("topi.array_api.subtract", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", BinaryLegalize("topi.array_api.subtract", "subtract"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutBinaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call tan(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.tan");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.tan").set_body_typed(tan);
TVM_RELAX_REGISTER_OP("tan")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.tan", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.tan", "tir_tan"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call tanh(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.tanh");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.tanh").set_body_typed(tanh);
TVM_RELAX_REGISTER_OP("tanh")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.tanh", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.tanh", "tir_tanh"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
relax::Call trunc(relax::Expr a) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.trunc");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.trunc").set_body_typed(trunc);
TVM_RELAX_REGISTER_OP("trunc")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                UnarySInfo("topi.array_api.trunc", runtime::DataType::Void()))
    .set_attr<FLegalize>("FLegalize", UnaryLegalize("topi.array_api.trunc", "tir_trunc"))
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
// (TVM-TOOL) cc_op end def/elementwise/*

}  // namespace relax
}  // namespace tvm
