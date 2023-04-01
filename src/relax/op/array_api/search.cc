#include "../arg2relax.h"
#include "../infer_struct_info.h"
#include "../op_common.h"
#include "../relax2te.h"

namespace tvm {
namespace relax {

StructInfo WhereSInfo(const Call& call, const BlockBuilder& bb) {
  using namespace relax::relax2te;
  FInferStructInfo f_info = TernarySInfo("topi.array_api.where", DataType::Void());
  try {
    TensorHandler _h;
    TETensorFromRelaxTensor<DTypeBool, false>(_h.Tensor("cond", DataType::Bool(), true),
                                              {})(call->args[0]);
    TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("a", DataType::Float(32), true),
                                             {})(call->args[1]);
    TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("b", DataType::Float(32), true),
                                             {})(call->args[2]);

  } catch (const NotDerivable& e) {
  }
  return f_info(call, bb);
}

// (TVM-TOOL) cc_op begin def/search/*
relax::Call argmax(relax::Expr a, Array<IntImm> axis, bool keepdims) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.argmax");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(Axes()(axis), axis, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(keepdims), keepdims, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.argmax").set_body_typed(argmax);
TVM_RELAX_REGISTER_OP("argmax")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
                                  using namespace relax::relax2te;
                                  using _RV = runtime::TVMRetValue;
                                  static const auto* _te =
                                      runtime::Registry::Get("topi.array_api.argmax");
                                  ICHECK(_te != nullptr);
                                  TensorHandler _h;
                                  try {
                                    _RV a = TETensorFromRelaxTensor<DTypeAll, false>(
                                        _h.Tensor("a", DataType::Float(32), true),
                                        {})(call->args[0]);
                                    _RV axis = Axes(_TensorNDim(a), false, true)(call->args[1]);
                                    _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
                                    return _h.AddOutput(CallGlobalFunc(_te, {a, axis, keepdims}));
                                  } catch (const NotDerivable& e) {
                                    return ReduceSInfoFallback(DataType::Int(64))(call, _bb);
                                  }
                                })
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.argmax");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV a = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("a", DataType::Float(32), true),
                                                         {})(call->args[0]);
        _RV axis = Axes(_TensorNDim(a), false, true)(call->args[1]);
        _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {a, axis, keepdims}));
        return _h.EmitTE(_bb, "argmax", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call argmin(relax::Expr a, Array<IntImm> axis, bool keepdims) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.argmin");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(a, a, _args);
  TVM_RELAX_OP_ARG_CHECK(Axes()(axis), axis, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(keepdims), keepdims, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.argmin").set_body_typed(argmin);
TVM_RELAX_REGISTER_OP("argmin")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
                                  using namespace relax::relax2te;
                                  using _RV = runtime::TVMRetValue;
                                  static const auto* _te =
                                      runtime::Registry::Get("topi.array_api.argmin");
                                  ICHECK(_te != nullptr);
                                  TensorHandler _h;
                                  try {
                                    _RV a = TETensorFromRelaxTensor<DTypeAll, false>(
                                        _h.Tensor("a", DataType::Float(32), true),
                                        {})(call->args[0]);
                                    _RV axis = Axes(_TensorNDim(a), false, true)(call->args[1]);
                                    _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
                                    return _h.AddOutput(CallGlobalFunc(_te, {a, axis, keepdims}));
                                  } catch (const NotDerivable& e) {
                                    return ReduceSInfoFallback(DataType::Int(64))(call, _bb);
                                  }
                                })
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.argmin");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV a = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("a", DataType::Float(32), true),
                                                         {})(call->args[0]);
        _RV axis = Axes(_TensorNDim(a), false, true)(call->args[1]);
        _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {a, axis, keepdims}));
        return _h.EmitTE(_bb, "argmin", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call where(relax::Expr condition, relax::Expr x, relax::Expr y) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.where");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(condition, condition, _args);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(y, y, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.where").set_body_typed(where);
TVM_RELAX_REGISTER_OP("where")
    .set_attr<FInferStructInfo>("FInferStructInfo", WhereSInfo)
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.where");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV condition = TETensorFromRelaxTensor<DTypeBool, false>(
            _h.Tensor("condition", DataType::Float(32), true), {})(call->args[0]);
        _RV x = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("x", DataType::Float(32), true),
                                                         {})(call->args[1]);
        _RV y = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("y", DataType::Float(32), true),
                                                         {})(call->args[2]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {condition, x, y}));
        return _h.EmitTE(_bb, "where", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
// (TVM-TOOL) cc_op end def/search/*

}  // namespace relax
}  // namespace tvm
