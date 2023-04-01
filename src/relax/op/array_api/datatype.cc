#include "../arg2relax.h"
#include "../op_common.h"
#include "../relax2te.h"

namespace tvm {
namespace relax {

StructInfo AsTypeSInfoFallback(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo x = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
  DataType dtype = relax2te::DType()(call->args[1]);
  if (x->shape.defined()) {
    return TensorStructInfo(x->shape.value(), dtype);
  } else {
    return TensorStructInfo(dtype, x->ndim);
  }
}

relax::Expr AsTypeLegalize(const BlockBuilder& bb, const Call& call) {
  using namespace relax::relax2te;
  using RV = runtime::TVMRetValue;
  if (const auto* x = call->args[0].as<relax::ConstantNode>()) {
    if (const auto* dtype = call->args[1].as<relax::DataTypeImmNode>()) {
      if (x->data->ndim == 0) {
        PrimExpr e = tvm::cast(dtype->value, ScalarFromUnitNDArray(x->data).value());
        if (!e->IsInstance<tir::CastNode>()) {
          if (Optional<runtime::NDArray> c = ScalarToUnitNDArray(e)) {
            return relax::Constant(c.value());
          }
        }
      }
    }
  }
  static const auto* te = runtime::Registry::Get("topi.array_api.astype");
  ICHECK(te != nullptr);
  TensorHandler h;
  try {
    RV x = TETensorFromRelaxTensor<DTypeAll, false>(h.Tensor("x", DataType::Float(32), true),
                                                    {})(call->args[0]);
    RV dtype = DType()(call->args[1]);
    StructInfo sinfo = h.AddOutput(CallGlobalFunc(te, {x, dtype}), dtype);
    return h.EmitTE(bb, "astype", sinfo);
  } catch (const NotDerivable& e) {
    return call;
  }
}

// (TVM-TOOL) cc_op begin def/datatype/*
relax::Call astype(relax::Expr x, runtime::DataType dtype) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.astype");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(DType()(dtype), dtype, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.astype").set_body_typed(astype);
TVM_RELAX_REGISTER_OP("astype")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
                                  using namespace relax::relax2te;
                                  using _RV = runtime::TVMRetValue;
                                  static const auto* _te =
                                      runtime::Registry::Get("topi.array_api.astype");
                                  ICHECK(_te != nullptr);
                                  TensorHandler _h;
                                  try {
                                    _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                                        _h.Tensor("x", DataType::Float(32), true),
                                        {})(call->args[0]);
                                    _RV dtype = DType()(call->args[1]);
                                    return _h.AddOutput(CallGlobalFunc(_te, {x, dtype}), dtype);
                                  } catch (const NotDerivable& e) {
                                    return AsTypeSInfoFallback(call, _bb);
                                  }
                                })
    .set_attr<FLegalize>("FLegalize", AsTypeLegalize)
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
// (TVM-TOOL) cc_op end def/datatype/*

}  // namespace relax
}  // namespace tvm
