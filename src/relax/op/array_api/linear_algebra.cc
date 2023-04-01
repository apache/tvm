#include <tvm/relax/op/linear_algebra.h>

#include "../arg2relax.h"
#include "../op_common.h"
#include "../relax2te.h"

namespace tvm {
namespace relax {

Call InferMixedPrecisionMatmul(const Call& call, const DataType& out_dtype) {
  return Downcast<Call>(matmul(call->args[0], call->args[1], out_dtype));
}

StructInfo MatmulSInfoFallback(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo a = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
  TensorStructInfo b = MatchStructInfo<TensorStructInfo>(call->args[1]).value();
  DataType out_dtype = Downcast<DataTypeImm>(call->args[2])->value;
  if (out_dtype.is_void()) {
    DataType a_dtype = a->dtype;
    DataType b_dtype = b->dtype;
    if (!a_dtype.is_void() && !b_dtype.is_void()) {
      te::Tensor a_tensor = te::placeholder({}, a_dtype, "a");
      te::Tensor b_tensor = te::placeholder({}, b_dtype, "b");
      te::Tensor c_tensor = te::compute(
          {}, [&](const Array<tir::Var>&) { return a_tensor(0) * b_tensor(0); }, "c");
      out_dtype = c_tensor->dtype;
    }
  }
  if (a->IsUnknownNdim() || b->IsUnknownNdim()) {
    return TensorStructInfo(out_dtype, kUnknownNDim);
  }
  int out_ndim = std::max(2, std::max(a->ndim, b->ndim));
  out_ndim -= static_cast<int>(a->ndim == 1);
  out_ndim -= static_cast<int>(b->ndim == 1);
  return TensorStructInfo(out_dtype, out_ndim);
}

// (TVM-TOOL) cc_op begin def/linear_algebra/*
relax::Call matmul(relax::Expr x1, relax::Expr x2, runtime::DataType out_dtype) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.matmul");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(x1, x1, _args);
  TVM_RELAX_OP_ARG_CHECK(x2, x2, _args);
  TVM_RELAX_OP_ARG_CHECK(DType()(out_dtype), out_dtype, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.matmul").set_body_typed(matmul);
TVM_RELAX_REGISTER_OP("matmul")
    .set_attr<FInferStructInfo>(
        "FInferStructInfo",
        [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
          using namespace relax::relax2te;
          using _RV = runtime::TVMRetValue;
          static const auto* _te = runtime::Registry::Get("topi.array_api.matmul");
          ICHECK(_te != nullptr);
          TensorHandler _h;
          try {
            _RV x1 = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x1", DataType::Float(32), false), {})(call->args[0]);
            _RV x2 = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x2", DataType::Float(32), false), {})(call->args[1]);
            _RV out_dtype = DType()(call->args[2]);
            return _h.AddOutput(CallGlobalFunc(_te, {x1, x2, out_dtype}), out_dtype);
          } catch (const NotDerivable& e) {
            return MatmulSInfoFallback(call, _bb);
          }
        })
    .set_attr<FLegalize>(
        "FLegalize",
        [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
          using namespace relax::relax2te;
          using _RV = runtime::TVMRetValue;
          static const auto* _te = runtime::Registry::Get("topi.array_api.matmul");
          ICHECK(_te != nullptr);
          TensorHandler _h;
          try {
            _RV x1 = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x1", DataType::Float(32), false), {})(call->args[0]);
            _RV x2 = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x2", DataType::Float(32), false), {})(call->args[1]);
            _RV out_dtype = DType()(call->args[2]);
            StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x1, x2, out_dtype}), out_dtype);
            return _h.EmitTE(_bb, "matmul", _sinfo);
          } catch (const NotDerivable& e) {
            return call;
          }
        })
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kAlways)
    .set_attr<FInferMixedPrecision>("FInferMixedPrecision", InferMixedPrecisionMatmul);
// (TVM-TOOL) cc_op end def/linear_algebra/*

}  // namespace relax
}  // namespace tvm
