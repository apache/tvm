#ifndef TVM_RELAX_OP_INFER_STRUCT_INFO_H_
#define TVM_RELAX_OP_INFER_STRUCT_INFO_H_

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>

#include "./relax2te.h"

namespace tvm {
namespace relax {

inline StructInfo InferStructInfoReturnFirstSInfoArg(const Call& call, const BlockBuilder& ctx) {
  if (call->sinfo_args.size() != 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "sinfo_args should have exact 1 output struct info.");
  }
  return call->sinfo_args[0];
}

inline StructInfo InferStructInfoReturnsObject(const Call& call, const BlockBuilder& ctx) {
  return ObjectStructInfo();
}

inline StructInfo InferStructInfoReturnsVoid(const Call& call, const BlockBuilder& ctx) {
  return TupleStructInfo(Array<StructInfo>{});
}

FInferStructInfo UnarySInfo(std::string te_name, runtime::DataType force_dtype);
FInferStructInfo BinarySInfo(std::string te_name, runtime::DataType force_dtype);
FInferStructInfo TernarySInfo(std::string te_name, runtime::DataType force_dtype);

inline FInferStructInfo ReduceSInfoFallback(DataType out_dtype) {
  return [out_dtype](const Call& call, const BlockBuilder& ctx) -> StructInfo {
    using relax2te::Axes;
    using relax2te::DTypeBool;
    using relax2te::ScalarFromPrimValue;
    TensorStructInfo x = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
    DataType dtype = out_dtype.is_void() ? x->dtype : out_dtype;
    bool keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
    if (x->IsUnknownNdim()) {
      ObjectRef axes = Downcast<AttrExpr>(call->args[1])->value;
      return keepdims || axes.defined() ? TensorStructInfo(dtype, x->ndim)
                                        : TensorStructInfo(ShapeExpr(Array<PrimExpr>{}), dtype);
    }
    int ndim = x->ndim;
    Array<IntImm> axes = Axes(x->ndim, false, true)(call->args[1]);
    int num_axes = axes.defined() ? axes.size() : x->ndim;
    if (keepdims && num_axes == ndim) {
      Array<PrimExpr> all_ones;
      for (int i = 0; i < ndim; ++i) {
        all_ones.push_back(IntImm(DataType::Int(64), 1));
      }
      return TensorStructInfo(ShapeExpr(all_ones), dtype);
    }
    if (!keepdims) {
      ndim -= num_axes;
    }
    // TODO: migrate this change to the constructor of TensorStructInfo
    return ndim == 0 ? TensorStructInfo(ShapeExpr(Array<PrimExpr>{}), dtype)
                     : TensorStructInfo(dtype, ndim);
  };
}

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_OP_INFER_STRUCT_INFO_H_
