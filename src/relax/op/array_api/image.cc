#include "../arg2relax.h"
#include "../op_common.h"
#include "../relax2te.h"

namespace tvm {
namespace relax {
namespace image {

InferLayoutOutput InferLayoutResize2d(const Call& call,
                                      const Map<String, Array<String>>& desired_layouts,
                                      const VarLayoutMap& var_layout_map) {
  // TODO
  LOG(FATAL) << "not implemented";
  // ICHECK(NoDesiredLayout(call, desired_layouts));
  // const auto* attrs = call->attrs.as<Resize2DAttrs>();
  // ICHECK(attrs) << "Invalid Call";
  // LayoutDecision layout = GetLayoutDecision(var_layout_map, call->args[0]);
  // ObjectPtr<Resize2DAttrs> new_attrs = make_object<Resize2DAttrs>(*attrs);
  // new_attrs->layout = TransposeLike(attrs->layout, InitialLayout(4), layout->layout).name();
  // return InferLayoutOutput({layout, InitialNLayout(call->args[1])}, {layout}, Attrs(new_attrs));
}

StructInfo Resize2dSInfoFallback(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo x = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
  ShapeStructInfo size = MatchStructInfo<ShapeStructInfo>(call->args[1]).value();
  DataType out_dtype = relax2te::DType()(call->args[10]);
  if (out_dtype.is_void()) {
    out_dtype = x->dtype;
  }
  String layout = relax2te::Str()(call->args[3]);
  auto [data_layout, data2NCHW] = CheckTensorLayout(call, ctx, layout,
                                                    /*tgt_layout=*/"NCHW",
                                                    /*tensor_name=*/"data");
  if (!x->IsUnknownNdim() && x->ndim != static_cast<int>(data_layout.ndim())) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "In " << call->op << ", layout " << layout << " requires the input to be "
                     << data_layout.ndim() << "-dim tensor. However, the given input has ndim "
                     << x->ndim);
  }
  if (size->ndim != 1 && size->ndim != 2) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "In " << call->op
                     << ", the size should be an integer or 2 integers. However, "
                     << "the given size has ndim " << size->ndim);
  }
  return TensorStructInfo(out_dtype, data_layout.ndim());
}

// (TVM-TOOL) cc_op begin def/image/*
relax::Call resize2d(relax::Expr x, relax::Expr size, Array<FloatImm> roi, String layout,
                     String method, String coordinate_transformation_mode, String rounding_method,
                     double bicubic_alpha, double bicubic_exclude, double extrapolation_value,
                     runtime::DataType out_dtype) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.image.resize2d");
  Array<relax::Expr> _args;
  _args.reserve(11);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(size, size, _args);
  TVM_RELAX_OP_ARG_CHECK(ArrayToOpaque(ScalarToPrimValue(false), {4})(roi), roi, _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(layout), layout, _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(method), method, _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(coordinate_transformation_mode), coordinate_transformation_mode,
                         _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(rounding_method), rounding_method, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(false)(bicubic_alpha), bicubic_alpha, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(false)(bicubic_exclude), bicubic_exclude, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(false)(extrapolation_value), extrapolation_value, _args);
  TVM_RELAX_OP_ARG_CHECK(DType()(out_dtype), out_dtype, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.image.resize2d").set_body_typed(resize2d);
TVM_RELAX_REGISTER_OP("image.resize2d")
    .set_attr<FInferStructInfo>(
        "FInferStructInfo",
        [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
          using namespace relax::relax2te;
          using _RV = runtime::TVMRetValue;
          static const auto* _te = runtime::Registry::Get("topi.array_api.image.resize2d");
          ICHECK(_te != nullptr);
          TensorHandler _h;
          try {
            _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x", DataType::Float(32), true), {})(call->args[0]);
            _RV size = ShapeArrayFromShape()(call->args[1]);
            _RV roi = ArrayFromOpaque()(call->args[2]);
            _RV layout = Str()(call->args[3]);
            _RV method = Str()(call->args[4]);
            _RV coordinate_transformation_mode = Str()(call->args[5]);
            _RV rounding_method = Str()(call->args[6]);
            _RV bicubic_alpha = ScalarFromPrimValue<DTypeFloat>()(call->args[7]);
            _RV bicubic_exclude = ScalarFromPrimValue<DTypeFloat>()(call->args[8]);
            _RV extrapolation_value = ScalarFromPrimValue<DTypeFloat>()(call->args[9]);
            _RV out_dtype = DType()(call->args[10]);
            return _h.AddOutput(
                CallGlobalFunc(_te, {x, size, roi, layout, method, coordinate_transformation_mode,
                                     rounding_method, bicubic_alpha, bicubic_exclude,
                                     extrapolation_value, out_dtype}),
                out_dtype);
          } catch (const NotDerivable& e) {
            return Resize2dSInfoFallback(call, _bb);
          }
        })
    .set_attr<FLegalize>(
        "FLegalize",
        [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
          using namespace relax::relax2te;
          using _RV = runtime::TVMRetValue;
          static const auto* _te = runtime::Registry::Get("topi.array_api.image.resize2d");
          ICHECK(_te != nullptr);
          TensorHandler _h;
          try {
            _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x", DataType::Float(32), true), {})(call->args[0]);
            _RV size = ShapeArrayFromShape()(call->args[1]);
            _RV roi = ArrayFromOpaque()(call->args[2]);
            _RV layout = Str()(call->args[3]);
            _RV method = Str()(call->args[4]);
            _RV coordinate_transformation_mode = Str()(call->args[5]);
            _RV rounding_method = Str()(call->args[6]);
            _RV bicubic_alpha = ScalarFromPrimValue<DTypeFloat>()(call->args[7]);
            _RV bicubic_exclude = ScalarFromPrimValue<DTypeFloat>()(call->args[8]);
            _RV extrapolation_value = ScalarFromPrimValue<DTypeFloat>()(call->args[9]);
            _RV out_dtype = DType()(call->args[10]);
            StructInfo _sinfo = _h.AddOutput(
                CallGlobalFunc(_te, {x, size, roi, layout, method, coordinate_transformation_mode,
                                     rounding_method, bicubic_alpha, bicubic_exclude,
                                     extrapolation_value, out_dtype}),
                out_dtype);
            return _h.EmitTE(_bb, "resize2d", _sinfo);
          } catch (const NotDerivable& e) {
            return call;
          }
        })
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutResize2d)
    .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow);
// (TVM-TOOL) cc_op end def/image/*

}  // namespace image
}  // namespace relax
}  // namespace tvm
