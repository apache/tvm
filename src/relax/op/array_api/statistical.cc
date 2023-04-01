#include "../arg2relax.h"
#include "../infer_struct_info.h"
#include "../op_common.h"
#include "../relax2te.h"

namespace tvm {
namespace relax {

InferLayoutOutput InferLayoutStatistical(const Call& call,
                                         const Map<String, Array<String>>& desired_layouts,
                                         const VarLayoutMap& var_layout_map) {
  throw;
  // ICHECK(NoDesiredLayout(call, desired_layouts));
  //
  // const auto* attrs = call->attrs.as<StatisticalAttrs>();
  // ICHECK(attrs != nullptr) << "Invalid Call";
  // const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  // ICHECK(tensor_sinfo != nullptr) << "Invalid Call";
  // ICHECK(!tensor_sinfo->IsUnknownNdim()) << "Only support known ndim";
  // int ndim = tensor_sinfo->ndim;
  //
  // Array<Integer> axis;
  // if (attrs->axis.defined()) {
  //   axis = attrs->axis.value();
  // } else {
  //   axis.reserve(ndim);
  //   for (int i = 0; i < ndim; ++i) {
  //     axis.push_back(Integer(i));
  //   }
  // }
  //
  // std::string axis_str(ndim, '0');
  // for (const auto& iter : axis) {
  //   axis_str[(iter->value + ndim) % ndim] = '1';
  // }
  // for (int i = 0, j = 0; i < ndim; ++i) {
  //   if (axis_str[i] != '1') {
  //     axis_str[i] = 'A' + j++;
  //   }
  // }
  //
  // LayoutDecision exisiting_layout = GetLayoutDecision(var_layout_map, call->args[0]);
  // String new_axis_str = TransposeStrLike(axis_str, InitialLayout(ndim),
  // exisiting_layout->layout); Array<Integer> new_axis; for (size_t i = 0; i < new_axis_str.size();
  // ++i) {
  //   if (new_axis_str.at(i) == '1') {
  //     new_axis.push_back(Integer(i));
  //   }
  // }
  // std::string output_layout = new_axis_str;
  // output_layout.erase(std::remove(output_layout.begin(), output_layout.end(), '1'),
  //                     output_layout.end());
  //
  // ObjectPtr<StatisticalAttrs> new_attrs = make_object<StatisticalAttrs>(*attrs);
  // new_attrs->axis = new_axis;
  // return InferLayoutOutput({exisiting_layout},
  //                          {attrs->keepdims ? exisiting_layout : Layout(output_layout)},
  //                          Attrs(new_attrs));
}

// (TVM-TOOL) cc_op begin def/statistical/*
relax::Call max(relax::Expr x, Array<IntImm> axis, bool keepdims) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.max");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(Axes()(axis), axis, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(keepdims), keepdims, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.max").set_body_typed(max);
TVM_RELAX_REGISTER_OP("max")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
                                  using namespace relax::relax2te;
                                  using _RV = runtime::TVMRetValue;
                                  static const auto* _te =
                                      runtime::Registry::Get("topi.array_api.max");
                                  ICHECK(_te != nullptr);
                                  TensorHandler _h;
                                  try {
                                    _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                                        _h.Tensor("x", DataType::Float(32), true),
                                        {})(call->args[0]);
                                    _RV axis = Axes(_TensorNDim(x), false, true)(call->args[1]);
                                    _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
                                    return _h.AddOutput(CallGlobalFunc(_te, {x, axis, keepdims}));
                                  } catch (const NotDerivable& e) {
                                    return ReduceSInfoFallback(DataType::Void())(call, _bb);
                                  }
                                })
    .set_attr<FLegalize>(
        "FLegalize",
        [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
          using namespace relax::relax2te;
          using _RV = runtime::TVMRetValue;
          static const auto* _te = runtime::Registry::Get("topi.array_api.max");
          ICHECK(_te != nullptr);
          TensorHandler _h;
          try {
            _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x", DataType::Float(32), true), {})(call->args[0]);
            _RV axis = Axes(_TensorNDim(x), false, true)(call->args[1]);
            _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
            StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, axis, keepdims}));
            return _h.EmitTE(_bb, "max", _sinfo);
          } catch (const NotDerivable& e) {
            return call;
          }
        })
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutStatistical);
relax::Call mean(relax::Expr x, Array<IntImm> axis, bool keepdims) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.mean");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(Axes()(axis), axis, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(keepdims), keepdims, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.mean").set_body_typed(mean);
TVM_RELAX_REGISTER_OP("mean")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
                                  using namespace relax::relax2te;
                                  using _RV = runtime::TVMRetValue;
                                  static const auto* _te =
                                      runtime::Registry::Get("topi.array_api.mean");
                                  ICHECK(_te != nullptr);
                                  TensorHandler _h;
                                  try {
                                    _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                                        _h.Tensor("x", DataType::Float(32), true),
                                        {})(call->args[0]);
                                    _RV axis = Axes(_TensorNDim(x), false, true)(call->args[1]);
                                    _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
                                    return _h.AddOutput(CallGlobalFunc(_te, {x, axis, keepdims}));
                                  } catch (const NotDerivable& e) {
                                    return ReduceSInfoFallback(DataType::Void())(call, _bb);
                                  }
                                })
    .set_attr<FLegalize>(
        "FLegalize",
        [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
          using namespace relax::relax2te;
          using _RV = runtime::TVMRetValue;
          static const auto* _te = runtime::Registry::Get("topi.array_api.mean");
          ICHECK(_te != nullptr);
          TensorHandler _h;
          try {
            _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x", DataType::Float(32), true), {})(call->args[0]);
            _RV axis = Axes(_TensorNDim(x), false, true)(call->args[1]);
            _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
            StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, axis, keepdims}));
            return _h.EmitTE(_bb, "mean", _sinfo);
          } catch (const NotDerivable& e) {
            return call;
          }
        })
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutStatistical);
relax::Call min(relax::Expr x, Array<IntImm> axis, bool keepdims) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.min");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(Axes()(axis), axis, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(keepdims), keepdims, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.min").set_body_typed(min);
TVM_RELAX_REGISTER_OP("min")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
                                  using namespace relax::relax2te;
                                  using _RV = runtime::TVMRetValue;
                                  static const auto* _te =
                                      runtime::Registry::Get("topi.array_api.min");
                                  ICHECK(_te != nullptr);
                                  TensorHandler _h;
                                  try {
                                    _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                                        _h.Tensor("x", DataType::Float(32), true),
                                        {})(call->args[0]);
                                    _RV axis = Axes(_TensorNDim(x), false, true)(call->args[1]);
                                    _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
                                    return _h.AddOutput(CallGlobalFunc(_te, {x, axis, keepdims}));
                                  } catch (const NotDerivable& e) {
                                    return ReduceSInfoFallback(DataType::Void())(call, _bb);
                                  }
                                })
    .set_attr<FLegalize>(
        "FLegalize",
        [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
          using namespace relax::relax2te;
          using _RV = runtime::TVMRetValue;
          static const auto* _te = runtime::Registry::Get("topi.array_api.min");
          ICHECK(_te != nullptr);
          TensorHandler _h;
          try {
            _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x", DataType::Float(32), true), {})(call->args[0]);
            _RV axis = Axes(_TensorNDim(x), false, true)(call->args[1]);
            _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
            StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, axis, keepdims}));
            return _h.EmitTE(_bb, "min", _sinfo);
          } catch (const NotDerivable& e) {
            return call;
          }
        })
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutStatistical);
relax::Call prod(relax::Expr x, Array<IntImm> axis, bool keepdims) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.prod");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(Axes()(axis), axis, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(keepdims), keepdims, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.prod").set_body_typed(prod);
TVM_RELAX_REGISTER_OP("prod")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
                                  using namespace relax::relax2te;
                                  using _RV = runtime::TVMRetValue;
                                  static const auto* _te =
                                      runtime::Registry::Get("topi.array_api.prod");
                                  ICHECK(_te != nullptr);
                                  TensorHandler _h;
                                  try {
                                    _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                                        _h.Tensor("x", DataType::Float(32), true),
                                        {})(call->args[0]);
                                    _RV axis = Axes(_TensorNDim(x), false, true)(call->args[1]);
                                    _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
                                    return _h.AddOutput(CallGlobalFunc(_te, {x, axis, keepdims}));
                                  } catch (const NotDerivable& e) {
                                    return ReduceSInfoFallback(DataType::Void())(call, _bb);
                                  }
                                })
    .set_attr<FLegalize>(
        "FLegalize",
        [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
          using namespace relax::relax2te;
          using _RV = runtime::TVMRetValue;
          static const auto* _te = runtime::Registry::Get("topi.array_api.prod");
          ICHECK(_te != nullptr);
          TensorHandler _h;
          try {
            _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x", DataType::Float(32), true), {})(call->args[0]);
            _RV axis = Axes(_TensorNDim(x), false, true)(call->args[1]);
            _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
            StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, axis, keepdims}));
            return _h.EmitTE(_bb, "prod", _sinfo);
          } catch (const NotDerivable& e) {
            return call;
          }
        })
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutStatistical);
relax::Call std(relax::Expr x, Array<IntImm> axis, bool keepdims) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.std");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(Axes()(axis), axis, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(keepdims), keepdims, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.std").set_body_typed(std);
TVM_RELAX_REGISTER_OP("std")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
                                  using namespace relax::relax2te;
                                  using _RV = runtime::TVMRetValue;
                                  static const auto* _te =
                                      runtime::Registry::Get("topi.array_api.std");
                                  ICHECK(_te != nullptr);
                                  TensorHandler _h;
                                  try {
                                    _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                                        _h.Tensor("x", DataType::Float(32), true),
                                        {})(call->args[0]);
                                    _RV axis = Axes(_TensorNDim(x), false, true)(call->args[1]);
                                    _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
                                    return _h.AddOutput(CallGlobalFunc(_te, {x, axis, keepdims}));
                                  } catch (const NotDerivable& e) {
                                    return ReduceSInfoFallback(DataType::Void())(call, _bb);
                                  }
                                })
    .set_attr<FLegalize>(
        "FLegalize",
        [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
          using namespace relax::relax2te;
          using _RV = runtime::TVMRetValue;
          static const auto* _te = runtime::Registry::Get("topi.array_api.std");
          ICHECK(_te != nullptr);
          TensorHandler _h;
          try {
            _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x", DataType::Float(32), true), {})(call->args[0]);
            _RV axis = Axes(_TensorNDim(x), false, true)(call->args[1]);
            _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
            StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, axis, keepdims}));
            return _h.EmitTE(_bb, "std", _sinfo);
          } catch (const NotDerivable& e) {
            return call;
          }
        })
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutStatistical);
relax::Call sum(relax::Expr x, Array<IntImm> axis, bool keepdims) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.sum");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(Axes()(axis), axis, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(keepdims), keepdims, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.sum").set_body_typed(sum);
TVM_RELAX_REGISTER_OP("sum")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
                                  using namespace relax::relax2te;
                                  using _RV = runtime::TVMRetValue;
                                  static const auto* _te =
                                      runtime::Registry::Get("topi.array_api.sum");
                                  ICHECK(_te != nullptr);
                                  TensorHandler _h;
                                  try {
                                    _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                                        _h.Tensor("x", DataType::Float(32), true),
                                        {})(call->args[0]);
                                    _RV axis = Axes(_TensorNDim(x), false, true)(call->args[1]);
                                    _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
                                    return _h.AddOutput(CallGlobalFunc(_te, {x, axis, keepdims}));
                                  } catch (const NotDerivable& e) {
                                    return ReduceSInfoFallback(DataType::Void())(call, _bb);
                                  }
                                })
    .set_attr<FLegalize>(
        "FLegalize",
        [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
          using namespace relax::relax2te;
          using _RV = runtime::TVMRetValue;
          static const auto* _te = runtime::Registry::Get("topi.array_api.sum");
          ICHECK(_te != nullptr);
          TensorHandler _h;
          try {
            _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x", DataType::Float(32), true), {})(call->args[0]);
            _RV axis = Axes(_TensorNDim(x), false, true)(call->args[1]);
            _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
            StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, axis, keepdims}));
            return _h.EmitTE(_bb, "sum", _sinfo);
          } catch (const NotDerivable& e) {
            return call;
          }
        })
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutStatistical);
relax::Call variance(relax::Expr x, Array<IntImm> axis, bool keepdims) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.variance");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(Axes()(axis), axis, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(keepdims), keepdims, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.variance").set_body_typed(variance);
TVM_RELAX_REGISTER_OP("variance")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                [](const Call& call, const BlockBuilder& _bb) -> StructInfo {
                                  using namespace relax::relax2te;
                                  using _RV = runtime::TVMRetValue;
                                  static const auto* _te =
                                      runtime::Registry::Get("topi.array_api.variance");
                                  ICHECK(_te != nullptr);
                                  TensorHandler _h;
                                  try {
                                    _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                                        _h.Tensor("x", DataType::Float(32), true),
                                        {})(call->args[0]);
                                    _RV axis = Axes(_TensorNDim(x), false, true)(call->args[1]);
                                    _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
                                    return _h.AddOutput(CallGlobalFunc(_te, {x, axis, keepdims}));
                                  } catch (const NotDerivable& e) {
                                    return ReduceSInfoFallback(DataType::Void())(call, _bb);
                                  }
                                })
    .set_attr<FLegalize>(
        "FLegalize",
        [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
          using namespace relax::relax2te;
          using _RV = runtime::TVMRetValue;
          static const auto* _te = runtime::Registry::Get("topi.array_api.variance");
          ICHECK(_te != nullptr);
          TensorHandler _h;
          try {
            _RV x = TETensorFromRelaxTensor<DTypeAll, false>(
                _h.Tensor("x", DataType::Float(32), true), {})(call->args[0]);
            _RV axis = Axes(_TensorNDim(x), false, true)(call->args[1]);
            _RV keepdims = ScalarFromPrimValue<DTypeBool>()(call->args[2]);
            StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, axis, keepdims}));
            return _h.EmitTE(_bb, "variance", _sinfo);
          } catch (const NotDerivable& e) {
            return call;
          }
        })
    .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutStatistical);
// (TVM-TOOL) cc_op end def/statistical/*

}  // namespace relax
}  // namespace tvm
