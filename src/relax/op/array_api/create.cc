#include <tvm/relax/op/create.h>

#include "../arg2relax.h"
#include "../op_common.h"
#include "../relax2te.h"

namespace tvm {
namespace relax {
namespace arg2relax {
relax::Expr PrimExprOr0DTensor(ObjectRef arg) {
  if (const auto* prim_expr = arg.as<PrimExprNode>()) {
    return relax::PrimValue(GetRef<PrimExpr>(prim_expr));
  } else if (const auto* relax_expr = arg.as<relax::ExprNode>()) {
    return GetRef<relax::Expr>(relax_expr);
  } else {
    LOG(FATAL) << "TypeError: Expected PrimExpr or 0-dim Tensor, but got: " << arg->GetTypeKey();
  }
}
}  // namespace arg2relax

namespace relax2te {

runtime::TVMRetValue PrimExprOr0DTensor(const relax::Expr& arg, const std::string& name,
                                        TensorHandler* h) {
  runtime::TVMRetValue ret;
  if (const auto* prim_expr = arg.as<relax::PrimValueNode>()) {
    ret = prim_expr->value;
    return ret;
  }
  if (!arg->struct_info_->IsInstance<TensorStructInfoNode>()) {
    LOG(FATAL) << "TypeError: Expected PrimExpr or 0-dim Tensor, but got: " << arg;
  }
  bool has_previous_tensor = !h->rx_tensors.empty();
  bool has_void_dtype = h->has_void_dtype;
  te::Tensor tensor = h->Tensor(name, DataType::Float(32), true)(arg);
  if (!has_void_dtype && has_previous_tensor) {
    h->has_void_dtype = false;
  }
  if (tensor->shape.size() > 0) {
    LOG(FATAL) << "TypeError: Expected PrimExpr or 0-dim Tensor, but got: " << arg;
  }
  ret = tensor;
  return ret;
}

}  // namespace relax2te

FInferStructInfo ShapeBasedCreationSInfo(int x_kind, int fill_value_index,
                                         int dtype_fallback_kind) {
  // x_kind:
  //   - 0: x is tensor
  //   - 1: x is shape
  // dtype_fallback_kind:
  //  - 0: use dtype of x
  //  - 1: use dtype of fill_value
  //  - 2: fallback to fp32
  return [=](const Call& call, const BlockBuilder& bb) -> StructInfo {
    // Handle `x`
    int x_ndim = -1;
    DataType x_dtype = DataType::Void();
    Optional<ShapeStructInfo> x_shape = NullOpt;
    Optional<Expr> x_shape_expr = NullOpt;
    if (x_kind == 0) {  // x is a tensor
      const auto* tensor = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
      CHECK(tensor != nullptr) << "TypeError: Expected `x` to be tensor, but got: "
                               << call->args[0]->struct_info_;
      x_ndim = tensor->ndim;
      x_dtype = tensor->dtype;
      x_shape_expr = tensor->shape;
      if (x_shape_expr.defined()) {
        x_shape = MatchStructInfo<ShapeStructInfo>(x_shape_expr.value());
      }
    } else if (x_kind == 1) {  // x is a shape
      const auto* shape = GetStructInfoAs<ShapeStructInfoNode>(call->args[0]);
      CHECK(shape != nullptr) << "TypeError: Expected `x` to be a shape, but got: "
                              << call->args[0]->struct_info_;
      x_ndim = shape->ndim;
      x_dtype = DataType::Void();
      x_shape_expr = NullOpt;
      x_shape = GetRef<ShapeStructInfo>(shape);
    } else {
      ICHECK(false);
    }
    // Handle `fill_value`
    DataType fill_value_dtype = DataType::Void();
    if (fill_value_index != -1) {
      const relax::Expr& fill_value = call->args[fill_value_index];
      // fill_value_dtype = DTypeOfFillValue(call->args[fill_value_index]);
      if (const auto* prim_value = fill_value.as<relax::PrimValueNode>()) {
        fill_value_dtype = prim_value->value.dtype();
      } else if (const auto* var = fill_value.as<relax::VarNode>()) {
        TensorStructInfo tensor = Downcast<TensorStructInfo>(var->struct_info_);
        if (tensor->ndim == 0) {
          fill_value_dtype = tensor->dtype;
        } else {
          LOG(FATAL) << "TypeError: Expected PrimExpr or 0-dim Tensor, but got: " << fill_value;
        }
      } else {
        LOG(FATAL) << "TypeError: Expected PrimExpr or 0-dim Tensor, but got: " << fill_value;
      }
    }
    // Infer dtype
    DataType dtype = relax2te::DType()(call->args.back());
    if (dtype.is_void()) {
      if (dtype_fallback_kind == 0) {
        dtype = x_dtype;
      } else if (dtype_fallback_kind == 1) {
        dtype = fill_value_dtype;
      } else {
        dtype = DataType::Float(32);
      }
    }
    if (x_shape.defined() && x_shape.value()->values.defined()) {
      return TensorStructInfo(ShapeExpr(x_shape.value()->values.value()), dtype);
    } else if (x_kind == 1) {
      return TensorStructInfo(call->args[0], dtype);
    } else if (x_shape_expr.defined()) {
      return TensorStructInfo(x_shape_expr.value(), dtype);
    } else {
      return TensorStructInfo(dtype, x_ndim);
    }
  };
}

StructInfo TrilTriuSInfo(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo tensor = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
  if (!tensor->IsUnknownNdim() && tensor->ndim < 2) {
    LOG(FATAL) << "TypeError: " << call->op << " requires the input tensor to have at least two "
               << "dimensions. However, the given input has " << tensor->ndim << " dimension(s).";
  }
  return tensor;
}

// (TVM-TOOL) cc_op begin def/create/*
relax::Call full(relax::Expr shape, ObjectRef fill_value, runtime::DataType dtype) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.full");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(shape, shape, _args);
  TVM_RELAX_OP_ARG_CHECK(arg2relax::PrimExprOr0DTensor(fill_value), fill_value, _args);
  TVM_RELAX_OP_ARG_CHECK(DType()(dtype), dtype, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.full").set_body_typed(full);
TVM_RELAX_REGISTER_OP("full")
    .set_attr<FInferStructInfo>("FInferStructInfo", ShapeBasedCreationSInfo(1, 1, 1))
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.full");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV shape = ShapeArrayFromShape()(call->args[0]);
        _RV fill_value = relax2te::PrimExprOr0DTensor(call->args[1], "fill_value", &_h);
        _RV dtype = DType()(call->args[2]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {shape, fill_value, dtype}));
        return _h.EmitTE(_bb, "full", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call full_like(relax::Expr x, ObjectRef fill_value, runtime::DataType dtype) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.full_like");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(arg2relax::PrimExprOr0DTensor(fill_value), fill_value, _args);
  TVM_RELAX_OP_ARG_CHECK(DType()(dtype), dtype, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.full_like").set_body_typed(full_like);
TVM_RELAX_REGISTER_OP("full_like")
    .set_attr<FInferStructInfo>("FInferStructInfo", ShapeBasedCreationSInfo(0, 1, 0))
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.full_like");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV x = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("x", DataType::Float(32), true),
                                                         {})(call->args[0]);
        _RV fill_value = relax2te::PrimExprOr0DTensor(call->args[1], "fill_value", &_h);
        _RV dtype = DType()(call->args[2]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, fill_value, dtype}));
        return _h.EmitTE(_bb, "full_like", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call ones(relax::Expr shape, runtime::DataType dtype) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.ones");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(shape, shape, _args);
  TVM_RELAX_OP_ARG_CHECK(DType()(dtype), dtype, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.ones").set_body_typed(ones);
TVM_RELAX_REGISTER_OP("ones")
    .set_attr<FInferStructInfo>("FInferStructInfo", ShapeBasedCreationSInfo(1, -1, 2))
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.ones");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV shape = ShapeArrayFromShape()(call->args[0]);
        _RV dtype = DType()(call->args[1]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {shape, dtype}));
        return _h.EmitTE(_bb, "ones", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call ones_like(relax::Expr x, runtime::DataType dtype) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.ones_like");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(DType()(dtype), dtype, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.ones_like").set_body_typed(ones_like);
TVM_RELAX_REGISTER_OP("ones_like")
    .set_attr<FInferStructInfo>("FInferStructInfo", ShapeBasedCreationSInfo(0, -1, 0))
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.ones_like");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV x = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("x", DataType::Float(32), true),
                                                         {})(call->args[0]);
        _RV dtype = DType()(call->args[1]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, dtype}));
        return _h.EmitTE(_bb, "ones_like", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call tril(relax::Expr x, PrimExpr k) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.tril");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(PrimExprToPrimValue()(k), k, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.tril").set_body_typed(tril);
TVM_RELAX_REGISTER_OP("tril")
    .set_attr<FInferStructInfo>("FInferStructInfo", TrilTriuSInfo)
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.tril");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV x = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("x", DataType::Float(32), true),
                                                         {})(call->args[0]);
        _RV k = PrimExprFromPrimValue<DTypeInt>()(call->args[1]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, k}));
        return _h.EmitTE(_bb, "tril", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call triu(relax::Expr x, PrimExpr k) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.triu");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(PrimExprToPrimValue()(k), k, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.triu").set_body_typed(triu);
TVM_RELAX_REGISTER_OP("triu")
    .set_attr<FInferStructInfo>("FInferStructInfo", TrilTriuSInfo)
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.triu");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV x = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("x", DataType::Float(32), true),
                                                         {})(call->args[0]);
        _RV k = PrimExprFromPrimValue<DTypeInt>()(call->args[1]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, k}));
        return _h.EmitTE(_bb, "triu", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call zeros(relax::Expr shape, runtime::DataType dtype) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.zeros");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(shape, shape, _args);
  TVM_RELAX_OP_ARG_CHECK(DType()(dtype), dtype, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.zeros").set_body_typed(zeros);
TVM_RELAX_REGISTER_OP("zeros")
    .set_attr<FInferStructInfo>("FInferStructInfo", ShapeBasedCreationSInfo(1, -1, 2))
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.zeros");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV shape = ShapeArrayFromShape()(call->args[0]);
        _RV dtype = DType()(call->args[1]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {shape, dtype}));
        return _h.EmitTE(_bb, "zeros", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
relax::Call zeros_like(relax::Expr x, runtime::DataType dtype) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.zeros_like");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  TVM_RELAX_OP_ARG_CHECK(DType()(dtype), dtype, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.zeros_like").set_body_typed(zeros_like);
TVM_RELAX_REGISTER_OP("zeros_like")
    .set_attr<FInferStructInfo>("FInferStructInfo", ShapeBasedCreationSInfo(0, -1, 0))
    .set_attr<FLegalize>("FLegalize", [](const BlockBuilder& _bb, const Call& call) -> relax::Expr {
      using namespace relax::relax2te;
      using _RV = runtime::TVMRetValue;
      static const auto* _te = runtime::Registry::Get("topi.array_api.zeros_like");
      ICHECK(_te != nullptr);
      TensorHandler _h;
      try {
        _RV x = TETensorFromRelaxTensor<DTypeAll, false>(_h.Tensor("x", DataType::Float(32), true),
                                                         {})(call->args[0]);
        _RV dtype = DType()(call->args[1]);
        StructInfo _sinfo = _h.AddOutput(CallGlobalFunc(_te, {x, dtype}));
        return _h.EmitTE(_bb, "zeros_like", _sinfo);
      } catch (const NotDerivable& e) {
        return call;
      }
    });
// (TVM-TOOL) cc_op end def/create/*

}  // namespace relax
}  // namespace tvm
