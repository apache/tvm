#include <tvm/relax/op/basic.h>

#include "../infer_struct_info.h"
#include "../op_common.h"
#include "../type_check.h"

namespace tvm {
namespace relax {

StructInfo InferStructInfoShapeOf(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo tensor_sinfo = MatchStructInfo<TensorStructInfo>(call->args[0]).value();
  // Case 1. Unknown `ndim`
  if (tensor_sinfo->IsUnknownNdim()) {
    return ShapeStructInfo(kUnknownNDim);
  }
  // Case 2. Known `ndim` but unknown `shape`
  if (!tensor_sinfo->shape.defined()) {
    return ShapeStructInfo(tensor_sinfo->ndim);
  }
  // Case 3. Known `ndim` and `shape`, and the shape is a relax Var
  if (tensor_sinfo->shape->IsInstance<VarNode>()) {
    return ShapeStructInfo(tensor_sinfo->ndim);
  }
  // Case 4. Known `ndim` and `shape`, and the shape is a relax ShapeExpr
  return ShapeStructInfo(Downcast<ShapeExpr>(tensor_sinfo->shape)->values);
}

// (TVM-TOOL) cc_op begin def/basic/*
relax::Call call_builtin_with_ctx(relax::ExternFunc func, relax::Tuple args,
                                  relax::StructInfo sinfo_args) {
  static const Op& op = Op::Get("relax.call_builtin_with_ctx");
  return Call(op, {FromExternFunc()(func), FromTupleExpr()(args)}, Attrs(nullptr),
              {FromStructInfo()(sinfo_args)});
}
TVM_REGISTER_GLOBAL("relax.op.call_builtin_with_ctx").set_body_typed(call_builtin_with_ctx);
TVM_RELAX_REGISTER_OP("call_builtin_with_ctx")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReturnFirstSInfoArg);
relax::Call call_dps_packed(relax::ExternFunc func, relax::Tuple args,
                            relax::StructInfo out_sinfo) {
  static const Op& op = Op::Get("relax.call_dps_packed");
  return Call(op, {FromExternFunc()(func), FromTupleExpr()(args)}, Attrs(nullptr),
              {FromStructInfo()(out_sinfo)});
}
TVM_REGISTER_GLOBAL("relax.op.call_dps_packed").set_body_typed(call_dps_packed);
TVM_RELAX_REGISTER_OP("call_dps_packed")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReturnFirstSInfoArg);
relax::Call call_tir(tvm::GlobalVar gvar, relax::Tuple args, relax::StructInfo out_sinfo,
                     Optional<relax::Expr> tir_vars) {
  static const Op& op = Op::Get("relax.call_tir");
  return Call(
      op,
      VariadicArgs(
          Array<Optional<relax::Expr>>{FromGlobalVar()(gvar), FromTupleExpr()(args), tir_vars}, 2),
      Attrs(nullptr), {FromStructInfo()(out_sinfo)});
}
TVM_REGISTER_GLOBAL("relax.op.call_tir").set_body_typed(call_tir);
TVM_RELAX_REGISTER_OP("call_tir")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReturnFirstSInfoArg);
relax::Call invoke_closure(relax::Expr closure, relax::Tuple args, relax::StructInfo sinfo_args) {
  static const Op& op = Op::Get("relax.invoke_closure");
  return Call(op, {FromAnyRelaxExpr()(closure), FromTupleExpr()(args)}, Attrs(nullptr),
              {FromStructInfo()(sinfo_args)});
}
TVM_REGISTER_GLOBAL("relax.op.invoke_closure").set_body_typed(invoke_closure);
TVM_RELAX_REGISTER_OP("invoke_closure")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReturnFirstSInfoArg);
relax::Call make_closure(tvm::GlobalVar func, relax::Tuple args) {
  static const Op& op = Op::Get("relax.make_closure");
  return Call(op, {FromGlobalVar()(func), FromTupleExpr()(args)}, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.make_closure").set_body_typed(make_closure);
TVM_RELAX_REGISTER_OP("make_closure")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReturnsObject);
relax::Call null_value() {
  static const Op& op = Op::Get("relax.null_value");
  return Call(op, {}, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.null_value").set_body_typed(null_value);
TVM_RELAX_REGISTER_OP("null_value")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReturnsObject);
relax::Call shape_of(relax::Expr expr) {
  static const Op& op = Op::Get("relax.shape_of");
  return Call(op, {FromTensor(DTypeAll, {})(expr)}, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.shape_of").set_body_typed(shape_of);
TVM_RELAX_REGISTER_OP("shape_of")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoShapeOf);
// (TVM-TOOL) cc_op end def/basic/*

}  // namespace relax
}  // namespace tvm
