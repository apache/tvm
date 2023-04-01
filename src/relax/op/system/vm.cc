#include <tvm/relax/op/vm.h>

#include "../arg2relax.h"
#include "../infer_struct_info.h"
#include "../op_common.h"

namespace tvm {
namespace relax {
namespace vm {

StructInfo InferStructInfoVMAllocTensor(const Call& call, const BlockBuilder& ctx) {
  runtime::DataType dtype = Downcast<DataTypeImm>(call->args[3])->value;
  // The shape cane be expressed as
  // 1) A shape expression
  // 2) A var whose struct info is a shape
  // TODO: revisit: why not simply use call->args[2]->struct_info?
  if (const auto* shape = call->args[2].as<ShapeExprNode>()) {
    return TensorStructInfo(GetRef<Expr>(shape), dtype);
  }
  return TensorStructInfo(dtype, kUnknownNDim);
}

// (TVM-TOOL) cc_op begin def/vm/*
relax::Call alloc_storage(relax::Expr size, PrimExpr runtime_device_index,
                          runtime::DataType dtype) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.vm.alloc_storage");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(size, size, _args);
  TVM_RELAX_OP_ARG_CHECK(PrimExprToPrimValue()(runtime_device_index), runtime_device_index, _args);
  TVM_RELAX_OP_ARG_CHECK(DType()(dtype), dtype, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.vm.alloc_storage").set_body_typed(alloc_storage);
TVM_RELAX_REGISTER_OP("vm.alloc_storage")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReturnsObject);
relax::Call alloc_tensor(relax::Expr storage, PrimExpr offset, relax::Expr shape,
                         runtime::DataType dtype) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.vm.alloc_tensor");
  Array<relax::Expr> _args;
  _args.reserve(4);
  TVM_RELAX_OP_ARG_CHECK(storage, storage, _args);
  TVM_RELAX_OP_ARG_CHECK(PrimExprToPrimValue()(offset), offset, _args);
  TVM_RELAX_OP_ARG_CHECK(shape, shape, _args);
  TVM_RELAX_OP_ARG_CHECK(DType()(dtype), dtype, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.vm.alloc_tensor").set_body_typed(alloc_tensor);
TVM_RELAX_REGISTER_OP("vm.alloc_tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoVMAllocTensor);
relax::Call call_tir_dyn(relax::ExternFunc func, relax::Tuple args) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.vm.call_tir_dyn");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(func, func, _args);
  TVM_RELAX_OP_ARG_CHECK(args, args, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.vm.call_tir_dyn").set_body_typed(call_tir_dyn);
TVM_RELAX_REGISTER_OP("vm.call_tir_dyn")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReturnsVoid);
// (TVM-TOOL) cc_op end def/vm/*

}  // namespace vm
}  // namespace relax
}  // namespace tvm
