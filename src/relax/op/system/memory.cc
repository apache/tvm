#include <tvm/relax/op/memory.h>

#include "../arg2relax.h"
#include "../infer_struct_info.h"
#include "../op_common.h"

namespace tvm {
namespace relax {
namespace memory {

StructInfo InferStructInfoMemoryAllocTensor(const Call& call, const BlockBuilder& ctx) {
  return TensorStructInfo(call->args[2], Downcast<DataTypeImm>(call->args[3])->value);
}

// (TVM-TOOL) cc_op begin def/memory/*
relax::Call alloc_storage(relax::Expr size, PrimExpr virtual_device_index, String storage_scope,
                          runtime::DataType dtype) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.memory.alloc_storage");
  Array<relax::Expr> _args;
  _args.reserve(4);
  TVM_RELAX_OP_ARG_CHECK(size, size, _args);
  TVM_RELAX_OP_ARG_CHECK(PrimExprToPrimValue()(virtual_device_index), virtual_device_index, _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(storage_scope), storage_scope, _args);
  TVM_RELAX_OP_ARG_CHECK(DType()(dtype), dtype, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.memory.alloc_storage").set_body_typed(alloc_storage);
TVM_RELAX_REGISTER_OP("memory.alloc_storage")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReturnsObject);
relax::Call alloc_tensor(relax::Expr storage, PrimExpr offset, relax::Expr shape,
                         runtime::DataType dtype) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.memory.alloc_tensor");
  Array<relax::Expr> _args;
  _args.reserve(4);
  TVM_RELAX_OP_ARG_CHECK(storage, storage, _args);
  TVM_RELAX_OP_ARG_CHECK(PrimExprToPrimValue()(offset), offset, _args);
  TVM_RELAX_OP_ARG_CHECK(shape, shape, _args);
  TVM_RELAX_OP_ARG_CHECK(DType()(dtype), dtype, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.memory.alloc_tensor").set_body_typed(alloc_tensor);
TVM_RELAX_REGISTER_OP("memory.alloc_tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoMemoryAllocTensor);
relax::Call kill_storage(relax::Expr storage) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.memory.kill_storage");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(storage, storage, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.memory.kill_storage").set_body_typed(kill_storage);
TVM_RELAX_REGISTER_OP("memory.kill_storage")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReturnsVoid);
relax::Call kill_tensor(relax::Expr tensor) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.memory.kill_tensor");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(tensor, tensor, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.memory.kill_tensor").set_body_typed(kill_tensor);
TVM_RELAX_REGISTER_OP("memory.kill_tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReturnsVoid);
// (TVM-TOOL) cc_op end def/memory/*

}  // namespace memory
}  // namespace relax
}  // namespace tvm
