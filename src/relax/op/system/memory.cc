#include <tvm/relax/op/memory.h>

#include "../infer_struct_info.h"
#include "../op_common.h"
#include "../type_check.h"

namespace tvm {
namespace relax {
namespace memory {

StructInfo InferStructInfoMemoryAllocTensor(const Call& call, const BlockBuilder& ctx) {
  return TensorStructInfo(call->args[2], Downcast<DataTypeImm>(call->args[3])->value);
}

// (TVM-TOOL) cc_op begin def/memory/*
relax::Call alloc_storage(relax::Expr size, PrimExpr virtual_device_index, String storage_scope,
                          runtime::DataType dtype) {
  static const Op& op = Op::Get("relax.memory.alloc_storage");
  return Call(op,
              {FromShape()(size), FromPrimExpr(DTypeInt)(virtual_device_index),
               FromStr()(storage_scope), FromDType()(dtype)},
              Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.memory.alloc_storage").set_body_typed(alloc_storage);
TVM_RELAX_REGISTER_OP("memory.alloc_storage")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReturnsObject);
relax::Call alloc_tensor(relax::Expr storage, PrimExpr offset, relax::Expr shape,
                         runtime::DataType dtype) {
  static const Op& op = Op::Get("relax.memory.alloc_tensor");
  return Call(op,
              {FromAnyRelaxExpr()(storage), FromPrimExpr(DTypeInt)(offset), FromShape()(shape),
               FromDType()(dtype)},
              Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.memory.alloc_tensor").set_body_typed(alloc_tensor);
TVM_RELAX_REGISTER_OP("memory.alloc_tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoMemoryAllocTensor);
relax::Call kill_storage(relax::Expr storage) {
  static const Op& op = Op::Get("relax.memory.kill_storage");
  return Call(op, {FromAnyRelaxExpr()(storage)}, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.memory.kill_storage").set_body_typed(kill_storage);
TVM_RELAX_REGISTER_OP("memory.kill_storage")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReturnsVoid);
relax::Call kill_tensor(relax::Expr tensor) {
  static const Op& op = Op::Get("relax.memory.kill_tensor");
  return Call(op, {FromTensor(DTypeAll, {})(tensor)}, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.memory.kill_tensor").set_body_typed(kill_tensor);
TVM_RELAX_REGISTER_OP("memory.kill_tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoReturnsVoid);
// (TVM-TOOL) cc_op end def/memory/*

}  // namespace memory
}  // namespace relax
}  // namespace tvm
