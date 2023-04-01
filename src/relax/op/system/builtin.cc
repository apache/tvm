#include <tvm/relax/op/builtin.h>

#include "../arg2relax.h"
#include "../op_common.h"

namespace tvm {
namespace relax {
namespace builtin {

StructInfo InferStructInfoAllocTensor(const Call& call, const BlockBuilder& ctx) {
  return TensorStructInfo(Downcast<ShapeExpr>(call->args[0]),
                          Downcast<DataTypeImm>(call->args[1])->value);
}

StructInfo InferStructInfoIdentical(const Call& call, const BlockBuilder& ctx) {
  ICHECK_EQ(call->args.size(), 1);
  return Downcast<StructInfo>(call->args[0]->struct_info_);
}

// (TVM-TOOL) cc_op begin def/builtin/*
relax::Call alloc_tensor(relax::Expr shape, runtime::DataType dtype, int64_t runtime_device_index) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.builtin.alloc_tensor");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(shape, shape, _args);
  TVM_RELAX_OP_ARG_CHECK(DType()(dtype), dtype, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(false)(runtime_device_index), runtime_device_index,
                         _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.builtin.alloc_tensor").set_body_typed(alloc_tensor);
TVM_RELAX_REGISTER_OP("builtin.alloc_tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAllocTensor);
relax::Call stop_lift_params(relax::Expr x) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.builtin.stop_lift_params");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(x, x, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.builtin.stop_lift_params").set_body_typed(stop_lift_params);
TVM_RELAX_REGISTER_OP("builtin.stop_lift_params")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoIdentical);
// (TVM-TOOL) cc_op end def/builtin/*

}  // namespace builtin
}  // namespace relax
}  // namespace tvm
