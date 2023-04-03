#include <tvm/relax/op/builtin.h>

#include "../op_common.h"
#include "../te_integration.h"
#include "../type_check.h"

namespace tvm {
namespace relax {
namespace builtin {

// (TVM-TOOL) cc_op begin def/elementwise/*
relax::Call acos(relax::Expr a) {
  static const Op& op = Op::Get("relax.acos");
  return Call(op, {FromTensor(DTypeFloat, {})(a)}, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.acos").set_body_typed(acos);
TVM_RELAX_REGISTER_OP("acos")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoFromTE("topi.array_api.acos"))
    .set_attr<FLegalize>("FLegalize", LegalizeFromTE("topi.array_api.acos", "acos"));
// (TVM-TOOL) cc_op end def/elementwise/*

}  // namespace builtin
}  // namespace relax
}  // namespace tvm
