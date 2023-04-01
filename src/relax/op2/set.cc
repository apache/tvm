#include "../op/arg2relax.h"
#include "../op/op_common.h"
#include "../op/relax2te.h"

namespace tvm {
namespace relax {
namespace {
#undef TVM_RELAX_REGISTER_OP
#undef TVM_REGISTER_GLOBAL
#define TVM_REGISTER_GLOBAL(OpName)                   \
  TVM_STR_CONCAT(TVM_FUNC_REG_VAR_DEF, __COUNTER__) = \
      ::tvm::runtime::Registry::Register("__" OpName)
#define TVM_RELAX_REGISTER_OP(OpName) TVM_REGISTER_OP("__" OpName)

}  // namespace
}  // namespace relax
}  // namespace tvm
