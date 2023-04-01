#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>

namespace tvm {
namespace relax {

inline StructInfo InferStructInfoReturnFirstSInfoArg(const Call& call, const BlockBuilder& ctx) {
  if (call->sinfo_args.size() != 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "sinfo_args should have exact 1 output struct info.");
  }
  return call->sinfo_args[0];
}

inline StructInfo InferStructInfoReturnsObject(const Call& call, const BlockBuilder& ctx) {
  return ObjectStructInfo();
}

inline StructInfo InferStructInfoReturnsVoid(const Call& call, const BlockBuilder& ctx) {
  return TupleStructInfo(Array<StructInfo>{});
}

}  // namespace relax
}  // namespace tvm
