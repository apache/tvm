#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include "tvm/ffi/string.h"
#include "tvm/ir/expr.h"
#include "tvm/tir/analysis.h"
#include "tvm/arith/analyzer.h"

namespace tvm::arith {

using namespace tir;
using namespace ffi;

class Z3Prover::Impl {};

TVM_DLL bool Z3Prover::CanProve(const PrimExpr & expr) { return false; }
TVM_DLL void Z3Prover::Bind(const Var& var, const Range& new_range, bool allow_override) {}
TVM_DLL void Z3Prover::Bind(const Var& var, const PrimExpr& expr, bool allow_override) {}
std::function<void()> Z3Prover::EnterConstraint(const PrimExpr& constraint, bool is_assume) { return [](){}; }
ffi::String Z3Prover::GetSMTLIB2(const ffi::Optional<PrimExpr> expr) { 
  return "; Z3 Prover is disabled.";
}
void Z3Prover::SetTimeoutMs(unsigned timeout_ms) {}
void Z3Prover::SetMaxStep(unsigned max_step) {}
void Z3Prover::CopyFrom(const Z3Prover & other) {}
ffi::String Z3Prover::GetStats() {
  return "; Z3 Prover is disabled.";
}
Z3Prover::Z3Prover(Analyzer*): impl_(nullptr) {}
TVM_DLL Z3Prover::~Z3Prover() {}

} // namespace tvm::arith