/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_cuda.h
 * \brief Utility to generate cuda code
 */
#ifndef TVM_CODEGEN_CODEGEN_CUDA_H_
#define TVM_CODEGEN_CODEGEN_CUDA_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "./codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenCUDA final : public CodeGenC {
 public:
  void Init(bool output_ssa);
  void AddFunction(LoweredFunc f);
  // override behavior
  void VisitStmt_(const ir::For* op) final;
  void PrintStorageSync(const Call* op) final;
  void PrintStorageScope(const std::string& scope, std::ostream& os) final;  // NOLINT(*)
  void PrintVecBinaryOp(
      const std::string&op, Type t,
      Expr lhs, Expr rhs, std::ostream& os) final;  // NOLINT(*)
  void PrintType(Type t, std::ostream& os) const final; // NOLINT(*)
  void PrintVecElemLoad(
      const std::string& vec, Type t, int i, std::ostream& os) final;  // NOLINT(*)
  void PrintVecElemStore(
      const std::string& vec, Type t, int i, const std::string& value) final;

  // overload visitor
  void VisitExpr_(const Broadcast* op, std::ostream& os) final; // NOLINT(*)
  void VisitStmt_(const Evaluate *op) final;

 private:
  // magic number to add pragma unroll to it.
  // used to generate code that is compact but still unrolls.
  int max_auto_unroll_{32};
  // Whether global barrier is needed.
  bool need_global_barrier_{false};
  // Global barrier state
  std::string vid_global_barrier_state_;
  // Global barrier expected node.
  std::string vid_global_barrier_expect_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_CODEGEN_CUDA_H_
