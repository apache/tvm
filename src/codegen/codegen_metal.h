/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_metal.h
 * \brief Generate Metal device code.
 */
#ifndef TVM_CODEGEN_CODEGEN_METAL_H_
#define TVM_CODEGEN_CODEGEN_METAL_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "./codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenMetal final : public CodeGenC {
 public:
  CodeGenMetal();
  void AddFunction(LoweredFunc f);
  // override print thread tag.
  void PrintArgUnionDecl();
  void InitFuncState(LoweredFunc f) final;
  void PrintStorageScope(const std::string& scope, std::ostream& os) final; // NOLINT(*)
  void PrintStorageSync(const Call* op) final;  // NOLINT(*)
  void PrintType(Type t, std::ostream& os) const final; // NOLINT(*)
  void BindThreadIndex(const IterVar& iv) final;  // NOLINT(*)
  // overload visitor
  void VisitExpr_(const Broadcast* op, std::ostream& os) final; // NOLINT(*)

 private:
  int thread_index_bits_{32};
};
}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_CODEGEN_METAL_H_
