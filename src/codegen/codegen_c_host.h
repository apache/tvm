/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_c_host.h
 * \brief Generate C host code.
 */
#ifndef TVM_CODEGEN_CODEGEN_C_HOST_H_
#define TVM_CODEGEN_CODEGEN_C_HOST_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenCHost final : public CodeGenC {
 public:
  CodeGenCHost();
  void Init(bool output_ssa);
  void AddFunction(LoweredFunc f);
  std::string Finish();

  void PrintType(Type t, std::ostream& os) final; // NOLINT(*)

  // overload visitor functions
  void VisitExpr_(const Broadcast* op, std::ostream& os) final; // NOLINT(*)
  void VisitExpr_(const Call *op, std::ostream& os) final; // NOLINT(*)
  void VisitStmt_(const AssertStmt *op) final; // NOLINT(*)

 private:
  std::string module_name;
  void PrintGetFuncFromBackend(std::string func_name, std::string packed_func_name);
  void PrintFuncCall(std::string packed_func_name, int num_args);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_CODEGEN_C_HOST_H_
