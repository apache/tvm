/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_opencl.h
 * \brief Utility to generate opencl code
 */
#ifndef TVM_CODEGEN_CODEGEN_OPENCL_H_
#define TVM_CODEGEN_CODEGEN_OPENCL_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "./codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenOpenCL : public CodeGenC {
 public:
  void AddFunction(LoweredFunc f);
  // override print thread tag.
  void InitFuncState(LoweredFunc f) final;
  void PrintThreadIndexExpr(
      std::string tag, std::ostream& os) final;  // NOLINT(*)
  void PrintStorageScope(const std::string& scope, std::ostream& os) final; // NOLINT(*)
  void PrintStorageSync(const std::string& scope) final;  // NOLINT(*)
  void PrintType(Type t, std::ostream& os) const final; // NOLINT(*)
  std::string GetVecLoad(const Variable* buffer,
                         Type t, Expr base) final;
  void PrintVecStore(const Variable* buffer,
                      Type t, Expr base,
                      const std::string& value) final;  // NOLINT(*)
  // the address of load/store
  void PrintVecAddr(const Variable* buffer, Type t,
                    Expr base, std::ostream& os);  // NOLINT(*)
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_CODEGEN_OPENCL_H_
