/*!
 *  Copyright (c) 2018 by Contributors
 * \file codegen_vhls.h
 * \brief Utility to generate vhls code
 */
#ifndef TVM_CODEGEN_CODEGEN_VHLS_H_
#define TVM_CODEGEN_CODEGEN_VHLS_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenVivadoHLS final : public CodeGenC {
 public:
  void Init(bool output_ssa);
  void PrintType(Type t, std::ostream& os);
  void AddFunction(LoweredFunc f);
  void PreFunctionBody(LoweredFunc f);
  void VisitExpr_(const Min *op, std::ostream& os);
  void VisitExpr_(const Max *op, std::ostream& os);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_CODEGEN_VHLS_H_
