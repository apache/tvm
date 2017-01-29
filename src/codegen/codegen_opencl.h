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
  /*!
   * \brief Generate the OpenCL code of statement
   * \param f The function to be compiled
   * \param output_ssa Whether output ssa form.
   * \note Only call compile once,
   *  create a new codegen object each time.
   */
  std::string Compile(LoweredFunc f,
                      bool output_ssa);
  // override print thread tag.
  void PrintThreadTagExpr(
      std::string thread_tag, std::ostream& os) const final;  // NOLINT(*)
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_CODEGEN_OPENCL_H_
