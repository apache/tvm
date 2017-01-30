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

class CodeGenCUDA : public CodeGenC {
 public:
  /*!
   * \brief Generate the C code of statement
   * \param f The function to be compiled
   * \param output_ssa Whether output ssa form.
   * \note Only call compile once,
   *  create a new codegen object each time.
   */
  std::string Compile(LoweredFunc f,
                      bool output_ssa);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_CODEGEN_CUDA_H_
