/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_opengl.h
 * \brief Generate OpenGL device code.
 */
#ifndef TVM_CODEGEN_CODEGEN_OPENGL_H_
#define TVM_CODEGEN_CODEGEN_OPENGL_H_

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <string>
#include "./codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenOpenGL final : public CodeGenC {
 public:
  explicit CodeGenOpenGL();
  void AddFunction(LoweredFunc f);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_CODEGEN_CODEGEN_OPENGL_H_
