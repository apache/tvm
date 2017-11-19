/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_opengl.cc
 */
#include <tvm/runtime/config.h>
#include <tvm/packed_func_ext.h>
#include <vector>
#include <string>
#include "./codegen_opengl.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace codegen {

CodeGenOpenGL::CodeGenOpenGL() {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream() << "CodeGenOpenGL::CodeGenOpenGL" << std::endl;
}

void CodeGenOpenGL::AddFunction(LoweredFunc f) {
  // TODO(zhixunt): Implement this.
  LOG_INFO.stream() << "CodeGenOpenGL::AddFunction" << std::endl;
  this->stream << "THE OPENGL FRAGMENT SHADER GOES HERE." << std::endl;
}

}  // namespace tvm
}  // namespace codegen
