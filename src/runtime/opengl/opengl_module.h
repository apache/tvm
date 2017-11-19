/*!
 *  Copyright (c) 2017 by Contributors
 * \file opengl_module.h
 * \brief Execution handling of OpenGL kernels
 */
#ifndef TVM_RUNTIME_OPENGL_OPENGL_MODULE_H_
#define TVM_RUNTIME_OPENGL_OPENGL_MODULE_H_

#include <tvm/runtime/config.h>
#include <tvm/runtime/packed_func.h>
#include <memory>
#include <vector>
#include <string>
#include "../meta_data.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Create an OpenGL module from data.
 *
 * \param data The module data.
 * \param fmt The format of the data,
 * \param fmap The map function information map of each function.
 */
Module OpenGLModuleCreate(std::string data,
                          std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap);

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_OPENGL_OPENGL_MODULE_H_
