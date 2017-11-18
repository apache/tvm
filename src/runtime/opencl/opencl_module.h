/*!
 *  Copyright (c) 2017 by Contributors
 * \file opencl_module.h
 * \brief Execution handling of OPENCL kernels
 */
#ifndef TVM_RUNTIME_OPENCL_OPENCL_MODULE_H_
#define TVM_RUNTIME_OPENCL_OPENCL_MODULE_H_

#include <tvm/runtime/config.h>
#include <tvm/runtime/packed_func.h>
#include <memory>
#include <vector>
#include <string>
#include "../meta_data.h"

namespace tvm {
namespace runtime {
/*!
 * \brief create a opencl module from data.
 *
 * \param data The module data.
 * \param fmt The format of the data, can be "clbin", "cl"
 * \param fmap The map function information map of each function.
 */
Module OpenCLModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_OPENCL_OPENCL_MODULE_H_
