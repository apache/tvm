/*!
 *  Copyright (c) 2018 by Contributors
 * \file sdaccel_module.h
 * \brief Execution handling of OPENCL kernels for SDAccel FPGAs
 */
#ifndef TVM_RUNTIME_OPENCL_SDACCEL_SDACCEL_MODULE_H_
#define TVM_RUNTIME_OPENCL_SDACCEL_SDACCEL_MODULE_H_

#include <tvm/runtime/packed_func.h>
#include <memory>
#include <vector>
#include <string>
#include "../../meta_data.h"

namespace tvm {
namespace runtime {
/*!
 * \brief create a opencl module for SDAccel from data.
 *
 * \param data The module data.
 * \param fmt The format of the data, can be "xclbin", "awsxclbin"
 * \param fmap The map function information map of each function.
 */
Module SDAccelModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string source);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_OPENCL_SDACCEL_SDACCEL_MODULE_H_
