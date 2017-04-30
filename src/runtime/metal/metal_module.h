/*!
 *  Copyright (c) 2017 by Contributors
 * \file metal_module.h
 * \brief Execution handling of Metal kernels
 */
#ifndef TVM_RUNTIME_METAL_METAL_MODULE_H_
#define TVM_RUNTIME_METAL_METAL_MODULE_H_

#include <tvm/runtime/config.h>
#include <tvm/runtime/packed_func.h>
#include <memory>
#include <vector>
#include <string>
#include "../meta_data.h"

namespace tvm {
namespace runtime {
/*! \brief Maximum number of GPU supported in MetalModule. */
static constexpr const int kMetalMaxNumDevice = 32;

/*!
 * \brief create a metal module from data.
 *
 * \param data The data content.
 * \param fmt The format of the data, can be "metal" or "metallib"
 * \param fmap The map function information map of each function.
 * \param source Optional, source file
 */
Module MetalModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string source);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_METAL_METAL_MODULE_H_
