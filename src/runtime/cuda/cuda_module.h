/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_module.h
 * \brief Execution handling of CUDA kernels
 */
#ifndef TVM_RUNTIME_CUDA_CUDA_MODULE_H_
#define TVM_RUNTIME_CUDA_CUDA_MODULE_H_

#include <tvm/runtime/config.h>
#include <tvm/runtime/module.h>
#include <memory>
#include <vector>
#include <string>
#include "../meta_data.h"

namespace tvm {
namespace runtime {

/*! \brief Maximum number of GPU supported in CUDAModule */
static constexpr const int kMaxNumGPUs = 32;

/*!
 * \brief create a cuda module from data.
 *
 * \param data The module data, can be ptx, cubin
 * \param fmt The format of the data, can be "ptx", "cubin"
 * \param fmap The map function information map of each function.
 * \param cuda_source Optional, cuda source file
 */
Module CUDAModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string cuda_source);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CUDA_CUDA_MODULE_H_
