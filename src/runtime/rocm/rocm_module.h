/*!
 *  Copyright (c) 2017 by Contributors
 * \file rocm_module.h
 * \brief Execution handling of ROCM kernels
 */
#ifndef TVM_RUNTIME_ROCM_ROCM_MODULE_H_
#define TVM_RUNTIME_ROCM_ROCM_MODULE_H_

#include <tvm/runtime/config.h>
#include <tvm/runtime/module.h>
#include <memory>
#include <vector>
#include <string>
#include "../meta_data.h"

namespace tvm {
namespace runtime {

/*! \brief Maximum number of GPU supported in ROCMModule */
static constexpr const int kMaxNumGPUs = 32;

/*!
 * \brief create a rocm module from data.
 *
 * \param data The module data, can be hsaco
 * \param fmt The format of the data, can be "hsaco"
 * \param fmap The map function information map of each function.
 * \param rocm_source Optional, rocm source file
 */
Module ROCMModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string rocm_source,
    std::string assembly);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_ROCM_ROCM_MODULE_H_
