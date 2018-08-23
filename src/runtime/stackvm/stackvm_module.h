/*!
 *  Copyright (c) 2018 by Contributors
 * \file stackvm_module.h
 * \brief StackVM module
 */
#ifndef TVM_RUNTIME_STACKVM_STACKVM_MODULE_H_
#define TVM_RUNTIME_STACKVM_STACKVM_MODULE_H_

#include <tvm/runtime/packed_func.h>
#include <string>
#include "stackvm.h"

namespace tvm {
namespace runtime {
/*!
 * \brief create a stackvm module
 *
 * \param fmap The map from name to function
 * \param entry_func The entry function name.
 * \return The created module
 */
Module StackVMModuleCreate(std::unordered_map<std::string, StackVM> fmap,
                           std::string entry_func);

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_STACKVM_STACKVM_MODULE_H_
