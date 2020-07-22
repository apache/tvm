/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file stackvm_module.h
 * \brief StackVM module
 */
#ifndef TVM_RUNTIME_STACKVM_STACKVM_MODULE_H_
#define TVM_RUNTIME_STACKVM_STACKVM_MODULE_H_

#include <tvm/runtime/packed_func.h>

#include <string>
#include <unordered_map>

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
Module StackVMModuleCreate(std::unordered_map<std::string, StackVM> fmap, std::string entry_func);

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_STACKVM_STACKVM_MODULE_H_
