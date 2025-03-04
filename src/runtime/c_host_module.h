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
 * \file c_module.h
 * \brief Execution handling of c kernels
 */
#ifndef TVM_RUNTIME_C_HOST_C_MODULE_H_
#define TVM_RUNTIME_C_HOST_C_MODULE_H_

#include <tvm/runtime/module.h>

#include <string>

namespace tvm {
namespace runtime {

/*!
 * \brief create a c module from data.
 * \param c_source The c source code
 */
Module CModuleCreate(std::string c_source);

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_C_HOST_C_MODULE_H_
