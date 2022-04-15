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
 * \file src/driver/driver_api.h
 * \brief Internal compiler driver APIs to drive the compilation.
 *
 * This module provides functionality that may be called internally
 * within TVM, but is not part of the public-facing API.
 */
#ifndef TVM_DRIVER_INTERNAL_DRIVER_API_H_
#define TVM_DRIVER_INTERNAL_DRIVER_API_H_

#include <tvm/ir/module.h>
#include <tvm/target/target.h>

namespace tvm {

/*!
 * \brief Build a device and host module for a specific target from a map
 * contains target to IRModule. This function is used
 * for heterogeneous build.
 * \param input The map contains target to an IRModule.
 * \param target_host The target for building host code. To use the default,
 *        pass Target().
 * \return The built module that contains code for different processors.
 */
runtime::Module TIRToRuntime(const Map<Target, IRModule>& input, const Target& target_host);

}  // namespace tvm

#endif  // TVM_DRIVER_INTERNAL_DRIVER_API_H_
