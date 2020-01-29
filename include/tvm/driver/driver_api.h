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
 * \file tvm/driver/driver_api.h
 * \brief Compiler driver APIs to drive the compilation.
 *
 * This module provides end-to-end utils to drive the compilation process.
 * We adopt the term "compiler driver" in common compiler infrastructures.
 * Note that a compiler driver is different from "runtime drivers".
 * Most of runtime related code are defined in the runtime folder instead.
 */
#ifndef TVM_DRIVER_DRIVER_API_H_
#define TVM_DRIVER_DRIVER_API_H_

#include <tvm/runtime/packed_func.h>
#include <tvm/target/target.h>
#include <tvm/support/with.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/lowered_func.h>

#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
/*!
* \brief Build a LoweredFunc given a schedule, args and binds
* \param sch The schedule to lower.
* \param args The arguments to the function.
* \param name The name of the lowered function.
* \param binds Buffer assignments.
* \param config The build configuration.
* \return The lowered function.
*/
TVM_DLL Array<tir::LoweredFunc> lower(
    te::Schedule sch,
    const Array<te::Tensor>& args,
    const std::string& name,
    const std::unordered_map<te::Tensor, tir::Buffer>& binds,
    const BuildConfig& config);
/*!
* \brief Split host/device function and running necessary pass before build
* \param funcs The functions to be built.
* \param target The target device to build for.
* \param target_host The target for building host code. To use the default, pass Target()
* \param config The build configuration.
* \return The Array<Array<LoweredFunc>> with 2 elements. First is host function Array,
          second is device function array
*/
TVM_DLL Array<Array<tir::LoweredFunc> > split_dev_host_funcs(
    const Array<tir::LoweredFunc>& funcs,
    const Target& target,
    const Target& target_host,
    const BuildConfig& config);

/*!
* \brief Build a device and host module for a specific target from an array of lowered functions.
* \param funcs The functions to be built.
* \param target The target device to build for.
* \param target_host The target for building host code. To use the default, pass Target()
* \param config The build configuration.
* \return The built module.
*/
TVM_DLL runtime::Module build(const Array<tir::LoweredFunc>& funcs,
                              const Target& target,
                              const Target& target_host,
                              const BuildConfig& config);

/*!
 * \brief Build a device and host module for a specific target from a map
 * contains target to a list of lowered functions pairs. This function is used
 * for heterogeneous build.
 * \param input The map contains target to a list of lowered functions pairs.
 * \param target_host The target for building host code. To use the default,
 *        pass Target().
 * \param config The build configuration.
 * \return The built module that contains code for different processors.
 */
TVM_DLL runtime::Module build(const Map<Target, Array<tir::LoweredFunc>>& input,
                              const Target& target_host,
                              const BuildConfig& config);

/*!
 * \brief Build a device and host module for a specific target from a map
 * contains target to a list of lowered functions pairs. This function is used
 * for heterogeneous build.
 * \param input The map contains target string to a list of lowered functions
 *        pairs.
 * \param target_host The target for building host code. To use the default,
 *        pass Target().
 * \param config The build configuration.
 * \return The built module that contains code for different processors.
 */
TVM_DLL runtime::Module build(const Map<std::string, Array<tir::LoweredFunc>>& input,
                              const Target& target_host,
                              const BuildConfig& config);
}  // namespace tvm

#endif  // TVM_DRIVER_DRIVER_API_H_
