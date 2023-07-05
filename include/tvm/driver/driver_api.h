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

#include <tvm/ir/global_var_supply.h>
#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/support/with.h>
#include <tvm/target/target.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/function.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
using tvm::transform::Pass;

/*!
 * \brief Configures and returns the composite Pass for the fused module (pre split) that contains
 * device and host code.
 * \param mixed_mod The original mixed module.
 * \param target The device Target.
 * \return The composite Pass for the fused module.
//  */
TVM_DLL transform::Sequential MixedModulePassManager(IRModule mixed_mod, Target target);

/*!
 * \brief Configures and returns the composite Pass for the device Target after device/host from
 * mixed module.
 * \param mixed_mod The optimized mixed module.
 * \param target The device Target.
 * \return The composite Pass for the device module.
 */
TVM_DLL transform::Sequential DeviceModulePassManager(IRModule mixed_mod, Target target);

/*!
 * \brief Configures and returns the composite Pass for the host Target after device/host from mixed
 * module.
 * \param mixed_mod The optimized mixed module.
 * \param target_host The host Target.
 * \return The composite Pass for the host module.
 */
TVM_DLL transform::Sequential HostModulePassManager(IRModule mixed_mod, Target target_host);

/*!
 * \brief Lower an IRModule (optimize with it with the pass list defined in CreatePassList)
 * \param mod The IRmodule to lower
 * \param simple_mode Disables the loop partition pass. Defaults to false.
 * \return The result module.
 */
TVM_DLL IRModule LowerModule(IRModule mod, bool simple_mode = false);

/*!
 * \brief Lower a primfunc and name (convert to IRModule, and optimize it with the pass list
 * defined in CreatePassList)
 * \param func The PrimFunc to lower
 * \param name The name of the lowered function.
 * \param simple_mode Disables the loop partition pass. Defaults to false.
 * \return The result module.
 */
TVM_DLL IRModule LowerPrimFunc(tvm::tir::PrimFunc func, const std::string& name,
                               bool simple_mode = false);

/*!
 * \brief Build an IRModule given a TE schedule, args and binds. This function also applies
 * the lowering passes defined in CreatePassList.
 * \param sch The TE schedule to lower.
 * \param args The arguments to the function.
 * \param name The name of the lowered function.
 * \param binds Buffer assignments.
 * \param global_var_supply The GlobalVarSupply to be used in the module.
 * \param simple_mode Disables the loop partition pass. Defaults to false.
 * \return The result module.
 */

TVM_DLL IRModule LowerSchedule(te::Schedule sch, const Array<te::Tensor>& args,
                               const std::string& name,
                               const std::unordered_map<te::Tensor, tir::Buffer>& binds,
                               GlobalVarSupply global_var_supply, bool simple_mode = false);

/*!
 * \brief Build an IRModule given a TE schedule, args and binds. This function also applies
 * the lowering passes defined in CreatePassList.
 * \param sch The TE schedule to lower.
 * \param args The arguments to the function (Array of Tensor, Buffer and Vars)
 * \param name The name of the lowered function.
 * \param binds Buffer assignments.
 * \param global_var_supply The GlobalVarSupply to be used in the module.
 * \param simple_mode Disables the loop partition pass. Defaults to false.
 * \return The result module.
 */
TVM_DLL IRModule LowerSchedule(te::Schedule sch, const Array<ObjectRef>& args,
                               const std::string& name,
                               const std::unordered_map<te::Tensor, tir::Buffer>& binds,
                               GlobalVarSupply global_var_supply, bool simple_mode = false);

/*!
 * \brief Create an IRModule out of a TE Schedule. It does not apply lowering passes. If you want
 * to apply lowering passes as well, use LowerSchedule.
 * \param sch The schedule
 * \param args The arguments to the function.
 * \param name The name of the lowered function.
 * \param binds Buffer assignments.
 * \param global_var_supply The GlobalVarSupply to be used in the module and when creating
 * GlobalVars.
 * \return The result module.
 */
IRModule ScheduleToModule(te::Schedule sch, const Array<ObjectRef>& args, const std::string& name,
                          const std::unordered_map<te::Tensor, tir::Buffer>& binds,
                          GlobalVarSupply global_var_supply);
/*!
 * \brief Build a device and host module for a specific target from an IRModule.
 * \param funcs The functions to be built.
 * \param target The target device to build for.
 * \param target_host The target for building host code. To use the default, pass Target()
 * \return The built module.
 */
TVM_DLL runtime::Module build(const IRModule& funcs, const Target& target,
                              const Target& target_host);

/*!
 * \brief Build a device and host module for a specific target from a map
 * contains target to IRModule. This function is used
 * for heterogeneous build.
 * \param input The map contains target to an IRModule.
 * \param target_host The target for building host code. To use the default,
 *        pass Target().
 * \return The built module that contains code for different processors.
 */
TVM_DLL runtime::Module build(const Map<Target, IRModule>& input, const Target& target_host);

/*!
 * \brief Build a device and host module for a specific target from a map
 * contains target to IRModule. This function is used
 * for heterogeneous build.
 * \param input The map contains target string to an  IRModule.
 * \param target_host The target for building host code. To use the default,
 *        pass Target().
 * \return The built module that contains code for different processors.
 */
TVM_DLL runtime::Module build(const Map<String, IRModule>& input, const Target& target_host);
}  // namespace tvm

#endif  // TVM_DRIVER_DRIVER_API_H_
