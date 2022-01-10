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
 * \file relay/backend/tir_compiler.h
 *  * \brief Internal compilation layer which lowers Relay "primitive functions" to TIR PrimFns.
 *
 *
 * This represents the new design of the Relay compilation flow and will replace the interface
 * contained in compile_engine.h as we migrate towards a standard pass based lowering of
 * Relay functions.
 *
 * This files provides an internal API which lowers Relay programs to components which
 * can be combined with TVM produced kernels to compile an entire program.
 *
 * The result of lowering contains a combination of `runtime::Module`s produced by external
 * compilers and a set of lowered PrimFns which can be code generated for targets.
 */
#ifndef TVM_RELAY_BACKEND_TE_COMPILER_H_
#define TVM_RELAY_BACKEND_TE_COMPILER_H_

#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op_strategy.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/module.h>
#include <tvm/topi/elemwise.h>

#include <functional>
#include <string>
#include <unordered_map>

#include "../transforms/infer_layout_utils.h"
#include "../transforms/pass_utils.h"
#include "./te_compiler_cache.h"
#include "./utils.h"

namespace tvm {
namespace relay {
namespace tec {

// TODO(@jroesch, @chrisS) these should be a tvm::Map for uniformity sake
// we should a version of context which works in Map
using TargetMap = std::unordered_map<DLDeviceType, Target, backend::EnumClassHash>;
using DeviceMap =
    std::unordered_map<Expr, tvm::Device, runtime::ObjectPtrHash, runtime::ObjectPtrEqual>;
using ProcessFn = std::function<void(BaseFunc)>;

/*!
 * \brief A compiler which lowers primitive Relay functions to tensor expressions
 * and schedules them into TIR functions.
 */
class TECompilerNode : public Object {
 public:
  /*! \brief destructor */
  virtual ~TECompilerNode() {}
  /*!
   * \brief Get lowered result.
   * \param key The key to the cached function.
   * \return The result.
   */
  virtual CachedFunc Lower(const CCacheKey& key, std::function<String(String)> mangle_fn) = 0;

  /*!
   * \brief Get lowered result.
   * \param key The key to the cached function.
   * \return The result.
   */
  virtual CachedFunc Lower(const CCacheKey& key, const String mod_name) = 0;

  /* Return all functions which have been lowered by the compiler in an IRModule, annotated with
   * their target. */
  virtual IRModule GetLoweredFunctions() = 0;

  /*!
   * \brief Just in time compile to get a PackedFunc.
   * \param key The key to the cached function.
   * \return The result.
   */
  virtual PackedFunc JIT(const CCacheKey& key) = 0;
  /*!
   * \brief Lower the shape function.
   * \param key The key to the cached function.
   * \return The result.
   */
  virtual CachedFunc LowerShapeFunc(const CCacheKey& key) = 0;
  /*!
   * \brief Lower the external function using external codegen tools.
   * \return The runtime modules for each needed external codegen tool.
   */
  virtual tvm::Array<tvm::runtime::Module> LowerExternalFunctions() = 0;

  /*!
   * \brief Update \p module to remove functions marked with the "Compiler" attribute and replace
   * them with their 'external' representation using the "ExternalSymbol" attribute.
   *
   * TODO(mbs): This is a stepping stone while we migrate to a more official representation
   * of 'external functions' in the IRModule and allow lowering to incrementally updatethe
   * module stead of forcing everything via the cache.
   *
   */
  virtual void AddExterns(IRModule module) = 0;

  /*!
   * \brief Get C Device API context mapping
   * \return Map of GlobalVar to associated C Device API context name (either Target or kCompiler
   * annotated)
   */
  virtual Map<GlobalVar, String> GetDeviceContexts() = 0;
  virtual void SetDeviceContexts(const Map<GlobalVar, String>& device_contexts) = 0;

  virtual Map<String, Integer> GetOpWeights() const = 0;

  /*! \brief clear the cache. */
  virtual void Clear() = 0;

  void VisitAttrs(AttrVisitor*) {}

  static constexpr const char* _type_key = "relay.TECompiler";
  TVM_DECLARE_FINAL_OBJECT_INFO(TECompilerNode, Object);
};

/*! \brief cache entry used in compile engine */
class TECompiler : public ObjectRef {
 public:
  explicit TECompiler(Optional<IRModule> opt_mod = {});
  explicit TECompiler(ObjectPtr<Object> n) : ObjectRef(n) {}
  TECompilerNode* operator->() { return static_cast<TECompilerNode*>(get_mutable()); }
  using ContainerType = TECompilerNode;
  TVM_DLL static TECompiler& Global();
};

/*!
 * \brief A function to create the function metadata for an input function (ie calculate buffer
 * input/output sizes)
 * \param func The function to calculate function metadata for
 * \param function_metadata The map that stores all the function metadatas
 * \param workspace_byte_alignment Byte alignment for allocations
 */
void UpdateFunctionMetadata(BaseFunc relay_func,
                            Map<String, backend::FunctionInfo>& function_metadata,  // NOLINT(*)
                            Integer workspace_byte_alignment = 16);

/*!
 * \brief Obtain the Target from the device type.
 * If homogenous compilation, this will return the only target.
 * If heterogeneous compilation, this will select the associated target using the
 * targets_ Map.
 *
 * \param dev_type
 * \return Target
 */
Target GetTargetFromInteger(DLDeviceType dev_type, tec::TargetMap targets);

/*!
 * \brief Update the "main" control function's metadata
 *
 * \param mod The module
 * \param targets Map of targets
 * \return function_infos Function info for each function in the module
 */
backend::FunctionInfo UpdateMainWorkspaceSize(const IRModule& mod, tec::TargetMap targets,
                                              Map<Expr, backend::StorageInfo> storage_info_map);

/*! \brief Returns all the global \p PrimFunc functions in \p mod, but separated into an \p IRModule
 * per \p Target.
 *
 * \param mod The IRModule to extract the per target module from
 * \return The map from Target to IRModule
 */
Map<Target, IRModule> GetPerTargetModules(IRModule mod);

/*! \brief Lower an IRModule's primitive functions to TIR.
 *
 * This is the "back half" of the Relay compiler which lowers "primitive functions"
 * to TE expressions, schedules them, and then to TIR.
 *
 * \param module The IRModule.
 * \param memory_plan The memory plan used during lowering
 * \param module_name The name of this module
 * \param process_fn Callback allowing one-level up code generators to process
 * each function that we lower
 * \return The lowered module, see above.
 */
IRModule LowerTE(
    const IRModule& module, backend::StaticMemoryPlan memory_plan, const String& module_name,
    ProcessFn process_fn = [](BaseFunc f) {});

/*! \brief Pass to lower an IRModule's primitive functions to TIR.
 *
 * This is the "back half" of the Relay compiler which lowers "primitive functions"
 * to TE expressions, schedules them, and then to TIR. It annotates all functions
 * with their target.
 *
 * \param module_name The name of this module
 * \param process_fn Callback allowing one-level up code generators to process
 * each function that we lower
 * \param host_virtual_device \p VirtualDevice for host data and computations
 * \returns The pass which lowers primative functions to TIR
 */
transform::Pass LowerTEPass(const String& module_name, ProcessFn process_fn,
                            VirtualDevice host_virtual_device);
}  // namespace tec
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_TE_COMPILER_H_
