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
#include "utils.h"

namespace tvm {
namespace relay {
namespace tec {

// This class is needed to avoid a GCC 5 bug that prevents maps containing enums
// from being compiled. If i386 GCC version is increased, we can remove it.
struct EnumClassHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

// TODO(@jroesch, @chrisS) these should be a tvm::Map for uniformity sake
// we should a version of context which works in Map
using TargetMap = std::unordered_map<DLDeviceType, Target, EnumClassHash>;
using DeviceMap =
    std::unordered_map<Expr, tvm::Device, runtime::ObjectPtrHash, runtime::ObjectPtrEqual>;
using ProcessFn = std::function<void(Function)>;

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

  /* Return all functions which have been lowered by the compiler, keyed by target. */
  virtual Map<String, IRModule> GetLoweredFunctions() = 0;

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
   * \return The runtime moduels for each needed external codegen tool.
   */
  virtual tvm::Array<tvm::runtime::Module> LowerExternalFunctions() = 0;

  virtual std::unordered_map<std::string, int> GetOpWeights() = 0;

  /*! \brief clear the cache. */
  virtual void Clear() = 0;

  void VisitAttrs(AttrVisitor*) {}

  static constexpr const char* _type_key = "relay.TECompiler";
  TVM_DECLARE_FINAL_OBJECT_INFO(TECompilerNode, Object);
};

/*! \brief cache entry used in compile engine */
class TECompiler : public ObjectRef {
 public:
  TECompiler();
  explicit TECompiler(ObjectPtr<Object> n) : ObjectRef(n) {}
  TECompilerNode* operator->() { return static_cast<TECompilerNode*>(get_mutable()); }
  using ContainerType = TECompilerNode;
};

/*! \brief The result of lowering a module, for now we need to pass an aggregate data structure
 * which contains more then a single module in order to interact with the today API.
 */
struct LoweredModule {
  /*! \brief The module which contains the Relay code. */
  IRModule main_module;
  /*! \brief The module which contains per target code. */
  Map<String, IRModule> per_target_module;
  /*! \brief The external runtime modules which must be combined with the lowered code. */
  Array<tvm::runtime::Module> external_mods;
  // TODO(@electriclilies): THis might need to become a map
  /*! \brief The info for this function (not sure what a better description is??)
   *
   */
  backend::FunctionInfo main_func_info;
};

/*!
 * \brief A function to create the function metadata for an input function (ie calculate buffer
 * input/output sizes)
 * \param relay_func The function to calculate function metadata for
 * \param function_metadata The map that stores all the function metadatas
 */
void UpdateFunctionMetadata(Function relay_func,
                            Map<String, backend::FunctionInfo>& function_metadata);  // NOLINT(*)

/*!
 * \brief Obtain the Target from the device type.
 * If homogenous compilation, this will return the only target.
 * If heteregenous compilation, this will select associated using the targets_ Map.
 *
 * \param dev_type
 * \return Target
 */
Target GetTargetFromInteger(DLDeviceType dev_type, TargetMap targets);

/*! \brief Lower an IRModule's primitive functions to TIR.
 *
 * This is the "back half" of the Relay compiler which lowers "primitive functions"
 * to TE expressions, schedules them, and then to TIR.
 *
 * \param compiler The TE-to-TIR compliler (which caches lowered functions)
 * \param module The IRModule.
 * \param targets The mapping for devices to targets.
 * \param device_map An analysis result mapping each sub-expression to a device.
 * \return The lowered module, see above.
 */
// TODO(@electriclilies): Not sure if this default initialization is correct...
LoweredModule LowerTE(
    const IRModule& module, TargetMap targets, DeviceMap device_map,
    backend::StaticMemoryPlan memory_plan, const String& module_name,
    ProcessFn process_fn = [](Function f) {});

}  // namespace tec
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_TE_COMPILER_H_
