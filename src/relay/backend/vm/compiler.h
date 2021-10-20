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
 * \file src/relay/backend/vm/compiler.h
 * \brief A compiler from relay::Module to the VM byte code.
 */

#ifndef TVM_RELAY_BACKEND_VM_COMPILER_H_
#define TVM_RELAY_BACKEND_VM_COMPILER_H_

#include <tvm/ir/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/tir/function.h>

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../../runtime/vm/naive_allocator.h"
#include "../../../runtime/vm/profiler/vm.h"
#include "../../transforms/pass_utils.h"
#include "../te_compiler.h"
#include "../te_compiler_cache.h"

namespace tvm {
namespace relay {
namespace vm {

using namespace tvm::runtime;
using namespace tvm::runtime::vm;
using namespace relay::transform;

template <typename T, typename U>
using NodeMap = std::unordered_map<T, U, ObjectPtrHash, ObjectPtrEqual>;
using TagMap = NodeMap<tvm::relay::Constructor, Index>;
using TagNameMap = std::unordered_map<size_t, tvm::relay::Constructor>;
using GlobalMap = NodeMap<GlobalVar, Index>;
using ConstMap = NodeMap<Constant, Index>;
using ConstTensorShapeMap = NodeMap<TensorType, std::pair<Index, NDArray>>;
using TargetsMap = Map<tvm::Integer, tvm::Target>;

struct VMCompilerContext {
  // The module context for the compilation
  IRModule module;
  // Error reporter
  ErrorReporter err_reporter;
  // Map from a unique integer to ADT constructor tag
  TagNameMap tag_index_map;
  // Map from ADT constructor tag to a unique integer
  TagMap tag_map;
  // Map from global var to a unique integer
  GlobalMap global_map;
  // TEcompiler for lowering
  tec::TECompiler compiler;
  // List of constants
  std::vector<NDArray> constants;
  // Device type for constants
  std::vector<Index> const_device_type;
  // List of cached functions
  std::vector<tec::CachedFunc> cached_funcs;
  // The functions that have been lowered.
  std::unordered_map<tir::PrimFunc, size_t, ObjectPtrHash, ObjectPtrEqual> seen_funcs;
};

class VMCompiler : public runtime::ModuleNode {
 public:
  virtual ~VMCompiler() {}

  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

  const char* type_key() const { return "VMCompiler"; }

  /*!
   * \brief Set the parameters
   *
   * \param name name of parameter
   * \param data_in input DLTensor
   */
  void SetParam(const std::string& name, runtime::NDArray data_in);

  /*!
   * \brief Lower the functions in a Module
   *
   * \param mod Relay Module
   * \param targets For heterogeneous compilation, it is a dictionary indicating device type
   *                to target mapping. For homogeneous compilation, it is a singleton build target.
   * \param target_host Host compilation target, if target is device.
   */
  void Lower(IRModule mod, const TargetsMap& targets, const tvm::Target& target_host);

  /*! \brief Generate the machine code for lowered functions. */
  void Codegen();

 protected:
  /*
   * \brief Perform a series of optimizations on the input IR module.
   *
   * \param mod The input IRModule.
   * \param targets For heterogeneous compilation, it is a dictionary indicating device type
   *                to target mapping. For homogeneous compilation, it is a singleton build target.
   * \param target_host Host compilation target.
   *
   * \return The optimized IRModule.
   */
  IRModule OptimizeModule(IRModule mod, const TargetsMap& targets, const Target& target_host);

  /*!
   * \brief Populate the global function names in a map where the value is used
   *        as the index by the VMFunctions.
   */
  void PopulateGlobalMap();

 protected:
  /*! \brief Target devices. */
  TargetsMap targets_;
  /*! \brief Target host device. */
  tvm::Target target_host_;
  /*! \brief Global shared meta data */
  VMCompilerContext context_;
  /*! \brief Compiled executable. */
  ObjectPtr<Executable> exec_;
  /*! \brief parameters */
  std::unordered_map<std::string, runtime::NDArray> params_;
};

}  // namespace vm
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_VM_COMPILER_H_
