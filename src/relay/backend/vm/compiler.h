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
#include <tvm/support/logging.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/vm.h>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "../../../runtime/vm/profiler/vm.h"
#include "../../../runtime/vm/naive_allocator.h"
#include "../../backend/compile_engine.h"
#include "../../transforms/pass_util.h"

namespace tvm {
namespace relay {
namespace vm {

using namespace tvm::runtime;
using namespace tvm::runtime::vm;
using namespace relay::transform;

template <typename T, typename U>
using NodeMap = std::unordered_map<T, U, ObjectHash, ObjectEqual>;
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
  // List of constants
  std::vector<NDArray> constants;
  // List of cached functions
  std::vector<CachedFunc> cached_funcs;
  // The functions that have been lowered.
  std::unordered_map<tir::PrimFunc, size_t, ObjectHash, ObjectEqual> seen_funcs;
};


class VMCompiler : public runtime::ModuleNode {
 public:
  virtual ~VMCompiler() {}

  virtual PackedFunc GetFunction(const std::string& name,
                                 const ObjectPtr<Object>& sptr_to_self);

  const char* type_key() const {
    return "VMCompiler";
  }

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
   * \param targets For heterogeneous compilation, it is a dictionary indicating context
                    to target mapping. For homogeneous compilation, it is a build target.
   * \param target_host Host compilation target, if target is device.
   */
  void Lower(IRModule mod,
             const TargetsMap& targets,
             const tvm::Target& target_host);

  /*! \brief Generate the machine code for lowered functions. */
  void Codegen();

 protected:
  IRModule OptimizeModule(const IRModule& mod, const TargetsMap& targets);

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
