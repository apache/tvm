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
 * \file relay/backend/compile_engine.h
 * \brief Internal compilation layer which lowers Relay "primitive functions" to TIR PrimFns.
 *
 * This layer represents the older design of the Relay compilation flow and is being deprecated
 * in favor of te_compiler.h which is a migration step towards a standard pass based lowering of
 * Relay functions.
 *
 */
#ifndef TVM_RELAY_BACKEND_COMPILE_ENGINE_H_
#define TVM_RELAY_BACKEND_COMPILE_ENGINE_H_

#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op_strategy.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/module.h>

#include <functional>
#include <string>

#include "te_compiler_cache.h"

namespace tvm {
namespace relay {

using namespace tvm::relay::tec;

/*!
 * \brief Backend compilation engine for
 *        low level code generation.
 */
class CompileEngineNode : public Object {
 public:
  /*! \brief destructor */
  virtual ~CompileEngineNode() {}
  /*!
   * \brief Get lowered result.
   * \param key The key to the cached function.
   * \param mod_name The mangling function for mangling names.
   * \return The result.
   */
  virtual CachedFunc Lower(const CCacheKey& key, std::function<String(String)> mangle_fn) = 0;

  /*!
   * \brief Get lowered result.
   * \param key The key to the cached function.
   * \param mod_name The module name to mangle the functions.
   * \return The result.
   */
  virtual CachedFunc Lower(const CCacheKey& key, const String mangle_fn) = 0;
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

  /*! \brief clear the cache. */
  virtual void Clear() = 0;

  // VisitAttrs
  void VisitAttrs(AttrVisitor*) {}

  static constexpr const char* _type_key = "relay.CompileEngine";
  TVM_DECLARE_FINAL_OBJECT_INFO(CompileEngineNode, Object);
};

/*! \brief cache entry used in compile engine */
class CompileEngine : public ObjectRef {
 public:
  CompileEngine() {}
  explicit CompileEngine(ObjectPtr<Object> n) : ObjectRef(n) {}
  CompileEngineNode* operator->() { return static_cast<CompileEngineNode*>(get_mutable()); }
  using ContainerType = CompileEngineNode;
  /*! \brief The global compile engine. */
  TVM_DLL static CompileEngine& Global();
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_COMPILE_ENGINE_H_
