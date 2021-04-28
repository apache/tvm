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
 * \file tvm/ir/instrument.h
 *
 * This file implements a pass instrument infrastructure, inspired from LLVM and MLIR.
 * It inserts instrumentation points between passes run.
 *
 * Within a pass context (tvm::transfom::PassContext), the instrumentation call sequence will like:
 *
 *   Instrument SetUp
 *
 *     if (Instrument Before Pass)
 *       Pass Run
 *       Instrument After Pass
 *
 *     if (Instrument Before Pass)
 *       Pass Run
 *       Instrument After Pass
 *
 *   Instrument TearDown
 *
 *
 * Instrument point before pass can determine particular pass is disable or not depends on the
 * callback registered.
 */
#ifndef TVM_IR_INSTRUMENT_H_
#define TVM_IR_INSTRUMENT_H_

#include <tvm/node/reflection.h>
#include <tvm/runtime/container.h>

#include <utility>
#include <vector>

namespace tvm {

class IRModule;

// Forward class for PassInstrumentNode methods
namespace transform {
class PassInfo;
}  // namespace transform

namespace instrument {

/*!
 * \brief A callback type for set up or clean up instrument environment.
 */
using InstrumentEnvFunc = runtime::TypedPackedFunc<void()>;

/*!
 * \brief A callback template for instrumenting before/after environment.
 * \tparam RetTy the return type of callback.
 */
template <typename RetTy = void>
using PassInstrumentFunc =
    runtime::TypedPackedFunc<RetTy(const IRModule&, const transform::PassInfo&)>;

/*!
 * \brief PassInstrumentNode forms an instrument implementation.
 * It provides API for users to register callbacks at different instrument point.
 * \sa PassInstrument
 */
class PassInstrumentNode : public Object {
 public:
  /*! \brief Name of this pass instrument object. */
  String name;

  /*! \brief Callback for instrumentation environment set up. */
  InstrumentEnvFunc set_up_callback;
  /*! \brief Callback for instrumentation environment clean up. */
  InstrumentEnvFunc tear_down_callback;

  /*! \brief Callback to run before a pass. */
  PassInstrumentFunc</* RetTy */ bool> run_before_pass_callback;
  /*! \brief Callback to run after a pass. */
  PassInstrumentFunc<> run_after_pass_callback;

  /*!
   * \brief Register a callback to run at set up point.
   *
   * \param callback The set up function.
   */
  void RegisterSetUpCallback(InstrumentEnvFunc callback) { set_up_callback = std::move(callback); }

  /*
   * \brief Register a callback to run at clean up point.
   *
   * \param callback The clean up function.
   */
  void RegisterTearDownCallback(InstrumentEnvFunc callback) {
    tear_down_callback = std::move(callback);
  }

  /*!
   * \brief Register a callback to run before pass run.
   *
   * \param callback The function to run before pass: return false to skip pass; return true to
   * run pass.
   */
  void RegisterRunBeforePassCallback(PassInstrumentFunc<bool> callback) {
    run_before_pass_callback = std::move(callback);
  }

  /*!
   * \brief Register a callback to run after pass run.
   *
   * \param callback The function to run after pass.
   */
  void RegisterRunAfterPassCallback(PassInstrumentFunc<> callback) {
    run_after_pass_callback = std::move(callback);
  }

  void VisitAttrs(AttrVisitor* v) { v->Visit("name", &name); }

  /*! \brief Set up environment for instrumentation. */
  void SetUp() const;

  /*! \brief Clean up instrumentation environment. */
  void TearDown() const;

  /*!
   * \brief Instrument before pass run, determine whether to run the pass or not.
   * \param mod The module that an optimization pass runs on.
   * \param info The pass information.
   *
   * \return true to run the pass; false to skip the pass.
   */
  bool RunBeforePass(const IRModule& mod, const transform::PassInfo& info) const;

  /*!
   * \brief Instrument after pass run.
   *
   * \param mod The module that an optimization pass runs on.
   * \param info The pass information.
   */
  void RunAfterPass(const IRModule& mod, const transform::PassInfo& info) const;

  static constexpr const char* _type_key = "instrument.PassInstrument";
  TVM_DECLARE_FINAL_OBJECT_INFO(PassInstrumentNode, Object);
};

/*!
 * \brief Managed reference class for PassInstrumentNode
 * \sa PassInstrumentNode
 */
class PassInstrument : public ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param name Name for this instrumentation.
   */
  TVM_DLL PassInstrument(String name);

  /*!
   * \brief mutable accessor.
   * \return mutable access pointer.
   */
  PassInstrumentNode* operator->() {
    ICHECK(get() != nullptr);
    return static_cast<PassInstrumentNode*>(get_mutable());
  }

  TVM_DEFINE_OBJECT_REF_METHODS(PassInstrument, ObjectRef, PassInstrumentNode);
};

/*!
 * \brief PassInstrumentorNode collects a set of PassInstrument implementations, invokes the
 * implementations' methods at different instrument points.
 * \sa PassInstrumentor
 */
class PassInstrumentorNode : public Object {
 public:
  Array<PassInstrument> pass_instruments;

  void VisitAttrs(AttrVisitor* v) { v->Visit("pass_instruments", &pass_instruments); }

  /*! \brief Set up environment for instrument implementations. */
  void SetUp() const;

  /*! \brief Clean up environment for instrument implementations. */
  void TearDown() const;

  /*!
   * \brief Instrument before pass run, determine whether to run the pass or not.
   *
   * \param mod The module that an optimization pass runs on.
   * \param info The pass information.
   *
   * \return true to run the pass; false to skip the pass.
   */
  bool RunBeforePass(const IRModule& mod, const transform::PassInfo& info) const;

  /*!
   * \brief Instrument after pass run.
   *
   * \param mod The module that an optimization pass runs on.
   * \param info The pass information.
   *
   * \return true to run the pass; false to skip the pass.
   */
  void RunAfterPass(const IRModule& mod, const transform::PassInfo& info) const;

  static constexpr const char* _type_key = "instrument.PassInstrumentor";
  TVM_DECLARE_FINAL_OBJECT_INFO(PassInstrumentorNode, Object);
};

/*!
 * \brief Managed reference class for PassInstrumentorNode
 * \sa PassInstrumentorNode
 */
class PassInstrumentor : public ObjectRef {
 public:
  /*!
   * \brief Constructor
   * \param pass_instruments A set of instrument implementations.
   */
  TVM_DLL PassInstrumentor(Array<PassInstrument> pass_instruments);

  TVM_DEFINE_OBJECT_REF_METHODS(PassInstrumentor, ObjectRef, PassInstrumentorNode);
};

}  // namespace instrument
}  // namespace tvm

#endif  // TVM_IR_INSTRUMENT_H_
