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
 * This file introduces a pass instrument infrastructure, inspired from LLVM and MLIR.
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
 * \brief PassInstrumentNode forms an instrument implementation.
 * It provides API for users to register callbacks at different instrument point.
 * \sa PassInstrument
 */
class PassInstrumentNode : public Object {
 public:
  /*! \brief Name of this pass instrument object. */
  String name;

  /*! \brief Callback for instrumentation environment set up. */
  runtime::TypedPackedFunc<void()> set_up_callback;
  /*! \brief Callback for instrumentation environment clean up. */
  runtime::TypedPackedFunc<void()> tear_down_callback;

  /*! \brief Callback to run before a pass. */
  runtime::TypedPackedFunc<bool(const IRModule&, const transform::PassInfo&)>
      run_before_pass_callback;
  /*! \brief Callback to run after a pass. */
  runtime::TypedPackedFunc<void(const IRModule&, const transform::PassInfo&)>
      run_after_pass_callback;

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
  TVM_DECLARE_BASE_OBJECT_INFO(PassInstrumentNode, Object);
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

}  // namespace instrument
}  // namespace tvm

#endif  // TVM_IR_INSTRUMENT_H_
