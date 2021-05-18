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
 * This file introduces a pass instrument infrastructure, inspired by LLVM and MLIR.
 * It inserts instrumentation points around passes.
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
 * It provides API for users to register callbacks at different instrumentation points.
 *
 * Within a pass context (tvm::transfom::PassContext), the instrumentation call sequence will like:
 *
 *   Instrument SetUp
 *
 *     if (Instrument Before Pass1())
 *       Pass1()
 *       Instrument After Pass1()
 *
 *     if (Instrument Before Pass2())
 *       Pass2()
 *       Instrument After Pass2()
 *
 *   Instrument TearDown
 *
 * The `Before Pass` instrumentation point can selectively disable passes by returning true (to
 * enable) or false (to disable).
 *
 * \sa PassInstrument
 */
class PassInstrumentNode : public Object {
 public:
  virtual ~PassInstrumentNode() {}

  /*! \brief Set up environment for instrumentation. */
  virtual void SetUp() const = 0;

  /*! \brief Clean up instrumentation environment. */
  virtual void TearDown() const = 0;

  /*!
   * \brief Instrument before pass run, determine whether to run the pass or not.
   * \param mod The module that an optimization pass runs on.
   * \param info The pass information.
   *
   * \return true to run the pass; false to skip the pass.
   */
  virtual bool RunBeforePass(const IRModule& mod, const transform::PassInfo& info) const = 0;

  /*!
   * \brief Instrument after pass run.
   *
   * \param mod The module that an optimization pass runs on.
   * \param info The pass information.
   */
  virtual void RunAfterPass(const IRModule& mod, const transform::PassInfo& info) const = 0;

  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "instrument.PassInstrument";
  TVM_DECLARE_BASE_OBJECT_INFO(PassInstrumentNode, Object);
};

/*!
 * \brief Managed reference class for PassInstrumentNode
 * \sa PassInstrumentNode
 */
class PassInstrument : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(PassInstrument, ObjectRef, PassInstrumentNode);
};

}  // namespace instrument
}  // namespace tvm

#endif  // TVM_IR_INSTRUMENT_H_
