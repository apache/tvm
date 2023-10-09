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
#ifndef TVM_TIR_SCHEDULE_ERROR_H_
#define TVM_TIR_SCHEDULE_ERROR_H_
#include <tvm/tir/schedule/state.h>

#include <string>
#include <utility>

namespace tvm {
namespace tir {

/*! \brief Error that happens during TensorIR scheduling */
class ScheduleError : public tvm::runtime::Error {
 public:
  /*! \brief Base constructor */
  ScheduleError() : tvm::runtime::Error("") {}
  /*! \brief The error occurred in this IRModule */
  virtual IRModule mod() const = 0;
  /*! \brief The locations of interest that we want to point out */
  virtual Array<ObjectRef> LocationsOfInterest() const = 0;
  /*!
   * \brief Returns an error string template for rendering, corresponds to the "detail" mode.
   * \sa ScheduleErrorRenderLevel
   * \note The template is a string, e.g.
   * "Some error occurred on block {0} and loop {1} blah blah"
   * And renderer will replace {0} and {1} according to the list provided LocationsOfInterest. Right
   * now it only printed out all the locations in plain text, but in the future, we may want to mark
   * the IR with underscores and attach names to each location of interest.
   */
  virtual String DetailRenderTemplate() const = 0;
  /*!
   * \brief Returns an error string without needing to render, corresponds to the "fast" mode
   * \sa ScheduleErrorRenderLevel
   */
  virtual String FastErrorString() const = 0;
  /*! \brief Render the ScheduleError with the template provided by `DetailRenderTemplate` */
  String RenderReport(const String& primitive) const;
};

class LoopPositionError : public ScheduleError {
 public:
  explicit LoopPositionError(IRModule mod, For loop, Block block, const std::string& primitive)
      : mod_(std::move(mod)),
        loop_(std::move(loop)),
        block_(std::move(block)),
        primitive_(primitive) {}

  String FastErrorString() const final {
    return "ScheduleError: " + primitive_ + " expect the loop to be an ancestor of block";
  }

  String DetailRenderTemplate() const final {
    std::ostringstream os;
    os << "ScheduleError: The input loop {0} of " << primitive_
       << " is required to be an ancestor of block {1}.";
    return os.str();
  }

  IRModule mod() const final { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const final { return {loop_, block_}; }

  IRModule mod_;
  For loop_;
  Block block_;
  std::string primitive_;
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_ERROR_H_
