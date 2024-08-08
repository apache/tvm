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
 * \file tvm/ir/analysis.h
 *
 * Analysis routines that must function across multiple IR types for
 * correctness.  For example, identifying unused functions, when both TIR
 *
 */
#ifndef TVM_IR_ANALYSIS_H_
#define TVM_IR_ANALYSIS_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/node/functor.h>
#include <tvm/runtime/container/array.h>

namespace tvm {
namespace ir {

class CalleeCollector {
 public:
  /* \brief Functor to be registered for IR types
   *
   * Should be implemented for each `BaseFunc` subclass.
   * Implementation should call `CalleeCollector::Mark` for each
   * `GlobalVar` in the function.
   */
  using FType = NodeFunctor<void(const ObjectRef&, CalleeCollector*)>;
  TVM_DLL static FType& vtable() {
    static FType inst;
    return inst;
  }

  virtual ~CalleeCollector() {}

  /* \brief Collect the GlobalVar in a function */
  virtual void Mark(GlobalVar gvar) = 0;
};

Map<GlobalVar, Array<GlobalVar>> CollectCallMap(const IRModule& mod);

}  // namespace ir
}  // namespace tvm

#endif  // TVM_IR_ANALYSIS_H_
