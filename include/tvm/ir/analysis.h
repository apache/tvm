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
 * \brief Structural analysis of shared IR objects.
 */
#ifndef TVM_IR_ANALYSIS_H_
#define TVM_IR_ANALYSIS_H_

#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/expr.h>

#include <unordered_set>

namespace tvm {

/*!
 * \brief Find variable uses without a preceding definition in well-formed IR.
 *
 * Definitions follow the structural definition-site metadata. Variable types
 * are traversed at definitions, while variable uses are leaves. Scope and
 * duplicate-definition validation belong to the IR verifier.
 *
 * \param object The IR object to visit.
 * \param defs Variables already defined outside the object.
 * \return Undefined variables, uniquely ordered by their first use.
 */
inline ffi::Array<Var> UndefinedVars(ffi::AnyView object, const ffi::Array<Var>& defs = {}) {
  std::unordered_set<const VarNode*> defined;
  std::unordered_set<const VarNode*> used;
  for (const Var& var : defs) defined.insert(var.get());
  ffi::Array<Var> undefined;
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
      object, [&](const Var& var, TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::WalkResult> {
        if (kind != kTVMFFIDefRegionKindNone) {
          defined.insert(var.get());
          return ffi::WalkResult::Advance();
        }
        if (!defined.count(var.get()) && used.insert(var.get()).second) {
          undefined.push_back(var);
        }
        return ffi::WalkResult::Skip();
      });
  return undefined;
}

}  // namespace tvm
#endif  // TVM_IR_ANALYSIS_H_
