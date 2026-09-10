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
 * \file var_touch.cc
 * \brief Implementation of simple passes
 */
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/analysis.h>

#include <utility>

namespace tvm {
namespace tirx {

namespace {

template <typename T>
bool UsesVarImpl(const T& value, std::function<bool(const VarNode*)> var_set) {
  bool use_var = false;
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
      value, [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
        if (var_set(var.get())) {
          use_var = true;
          return ffi::WalkResult::Interrupt();
        }
        return ffi::WalkResult::Advance();
      });
  return use_var;
}

}  // namespace

bool UsesVar(const Stmt& stmt, std::function<bool(const VarNode*)> var_set) {
  return UsesVarImpl(stmt, std::move(var_set));
}

bool UsesVar(const PrimExpr& expr, std::function<bool(const VarNode*)> var_set) {
  return UsesVarImpl(expr, std::move(var_set));
}

}  // namespace tirx
}  // namespace tvm
