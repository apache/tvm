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
 * \file tirx/analysis/verify_well_formed.cc
 * \brief Check if TIRx is well-formed.
 */

#include "verify_well_formed.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <exception>
#include <optional>
#include <tuple>
#include <variant>

#include "../ir/tir_visitor_with_path.h"
#include "tvm/ir/module.h"

namespace tvm {
namespace tirx {

bool VerifyWellFormed(const Function& func, bool assert_mode) {
  return VerifyWellFormedCommon<TIRVisitorWithPath>(func, assert_mode);
}

bool VerifyWellFormed(const IRModule& mod, bool assert_mode) {
  return VerifyWellFormedCommon<TIRVisitorWithPath>(mod, assert_mode);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.analysis.VerifyWellFormed", [](const ffi::ObjectRef& obj, bool assert_mode) {
        if (auto opt = obj.as<Function>()) {
          return VerifyWellFormed(opt.value(), assert_mode);
        } else if (auto opt = obj.as<IRModule>()) {
          return VerifyWellFormed(opt.value(), assert_mode);
        } else {
          TVM_FFI_THROW(InternalError)
              << "Expected VerifyWellFormed argument to be a Function or IRModule, but found "
              << obj->GetTypeKey();
        }
      });
}

}  // namespace tirx
}  // namespace tvm
