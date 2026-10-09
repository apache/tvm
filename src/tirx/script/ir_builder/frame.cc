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
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/script/ir_builder/ir.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/script/ir_builder/frame.h>

#include <map>

#include "./utils.h"

namespace tvm {
namespace script {

namespace ir_builder {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() {
  TIRFrameNode::RegisterReflection();
  FunctionFrameNode::RegisterReflection();
}

namespace {
std::map<ffi::String, FunctionFrameNode::AttrValidator>& AttrValidators() {
  static std::map<ffi::String, FunctionFrameNode::AttrValidator> validators;
  return validators;
}
}  // namespace

void FunctionFrameNode::RegisterAttrValidator(ffi::String key, AttrValidator validator) {
  TVM_FFI_ICHECK(AttrValidators().emplace(std::move(key), std::move(validator)).second)
      << "Duplicate function attribute validator";
}

void FunctionFrameNode::ValidateAttrs() const {
  for (const auto& [key, value] : attrs) {
    auto it = AttrValidators().find(key);
    if (it != AttrValidators().end()) it->second(this, value);
  }
}

void TIRFrameNode::BindBufferRegion(tvm::tirx::TensorVar buffer, tvm::TensorRegion region) {
  TVM_FFI_THROW(ValueError) << "match_buffer requires a frame that supports region aliases";
}

tvm::tirx::Function FunctionFrameNode::FinalizeFunction(tvm::tirx::Function func) { return func; }

void FunctionFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  ValidateAttrs();
  // if the function is not private and there isn't already a global symbol,
  // add a global symbol
  auto insert_attr = [&](ffi::String key, ffi::Any value) {
    if (!attrs.defined()) {
      attrs = {{key, value}};
    } else if (!attrs.count(key)) {
      // copy over attributes (can't mutate the dict inside the optional in-place)
      ffi::Map<ffi::String, ffi::Any> new_attrs;
      for (auto kv : attrs) {
        new_attrs.Set(kv.first, kv.second);
      }
      new_attrs.Set(key, value);
      attrs = std::move(new_attrs);
    }
  };
  // Default attributes belong to the completed function. A declaration must
  // leave body-level func_attr free to supply those values on the same frame.
  if (!is_declaration && !is_private && name.has_value() &&
      !attrs.count(tvm::attr::kGlobalSymbol)) {
    insert_attr(tvm::attr::kGlobalSymbol, name.value());
  }
  if (!is_declaration && persistent) {
    insert_attr(tvm::tirx::attr::kPersistentKernel, true);
  }
  TVM_FFI_CHECK(!is_declaration || stmts.empty(), ValueError)
      << "A function declaration cannot contain body statements";
  ffi::Optional<tvm::SeqStmt> body = std::nullopt;
  if (!is_declaration) body = AsStmt(stmts);
  tvm::tirx::Function func(
      /*params=*/args,
      /*body=*/body,
      /*ret_type=*/ret_type.value_or(TupleType::Empty()),
      /*attrs=*/attrs.defined() ? DictAttrs(attrs) : DictAttrs(),
      /*span=*/source_span);
  func = FinalizeFunction(std::move(func));
  function = func;
  IRBuilder builder = IRBuilder::Current();
  if (builder->frames.empty()) {
    TVM_FFI_CHECK(!builder->result.has_value(), ValueError)
        << "Builder.result has already been set";
    if (!is_declaration) builder->result = func;
  } else if (ffi::Optional<ir::IRModuleFrame> opt_frame = builder->FindFrame<ir::IRModuleFrame>()) {
    TVM_FFI_CHECK(name.has_value(), ValueError)
        << "The function name must be defined before exiting the "
           "function scope, if it's defined in a Module";
    const ir::IRModuleFrame& frame = opt_frame.value();
    const ffi::String& func_name = name.value_or("");
    if (!frame->global_var_map.count(func_name) ||
        !frame->functions.count(frame->global_var_map.at(func_name))) {
      // Case. First time visiting the function.
      global_var = ir::DeclFunction(func_name, func);
    }
    // Define the function.
    // Note we do checks to disallow redefinition of functions inside the `DefFunction`.
    if (!global_var.has_value()) {
      TVM_FFI_CHECK(!is_declaration, ValueError) << "function " << func_name << " already exists";
      global_var = frame->global_var_map.at(func_name);
    }
    if (!is_declaration) {
      ir::DefFunction(func_name, func);
    }
  } else {
    TVM_FFI_THROW(ValueError) << "Cannot find where to insert Function";
  }
  is_declaration = false;
}

}  // namespace tirx
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
