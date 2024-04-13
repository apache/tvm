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
 * \file src/ir/apply_pass_to_function.cc
 * \brief Utility transformation that applies an inner pass to a subset of an IRModule
 */
#include <tvm/ir/transform.h>
#include <tvm/relax/expr.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/function.h>

#include <unordered_set>

#include "../runtime/regex.h"

namespace tvm {
namespace transform {

namespace {
BaseFunc BaseFuncWithAttr(BaseFunc func, const std::string& attr_key, ObjectRef attr_value) {
  if (auto tir = func.as<tir::PrimFunc>()) {
    return WithAttr(tir.value(), attr_key, attr_value);
  } else if (auto relax = func.as<relax::Function>()) {
    return WithAttr(relax.value(), attr_key, attr_value);
  } else {
    return func;
  }
}

BaseFunc BaseFuncWithoutAttr(BaseFunc func, const std::string& attr_key) {
  if (auto tir = func.as<tir::PrimFunc>()) {
    return WithoutAttr(tir.value(), attr_key);
  } else if (auto relax = func.as<relax::Function>()) {
    return WithoutAttr(relax.value(), attr_key);
  } else {
    return func;
  }
}
}  // namespace

Pass ApplyPassToFunction(Pass pass, String func_name_regex,
                         bool error_if_no_function_matches_regex) {
  auto pass_name =
      static_cast<const std::stringstream&>(std::stringstream() << "ApplyPassTo" << func_name_regex)
          .str();

  auto pass_func = [pass, func_name_regex, error_if_no_function_matches_regex](
                       IRModule mod, PassContext) -> IRModule {
    bool at_least_one_function_matched_regex = false;
    std::unordered_set<String> keep_original_version;
    std::unordered_set<String> internal_functions;
    IRModule subset;

    for (auto [gvar, func] : mod->functions) {
      std::string name = gvar->name_hint;
      if (tvm::runtime::regex_match(name, func_name_regex)) {
        at_least_one_function_matched_regex = true;
        if (!func->GetAttr<String>(tvm::attr::kGlobalSymbol).defined()) {
          // Function may be mutated, but is an internal function.  Mark
          // it as externally-exposed, so that any call-tracing internal
          // transforms do not remove this function, in case it its
          // callers are not being mutated.

          internal_functions.insert(gvar->name_hint);
          func = BaseFuncWithAttr(func, tvm::attr::kGlobalSymbol, gvar->name_hint);
        }
      } else {
        // Function may not be mutated.  Replace it with a
        // `relax::ExternFunc` to prevent references to it from
        // dangling.
        keep_original_version.insert(gvar->name_hint);
        func = relax::ExternFunc("dummy_" + name);
        func->struct_info_ = gvar->struct_info_;
        func->checked_type_ = gvar->checked_type_;
      }

      subset->Add(gvar, func);
    }

    if (error_if_no_function_matches_regex) {
      CHECK(at_least_one_function_matched_regex)
          << "No function matched regex '" << func_name_regex << "', out of functions " << [&]() {
               Array<String> function_names;
               for (const auto& [gvar, func] : mod->functions) {
                 function_names.push_back(gvar->name_hint);
               }
               return function_names;
             }();
    }

    IRModule new_subset = pass(subset);
    if (new_subset.same_as(subset)) {
      return mod;
    }

    auto write_ptr = mod.CopyOnWrite();
    for (auto [gvar, func] : new_subset->functions) {
      if (!keep_original_version.count(gvar->name_hint)) {
        if (auto it = write_ptr->global_var_map_.find(gvar->name_hint);
            it != write_ptr->global_var_map_.end()) {
          write_ptr->Remove((*it).second);
        }
        if (internal_functions.count(gvar->name_hint)) {
          func = BaseFuncWithoutAttr(func, tvm::attr::kGlobalSymbol);
        }
        write_ptr->Add(gvar, func);
      }
    }

    return mod;
  };

  return CreateModulePass(pass_func, 0, pass_name, {});
}

TVM_REGISTER_GLOBAL("transform.ApplyPassToFunction").set_body_typed(ApplyPassToFunction);

}  // namespace transform
}  // namespace tvm
