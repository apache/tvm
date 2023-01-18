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
#include <tvm/ir/module.h>
#include <tvm/relay/base.h>
#include <tvm/relay/error.h>

// clang-format off
#include <string>
#include <vector>
#include <rang.hpp>
// clang-format on

namespace tvm {
namespace relay {

template <typename T, typename U>
using NodeMap = std::unordered_map<T, U, ObjectPtrHash, ObjectPtrEqual>;

void ErrorReporter::RenderErrors(const IRModule& module, bool use_color) {
  // First we pick an error reporting strategy for each error.
  // TODO(@jroesch): Spanned errors are currently not supported.
  for (auto err : this->errors_) {
    ICHECK(!err.span.defined()) << "attempting to use spanned errors, currently not supported";
  }

  NodeMap<GlobalVar, NodeMap<ObjectRef, std::string>> error_maps;

  // Set control mode in order to produce colors;
  if (use_color) {
    rang::setControlMode(rang::control::Force);
  }

  for (auto pair : this->node_to_gv_) {
    auto node = pair.first;
    auto global = Downcast<GlobalVar>(pair.second);

    auto has_errs = this->node_to_error_.find(node);

    ICHECK(has_errs != this->node_to_error_.end());

    const auto& error_indices = has_errs->second;

    std::stringstream err_msg;

    err_msg << rang::fg::red;
    err_msg << " ";
    for (auto index : error_indices) {
      err_msg << this->errors_[index].what() << "; ";
    }
    err_msg << rang::fg::reset;

    // Setup error map.
    auto it = error_maps.find(global);
    if (it != error_maps.end()) {
      it->second.insert({node, err_msg.str()});
    } else {
      error_maps.insert({global, {{node, err_msg.str()}}});
    }
  }

  // Now we will construct the fully-annotated program to display to
  // the user.
  std::stringstream annotated_prog;

  // First we output a header for the errors.
  annotated_prog << rang::style::bold << std::endl
                 << "Error(s) have occurred. The program has been annotated with them:" << std::endl
                 << std::endl
                 << rang::style::reset;

  // For each global function which contains errors, we will
  // construct an annotated function.
  for (auto pair : error_maps) {
    auto global = pair.first;
    auto err_map = pair.second;
    auto func = module->Lookup(global);

    // We output the name of the function before displaying
    // the annotated program.
    annotated_prog << rang::style::bold << "In `" << global->name_hint << "`: " << std::endl
                   << rang::style::reset;

    // We then call into the Relay printer to generate the program.
    //
    // The annotation callback will annotate the error messages
    // contained in the map.
    annotated_prog << AsText(func, false, [&err_map](const ObjectRef& expr) {
      auto it = err_map.find(expr);
      if (it != err_map.end()) {
        ICHECK_NE(it->second.size(), 0);
        return it->second;
      } else {
        return std::string("");
      }
    });
  }

  auto msg = annotated_prog.str();

  if (use_color) {
    rang::setControlMode(rang::control::Auto);
  }

  // Finally we report the error, currently we do so to LOG(FATAL),
  // it may be good to instead report it to std::cout.
  LOG(FATAL) << annotated_prog.str() << std::endl;
}

void ErrorReporter::ReportAt(const GlobalVar& global, const ObjectRef& node,
                             const CompileError& err) {
  size_t index_to_insert = this->errors_.size();
  this->errors_.push_back(err);
  auto it = this->node_to_error_.find(node);
  if (it != this->node_to_error_.end()) {
    it->second.push_back(index_to_insert);
  } else {
    this->node_to_error_.insert({node, {index_to_insert}});
  }
  this->node_to_gv_.insert({node, global});
}
}  // namespace relay
}  // namespace tvm
