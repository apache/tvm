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

#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/var.h>

#include <string>

#include "text_printer.h"

namespace tvm {
namespace relay {

class ModelLibraryFormatPrinter : public ::tvm::runtime::ModuleNode {
 public:
  ModelLibraryFormatPrinter(bool show_meta_data,
                            const runtime::TypedPackedFunc<std::string(ObjectRef)>& annotate,
                            bool show_warning)
      : text_printer_{show_meta_data, annotate, show_warning} {}

  const char* type_key() const final { return "model_library_format_printer"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return runtime::ModulePropertyMask::kRunnable; }

  std::string Print(const ObjectRef& node) {
    std::ostringstream oss;
    oss << node;
    return oss.str();
  }

  TVMRetValue GetVarName(tir::Var var) {
    TVMRetValue rv;
    std::string var_name;
    if (text_printer_.GetVarName(var, &var_name)) {
      rv = var_name;
    }

    return rv;
  }

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) override {
    if (name == "print") {
      return TypedPackedFunc<std::string(ObjectRef)>(
          [sptr_to_self, this](ObjectRef node) { return Print(node); });
    } else if (name == "get_var_name") {
      return TypedPackedFunc<TVMRetValue(tir::Var)>(
          [sptr_to_self, this](tir::Var var) { return GetVarName(var); });
    } else {
      return PackedFunc();
    }
  }

 private:
  TextPrinter text_printer_;
};

TVM_REGISTER_GLOBAL("relay.ir.ModelLibraryFormatPrinter")
    .set_body_typed([](bool show_meta_data,
                       const runtime::TypedPackedFunc<std::string(ObjectRef)>& annotate,
                       bool show_warning) {
      return ObjectRef(
          make_object<ModelLibraryFormatPrinter>(show_meta_data, annotate, show_warning));
    });

}  // namespace relay
}  // namespace tvm
