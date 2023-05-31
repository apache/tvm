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
 * \file onnx_module.cc
 * \brief ONNX Module without runtime support
 */
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace codegen {
using namespace tvm::runtime;

class ONNXSourceModuleNode : public runtime::ModuleNode {
 public:
  explicit ONNXSourceModuleNode(const std::string& code, const std::string& symbol,
                                const Array<String>& const_vars)
      : code_(code), symbol_(symbol), const_vars_(const_vars) {}
  const char* type_key() const { return "onnx"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return ModulePropertyMask::kRunnable; };

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_symbol") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->symbol_; });
    } else if (name == "get_const_vars") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->const_vars_; });
    } else {
      LOG(FATAL) << "ONNX Source module cannot execute, to get executable module"
                 << " build TVM with 'onnx' runtime support";
      return PackedFunc(nullptr);
    }
  }

  String GetSource(const String& format) final { return code_; }

  void SaveToFile(const String& path, const String& format) final {
    ICHECK_EQ(format, "onnx") << "Can only save to onnx format";
    ICHECK_NE(code_.length(), 0);
    const PackedFunc* to_onnx_ = runtime::Registry::Get("relay.ext.onnx.save_to_file");
    (*to_onnx_)(code_, path, format);
  }

 protected:
  String code_;
  std::string symbol_;
  Array<String> const_vars_;
};

Module ONNXSourceModuleNodeCreate(const String& code, const String& symbol,
                                  const Array<String>& const_vars) {
  auto n = make_object<ONNXSourceModuleNode>(code.operator std::string(),
                                             symbol.operator std::string(), const_vars);
  return Module(n);
}

TVM_REGISTER_GLOBAL("runtime.ONNXModuleCreate").set_body_typed(ONNXSourceModuleNodeCreate);

}  // namespace codegen
}  // namespace tvm
