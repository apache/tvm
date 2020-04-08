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
 * \file micro_module.cc
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module.h>
#include <unordered_map>
#include <string>
#include "micro_session.h"
#include "low_level_device.h"
#include "micro_common.h"
#include "../pack_args.h"

namespace tvm {
namespace runtime {
/*!
 * \brief module for uTVM micro devices
 */
class MicroModuleNode final : public ModuleNode {
 public:
  MicroModuleNode() {}

  ~MicroModuleNode() {}

  const char* type_key() const final {
    return "micro";
  }

  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final;

  /*!
   * \brief initializes module by establishing device connection and loads binary
   * \param binary_path path of the binary to be loaded
   */
  void InitMicroModule(const std::string& binary_path) {
    // std::cout << "[MicroModuleNode::InitMicroModule]" << std::endl;
    // std::cout << "  start" << std::endl;
    session_ = MicroSession::Current();
    symbol_map_ = session_->LoadBinary(binary_path, true).symbol_map;
  }

 private:
  SymbolMap symbol_map_;
  /*! \brief global session pointer */
  ObjectPtr<MicroSession> session_;
};

class MicroWrappedFunc {
 public:
  MicroWrappedFunc(ObjectPtr<MicroSession> session,
                   TargetPtr func_ptr) {
    session_ = session;
    func_ptr_ = func_ptr;
  }

  void operator()(TVMArgs args, TVMRetValue* rv) const {
    session_->PushToTaskQueue(func_ptr_, args);
  }

 private:
  /*! \brief reference to the session for this function (to keep the session alive) */
  ObjectPtr<MicroSession> session_;
  /*! \brief offset of the function to be called */
  TargetPtr func_ptr_;
};

PackedFunc MicroModuleNode::GetFunction(
    const std::string& name,
    const ObjectPtr<Object>& sptr_to_self) {
  TargetPtr func_ptr;
  if (name == tvm::runtime::symbol::tvm_module_main) {
    if (symbol_map_.HasSymbol(tvm::runtime::symbol::tvm_module_main)) {
      func_ptr = symbol_map_[tvm::runtime::symbol::tvm_module_main];
    } else {
      func_ptr = symbol_map_["default_function"];
    }
  } else {
    func_ptr = symbol_map_[name];
  }
  MicroWrappedFunc f(session_, func_ptr);
  return PackedFunc(f);
}

// register loadfile function to load module from Python frontend
TVM_REGISTER_GLOBAL("runtime.module.loadfile_micro_dev")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    auto n = make_object<MicroModuleNode>();
    n->InitMicroModule(args[0]);
    *rv = runtime::Module(n);
  });
}  // namespace runtime
}  // namespace tvm
