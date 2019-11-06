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
  // TODO(weberlo): enqueue each loaded module into a vector of bin contents.
  // then concatenate the contents, build it, and flush it once a function call
  // is attempted.
  //
  // We might not be able to flush *all* sections.  Depends how st-flash works.
  // it only asks to specify the start of the flash section, so does it also
  // flash the RAM sections? It's also weird that it asks for the start of the
  // flash section, because that should already be encoded in the binary. check
  // the .bin files to see if symbol addrs are assigned. also, check the
  // st-flash docs, because the arg could just be for the address of `main`.

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
    std::cout << "[MicroModuleNode::InitMicroModule]" << std::endl;
    std::cout << "  start" << std::endl;
    session_ = MicroSession::Current();
    symbol_map_ = session_->LoadBinary(binary_path, true).symbol_map;
    std::cout << "  end" << std::endl;
  }

  ///*!
  // * \brief runs selected function on the micro device
  // * \param func_name name of the function to be run
  // * \param func_ptr offset of the function to be run
  // * \param args type-erased arguments passed to the function
  // */
  //void RunFunction(DevPtr func_ptr, const TVMArgs& args) {
  //  session_->PushToExecQueue(func_ptr, args);
  //}

 private:
  SymbolMap symbol_map_;
  /*! \brief global session pointer */
  ObjectPtr<MicroSession> session_;
};

class MicroWrappedFunc {
 public:
  MicroWrappedFunc(ObjectPtr<MicroSession> session,
                   DevPtr func_ptr) {
    session_ = session;
    func_ptr_ = func_ptr;
  }

  void operator()(TVMArgs args, TVMRetValue* rv) const {
    std::cout << "[MicroWrappedFunc::operator()]" << std::endl;
    *rv = session_->PushToExecQueue(func_ptr_, args);
  }

 private:
  /*! \brief reference to the session for this function (to keep the session alive) */
  std::shared_ptr<MicroSession> session_;
  /*! \brief offset of the function to be called */
  DevPtr func_ptr_;
};

PackedFunc MicroModuleNode::GetFunction(
    const std::string& name,
    const ObjectPtr<Object>& sptr_to_self) {
  std::cout << "[MicroModuleNode::GetFunction(name=" << name << ")]" << std::endl;
  DevPtr func_ptr;
  if (name == tvm::runtime::symbol::tvm_module_main) {
    std::cout << "  here" << std::endl;
    if (symbol_map_.HasSymbol(tvm::runtime::symbol::tvm_module_main)) {
      std::cout << "  ayy" << std::endl;
      func_ptr = symbol_map_[tvm::runtime::symbol::tvm_module_main];
    } else {
      std::cout << "  lmao" << std::endl;
      func_ptr = symbol_map_["default_function"];
    }
    //std::cout << "  symbols:" << std::endl;
    //for (const auto& sym_name : symbol_map_.GetSymbols()) {
    //  std::cout << "    " << sym_name << std::endl;
    //}
    //CHECK(symbol_map_.size() == 1) << "entry point requested with multiple functions in module";
    //func_ptr = symbol_map_[symbol_map_.GetSymbols()[0]];
  } else {
    func_ptr = symbol_map_[name];
  }
  MicroWrappedFunc f(session_, func_ptr);
  return PackedFunc(f);
}

// register loadfile function to load module from Python frontend
TVM_REGISTER_GLOBAL("module.loadfile_micro_dev")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    auto n = make_object<MicroModuleNode>();
    n->InitMicroModule(args[0]);
    *rv = runtime::Module(n);
  });
}  // namespace runtime
}  // namespace tvm
