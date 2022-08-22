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
 * \file src/relay/backend/aot_executor_codegen.cc
 * \brief AOT executor codegen
 */

#include <tvm/ir/module.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/call.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/runtime.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/usmp/utils.h>

#include <algorithm>
#include <list>
#include <string>
#include <vector>

#include "../te_compiler.h"
#include "../utils.h"
#include "./aot_lower_main.h"
#include "./create_executor_metadata.h"
#include "./create_function_metadata.h"

namespace tvm {
namespace relay {
namespace backend {
namespace aot {

std::unordered_map<std::string, tvm::runtime::NDArray> CreateParamMap(
    const IRModule& mod, const std::unordered_map<std::string, runtime::NDArray>& external_params) {
  auto params = std::unordered_map<std::string, tvm::runtime::NDArray>();
  // Collect any constants extracted by external codegen.
  Map<String, runtime::NDArray> const_name_to_constant =
      mod->GetAttr<Map<String, runtime::NDArray>>(tvm::attr::kConstNameToConstant).value_or({});
  for (const auto& kv : const_name_to_constant) {
    params[kv.first] = kv.second;
  }

  // Collect any constants extracted during lowering.
  for (const auto& kv : external_params) {
    params[kv.first] = kv.second;
  }

  return params;
}

LoweredOutput Codegen(IRModule mod, String mod_name, CompilationConfig config, Executor executor,
                      CallType call_type) {
  Integer workspace_byte_alignment =
      executor->GetAttr<Integer>("workspace-byte-alignment").value_or(16);
  Integer constant_byte_alignment =
      executor->GetAttr<Integer>("constant-byte-alignment").value_or(16);
  // Required Relay passes prior to AOT codegen (should be refactored out of executors)
  mod = transform::ToANormalForm()(mod);
  mod = transform::InferType()(mod);
  mod = transform::AnnotateUsedMemory()(mod);  // TODO(mbaret) Move into Ethos-U hook
  std::unordered_map<std::string, runtime::NDArray> external_params;
  mod = tec::LowerTE(mod_name, config, [&external_params](BaseFunc func) {
    if (func->GetAttr<String>(attr::kCompiler).defined()) {
      UpdateConstants(func, &external_params);
    }
  })(mod);

  transform::PassContext pass_ctx = transform::PassContext::Current();
  bool enable_remove_reshapes =
      pass_ctx->GetConfig<Bool>("relay.remove_standalone_reshapes.enable", Bool(true)).value();
  if (enable_remove_reshapes) {
    mod = transform::RemoveStandaloneReshapes()(mod);
  }

  // Lower the main Relay function to a TIR PrimFunc
  // After this point the entire module is composed of PrimFuncs
  mod = AOTLowerMain(mod_name, config, call_type)(mod);

  mod = tir::transform::ConvertForLoopsToSerial()(mod);  // TODO(mbaret) Make this optional
  bool enable_usmp = pass_ctx->GetConfig<Bool>(kUSMPEnableOption, Bool(false)).value();
  if (enable_usmp) {
    mod = tir::transform::UnifiedStaticMemoryPlanner()(mod);
  } else {
    tir::PrimFunc tir_main_func =
        Downcast<tir::PrimFunc>(mod->Lookup(::tvm::runtime::symbol::tvm_module_main));
    IRModule main_func_mod;
    main_func_mod->Update(mod->GetGlobalVar(::tvm::runtime::symbol::tvm_module_main),
                          tir_main_func);
    main_func_mod = tir::transform::StorageRewrite()(main_func_mod);
    mod->Update(mod->GetGlobalVar(::tvm::runtime::symbol::tvm_module_main),
                main_func_mod->Lookup(::tvm::runtime::symbol::tvm_module_main));
  }
  mod = tir::transform::LegalizePackedCalls()(mod);

  // Collect the various functions, params and metadata into a LoweredOutput
  LoweredOutput ret;
  ret.params = CreateParamMap(mod, external_params);
  ret.external_mods =
      mod->GetAttr<Array<tvm::runtime::Module>>(tvm::attr::kExternalMods).value_or({});
  ret.function_metadata =
      std::move(CreateFunctionMetadata(mod, workspace_byte_alignment, constant_byte_alignment));
  ret.lowered_funcs = tec::GetPerTargetModules(mod);
  ret.metadata = CreateExecutorMetadata(mod, mod_name, executor, workspace_byte_alignment,
                                        constant_byte_alignment);
  return LoweredOutput(std::move(ret));
}

class AOTExecutorCodegenModule : public runtime::ModuleNode {
 public:
  AOTExecutorCodegenModule() {}
  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
    if (name == "init") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        // Do nothing
      });
    } else if (name == "codegen") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        IRModule mod = args[0];
        Function func = args[1];
        String mod_name = args[2];
        CompilationConfig config = args[3];
        Executor executor = args[4];
        Integer call_type = args[5];
        this->output_ =
            Codegen(mod, mod_name, config, executor, static_cast<CallType>(call_type->value));
      });
    } else if (name == "list_params_name") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = list_params_name(); });
    } else if (name == "get_param_by_name") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        String key = args[0];
        *rv = get_param_by_name(key);
      });
    } else if (name == "get_irmodule") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = get_irmodule(); });
    } else if (name == "get_external_modules") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = get_external_modules(); });
    } else if (name == "get_function_metadata") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->output_.function_metadata;
      });
    } else if (name == "get_devices") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = output_.metadata->devices; });
    } else if (name == "get_executor_codegen_metadata") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = output_.metadata; });
    } else {
      return PackedFunc([](TVMArgs args, TVMRetValue* rv) {});
    }
  }

  const char* type_key() const final { return "RelayGraphRuntimeCodegenModule"; }

 private:
  Array<runtime::String> list_params_name() {
    Array<runtime::String> ret;
    for (const auto& kv : this->output_.params) {
      ret.push_back(kv.first);
    }
    return ret;
  }

  runtime::NDArray get_param_by_name(String key) {
    auto it = this->output_.params.find(key);
    CHECK(it != this->output_.params.end()) << "no such parameter " << key;
    return (*it).second;
  }

  Array<tvm::runtime::Module> get_external_modules() { return output_.external_mods; }

  Map<Target, IRModule> get_irmodule() { return this->output_.lowered_funcs; }

  LoweredOutput output_;
};

runtime::Module CreateAOTExecutorCodegenMod() {
  auto ptr = make_object<AOTExecutorCodegenModule>();
  return runtime::Module(ptr);
}

TVM_REGISTER_GLOBAL("relay.build_module._AOTExecutorCodegen")
    .set_body([](TVMArgs args, TVMRetValue* rv) { *rv = CreateAOTExecutorCodegenMod(); });

}  // namespace aot
}  // namespace backend
}  // namespace relay
}  // namespace tvm
