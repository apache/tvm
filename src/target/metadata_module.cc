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
 * \file metadata_module.cc
 * \brief Defines functions that build MetadataModules for C++ and C runtimes.
 */
#include "metadata_module.h"

#include <vector>

#include "../runtime/meta_data.h"
#include "llvm/llvm_module.h"
#include "source/source_module.h"

namespace tvm {
namespace codegen {

/*!
 * \brief Create a metadata module wrapper. The helper is used by different
 *        codegens, such as graph executor codegen and the vm compiler.
 *
 * \param params The metadata for initialization of all modules.
 * \param target_module the internal module that is compiled by tvm.
 * \param ext_modules The external modules that needs to be imported inside the metadata
 * module(s).
 * \param target The target that all the modules are compiled for
 * \return The created metadata module that manages initialization of metadata.
 */
runtime::Module CreateMetadataModule(
    const std::unordered_map<std::string, runtime::NDArray>& params,
    tvm::runtime::Module target_module, const Array<runtime::Module>& ext_modules, Target target,
    runtime::Metadata metadata) {
  // Here we split modules into two groups:
  //  1. Those modules which can be exported to C-runtime. These are DSO-exportable
  //     (i.e. llvm or c) modules which return nothing from get_const_vars().
  //  2. Other modules.
  Array<runtime::Module> crt_exportable_modules;
  Array<runtime::Module> non_crt_exportable_modules;

  auto DSOExportable = [](tvm::runtime::Module& mod) {
    return !std::strcmp(mod->type_key(), "llvm") || !std::strcmp(mod->type_key(), "c");
  };

  bool is_targeting_crt =
      target.defined() && target->GetAttr<String>("runtime").value_or(String("")) == kTvmRuntimeCrt;

  // Wrap all submodules in the initialization wrapper.
  std::unordered_map<std::string, std::vector<std::string>> sym_metadata;
  for (tvm::runtime::Module mod : ext_modules) {
    auto pf_sym = mod.GetFunction("get_symbol");
    auto pf_var = mod.GetFunction("get_const_vars");
    std::vector<std::string> arrays;
    if (pf_sym != nullptr && pf_var != nullptr) {
      String symbol = pf_sym();
      Array<String> variables = pf_var();
      for (size_t i = 0; i < variables.size(); i++) {
        arrays.push_back(variables[i].operator std::string());
      }
      ICHECK_EQ(sym_metadata.count(symbol), 0U) << "Found duplicated symbol: " << symbol;
      sym_metadata[symbol] = arrays;
    }
    // We only need loading of serialized constant data
    // if there are constants present and required by the
    // runtime module to be initialized by the binary
    // metadata module. If not rest of the modules are
    // wrapped in c-source metadata module.

    // TODO(@manupa-arm) : we should be able to use csource_metadata
    // if the variables are empty when all the runtime modules implement get_func_names
    if (arrays.empty() && is_targeting_crt && DSOExportable(mod) &&
        (target->kind->name == "c" || target->kind->name == "llvm")) {
      crt_exportable_modules.push_back(mod);
    } else {
      non_crt_exportable_modules.push_back(mod);
    }
  }

  if (is_targeting_crt) {
    if (!non_crt_exportable_modules.empty()) {
      std::string non_exportable_modules;
      for (unsigned int i = 0; i < non_crt_exportable_modules.size(); i++) {
        if (i > 0) {
          non_exportable_modules += ", ";
        }
        auto mod = non_crt_exportable_modules[i];
        auto pf_sym = mod.GetFunction("get_symbol");
        if (pf_sym != nullptr) {
          non_exportable_modules += pf_sym().operator std::string();
        } else {
          non_exportable_modules +=
              std::string{"(module type_key="} + mod->type_key() + std::string{")"};
        }
      }
      CHECK(false) << "These " << non_crt_exportable_modules.size()
                   << " modules are not exportable to C-runtime: " << non_exportable_modules;
    }

    if (target->kind->name == "c") {
      crt_exportable_modules.push_back(target_module);
      target_module = CreateCSourceCrtMetadataModule(crt_exportable_modules, target, metadata);
    } else if (target->kind->name == "llvm") {
#ifdef TVM_LLVM_VERSION
      crt_exportable_modules.push_back(target_module);
      target_module = CreateLLVMCrtMetadataModule(crt_exportable_modules, target);
#else   // TVM_LLVM_VERSION
      LOG(FATAL) << "TVM was not built with LLVM enabled.";
#endif  // TVM_LLVM_VERSION
    }
  } else {
    if (!non_crt_exportable_modules.empty()) {
      runtime::Module binary_meta_mod = runtime::MetadataModuleCreate(params, sym_metadata);
      binary_meta_mod.Import(target_module);
      for (const auto& it : non_crt_exportable_modules) {
        binary_meta_mod.Import(it);
      }
      return binary_meta_mod;
    }
  }
  return target_module;
}

}  // namespace codegen
}  // namespace tvm
