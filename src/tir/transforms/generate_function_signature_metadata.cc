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
 * \file generate_function_signature_metadata.cc
 * \brief Split device function from host.
 */

#define PICOJSON_USE_INT64
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <picojson.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

namespace {
picojson::value GenerateMetadata(const PrimFunc& func) {
  std::vector<picojson::value> params;

  return picojson::value(picojson::object({
      {"params", picojson::value(params)},
  }));
}

picojson::value GenerateMetadata(const IRModule& mod) {
  picojson::object functions;
  for (const auto& [gvar, base_func] : mod->functions) {
    bool is_externally_exposed = base_func->GetAttr<String>(tvm::attr::kGlobalSymbol).defined();
    if (auto func = base_func.as<PrimFunc>(); func && is_externally_exposed) {
      functions[gvar->name_hint] = GenerateMetadata(func.value());
    }
  }

  return picojson::value(picojson::object({
      {"functions", picojson::value(functions)},
  }));
}

std::string GenerateMetadataString(const IRModule& mod) {
  return GenerateMetadata(mod).serialize(/* prettify = */ true);
}
}  // namespace

namespace transform {

Pass GenerateFunctionSignatureMetadata() {
  auto pass_func = [](IRModule mod, PassContext ctx) -> IRModule {
    if (mod->ContainGlobalVar(runtime::symbol::tvm_get_tir_function_metadata)) {
      return mod;
    }

    std::string metadata = GenerateMetadataString(mod);

    Map<String, ObjectRef> func_attrs{
        {tvm::attr::kGlobalSymbol, String(runtime::symbol::tvm_get_tir_function_metadata)},
        {tvm::tir::attr::kIsHostFunc, Bool(true)},
    };
    Type ret_type = PrimType(DataType::Handle());
    PrimFunc metadata_func({}, Evaluate(ret(StringImm(metadata))), ret_type, {},
                           DictAttrs(func_attrs));
    GlobalVar gvar(runtime::symbol::tvm_get_tir_function_metadata, FuncType({}, ret_type, {}, {}));

    mod.CopyOnWrite()->Add(gvar, metadata_func);
    return mod;
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tir.GenerateFunctionSignatureMetadata",
                                          {});
}

TVM_REGISTER_GLOBAL("tir.transform.GenerateFunctionSignatureMetadata")
    .set_body_typed(GenerateFunctionSignatureMetadata);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
