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
#include <tvm/relax/analysis.h>
#include <tvm/runtime/registry.h>
#include <tvm/script/ir_builder/ir/ir.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace ir {

IRModuleFrame IRModule() {
  ObjectPtr<IRModuleFrameNode> n = make_object<IRModuleFrameNode>();
  n->global_var_map.clear();
  n->functions.clear();
  return IRModuleFrame(n);
}

GlobalVar DeclFunction(const String& func_name, const BaseFunc& func_signature) {
  IRModuleFrame frame = FindModuleFrame("I.DeclFunction");
  CHECK(!frame->global_var_map.count(func_name))
      << "ValueError: function " << func_name << " already exists";
  GlobalVar gv = GlobalVar(func_name);
  if (func_signature->struct_info_.defined()) {
    gv->struct_info_ = tvm::relax::GetStructInfo(func_signature);
  } else if (const auto* prim_func = func_signature.as<tvm::tir::PrimFuncNode>()) {
    gv->struct_info_ =
        tvm::relax::FuncStructInfo::OpaqueFunc(tvm::relax::StructInfoFromType(prim_func->ret_type));
  } else {
    LOG(FATAL) << "Unsupported function type: " << func_signature->GetTypeKey();
  }
  CHECK(frame->functions.find(gv) == frame->functions.end())
      << "ValueError: function " << func_name << " has already been defined.";
  frame->global_var_map.Set(func_name, gv);
  if (func_signature.defined()) {
    frame->functions.Set(gv, func_signature);
  }
  return gv;
}

void DefFunction(const String& func_name, const BaseFunc& func) {
  IRModuleFrame frame = FindModuleFrame("I.DefFunction");
  auto it = frame->global_var_map.find(func_name);
  CHECK(it != frame->global_var_map.end())
      << "ValueError: function " << func_name << " does not exist, please declare it first.";
  const GlobalVar& gv = (*it).second;
  frame->functions.Set(gv, func);
  if (func->checked_type_.defined()) {
    gv->checked_type_ = func->checked_type_;
  }
}

GlobalVar AddFunction(const BaseFunc& func, String func_name_hint) {
  IRBuilder builder = IRBuilder::Current();
  GlobalVar gvar = GlobalVar(func_name_hint);
  if (func->struct_info_.defined()) {
    gvar->struct_info_ = tvm::relax::GetStructInfo(func);
  } else if (const auto* prim_func = func.as<tvm::tir::PrimFuncNode>()) {
    gvar->struct_info_ =
        tvm::relax::FuncStructInfo::OpaqueFunc(tvm::relax::StructInfoFromType(prim_func->ret_type));
  } else {
    LOG(FATAL) << "Unsupported function type: " << func->GetTypeKey();
  }
  if (Optional<ir::IRModuleFrame> opt_frame = builder->FindFrame<ir::IRModuleFrame>()) {
    const ir::IRModuleFrame& mod_frame = opt_frame.value();
    if (mod_frame->global_var_map.count(func_name_hint)) {
      return mod_frame->global_var_map[func_name_hint];
    }
    mod_frame->global_var_map.Set(func_name_hint, gvar);
    mod_frame->functions.Set(gvar, func);
  } else {
    LOG(FATAL) << "ValueError: IRModule frame not find. Please ensure "
               << "AddFunction called under I.ir_module()";
  }
  return gvar;
}

void UpdateFunction(const GlobalVar& gv, BaseFunc function) {
  IRBuilder builder = IRBuilder::Current();
  if (Optional<ir::IRModuleFrame> opt_frame = builder->FindFrame<ir::IRModuleFrame>()) {
    const ir::IRModuleFrame& mod_frame = opt_frame.value();
    // invalidate old function first
    if (mod_frame->global_var_map.count(gv->name_hint)) {
      mod_frame->global_var_map.erase(gv->name_hint);
      mod_frame->functions.erase(gv);
    }
    // add func in
    mod_frame->global_var_map.Set(gv->name_hint, gv);
    mod_frame->functions.Set(gv, function);
  } else {
    LOG(FATAL) << "ValueError: IRModule frame not find. Please ensure "
               << "UpdateFunction called under I.ir_module()";
  }
}

void ModuleAttrs(Map<String, ObjectRef> attrs) {
  if (IRBuilder::IsInScope()) {
    // TODO(hongyi): add comments to explain why we need to check if the module frame is in scope
    IRModuleFrame frame = FindModuleFrame("I.ModuleAttr");
    if (!frame->attrs.empty()) {
      LOG(FATAL) << "ValueError: Duplicate module attrs, previous one is:\n" << frame->attrs;
    }
    frame->attrs = attrs;
  }
}

void ModuleGlobalInfos(Map<String, Array<GlobalInfo>> global_infos) {
  if (IRBuilder::IsInScope()) {
    IRModuleFrame frame = FindModuleFrame("I.ModuleGlobalInfos");
    if (!frame->global_infos.empty()) {
      LOG(FATAL) << "ValueError: Duplicate module global_infos, previous one is:\n"
                 << frame->global_infos;
    }
    frame->global_infos = global_infos;
  }
}

TVM_REGISTER_GLOBAL("script.ir_builder.ir.IRModule").set_body_typed(IRModule);
TVM_REGISTER_GLOBAL("script.ir_builder.ir.DeclFunction").set_body_typed(DeclFunction);
TVM_REGISTER_GLOBAL("script.ir_builder.ir.DefFunction").set_body_typed(DefFunction);
TVM_REGISTER_GLOBAL("script.ir_builder.ir.AddFunction").set_body_typed(AddFunction);
TVM_REGISTER_GLOBAL("script.ir_builder.ir.UpdateFunction").set_body_typed(UpdateFunction);
TVM_REGISTER_GLOBAL("script.ir_builder.ir.ModuleAttrs").set_body_typed(ModuleAttrs);
TVM_REGISTER_GLOBAL("script.ir_builder.ir.ModuleGlobalInfos").set_body_typed(ModuleGlobalInfos);

}  // namespace ir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
