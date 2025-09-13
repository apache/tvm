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
 * \file  module.cc
 * \brief The global module in TVM.
 */
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/rvalue_ref.h>
#include <tvm/ir/global_var_supply.h>
#include <tvm/ir/module.h>
#include <tvm/ir/type_functor.h>
#include <tvm/node/structural_equal.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() { IRModuleNode::RegisterReflection(); }

IRModule::IRModule(tvm::ffi::Map<GlobalVar, BaseFunc> functions, SourceMap source_map,
                   DictAttrs attrs, ffi::Map<ffi::String, ffi::Array<GlobalInfo>> global_infos) {
  auto n = ffi::make_object<IRModuleNode>();
  n->functions = std::move(functions);
  n->global_var_map_ = {};
  n->source_map = source_map;
  n->attrs = std::move(attrs);
  n->global_infos = std::move(global_infos);

  for (const auto& kv : n->functions) {
    // set global var map
    ICHECK(n->global_var_map_.count(kv.first->name_hint) == 0)
        << "Duplicate global function name " << kv.first->name_hint;
    n->global_var_map_.Set(kv.first->name_hint, kv.first);
  }

  data_ = std::move(n);
}

bool IRModuleNode::SEqual(const IRModuleNode* other,
                          ffi::TypedFunction<bool(AnyView, AnyView, bool, AnyView)> equal) const {
  if (!equal(this->attrs, other->attrs, false, "attrs")) {
    return false;
  }
  if (!equal(this->global_infos, other->global_infos, false, "global_infos")) {
    return false;
  }

  // Define remaps for GlobalVar and GlobalTypeVar based on their string name.
  for (const auto& gv : this->GetGlobalVars()) {
    if (other->ContainGlobalVar(gv->name_hint)) {
      if (!equal(gv, other->GetGlobalVar(gv->name_hint), true, "functions")) return false;
    }
  }

  // now check the functions with the GlobalVar remappped
  if (!equal(this->functions, other->functions, false, "functions")) {
    return false;
  }

  return true;
}

uint64_t IRModuleNode::SHash(uint64_t init_hash,
                             ffi::TypedFunction<uint64_t(AnyView, uint64_t, bool)> hash) const {
  uint64_t hash_value = init_hash;
  hash_value = hash(this->attrs, hash_value, false);
  hash_value = hash(this->global_infos, hash_value, false);

  // hash the functions.
  using KV = std::tuple<std::string, ObjectRef, ObjectRef>;
  std::vector<KV> temp;
  for (const auto& kv : this->functions) {
    temp.emplace_back(kv.first->name_hint, kv.first, kv.second);
  }
  // sort by the hash key of the keys.
  std::sort(temp.begin(), temp.end(),
            [](const KV& lhs, const KV& rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });
  hash_value = hash(static_cast<uint64_t>(temp.size()), hash_value, false);
  // first need to define the GlobalVar in the order of the keys
  for (size_t i = 0; i < temp.size(); ++i) {
    hash_value = hash(std::get<1>(temp[i]), hash_value, true);
  }
  // hash the name and content
  for (size_t i = 0; i < temp.size(); ++i) {
    hash_value = hash(std::get<0>(temp[i]), hash_value, false);
    hash_value = hash(std::get<2>(temp[i]), hash_value, false);
  }
  return hash_value;
}

bool IRModuleNode::ContainGlobalVar(const ffi::String& name) const {
  return global_var_map_.find(name) != global_var_map_.end();
}

GlobalVar IRModuleNode::GetGlobalVar(const ffi::String& name) const {
  auto it = global_var_map_.find(name);
  if (it == global_var_map_.end()) {
    std::ostringstream msg;
    msg << "ValueError: Cannot find global var \"" << name << "\" in the Module\n"
        << "candidates are: [";
    int counter = 0;
    for (auto kv : global_var_map_) {
      if (counter++ != 0) {
        msg << ", ";
      }
      msg << "\"" << kv.first << "\"";
    }
    msg << "]";
    LOG(FATAL) << msg.str();
  }
  return (*it).second;
}

tvm::ffi::Array<GlobalVar> IRModuleNode::GetGlobalVars() const {
  std::vector<GlobalVar> global_vars;
  for (const auto& pair : global_var_map_) {
    global_vars.push_back(pair.second);
  }
  std::sort(global_vars.begin(), global_vars.end(), [](const GlobalVar& lhs, const GlobalVar& rhs) {
    return lhs->name_hint < rhs->name_hint;
  });
  return tvm::ffi::Array<GlobalVar>(global_vars);
}

void IRModuleNode::Add(const GlobalVar& var, const BaseFunc& f, bool update) {
  BaseFunc checked_func = f;
  AddUnchecked(var, checked_func);
}

void IRModuleNode::AddUnchecked(const GlobalVar& var, const BaseFunc& func) {
  this->functions.Set(var, func);

  auto it = global_var_map_.find(var->name_hint);
  if (it != global_var_map_.end()) {
    ICHECK_EQ((*it).second, var);
  } else {
    ICHECK(global_var_map_.count(var->name_hint) == 0) << "Duplicate global function name " << var;
  }

  global_var_map_.Set(var->name_hint, var);
}

void IRModuleNode::Update(const GlobalVar& var, const BaseFunc& func) {
  this->Add(var, func, true);
}

void IRModuleNode::UpdateGlobalInfo(const ffi::String& name, const ffi::Array<GlobalInfo>& info) {
  this->global_infos.Set(name, info);
}

void IRModuleNode::Remove(const GlobalVar& var) {
  auto functions_node = this->functions.CopyOnWrite();
  functions_node->erase(var);
  auto gvar_node = global_var_map_.CopyOnWrite();
  gvar_node->erase(var->name_hint);
}

BaseFunc IRModuleNode::Lookup(const GlobalVar& var) const {
  auto it = functions.find(var);
  ICHECK(it != functions.end()) << "There is no definition of " << var;
  return (*it).second;
}

BaseFunc IRModuleNode::Lookup(const ffi::String& name) const {
  GlobalVar id = this->GetGlobalVar(name);
  return this->Lookup(id);
}

void IRModuleNode::Update(const IRModule& mod) {
  for (auto pair : mod->functions) {
    // TODO(@jroesch): rename into IRModule.
    this->AddUnchecked(pair.first, pair.second);
  }
}

IRModule IRModuleNode::ShallowCopy() {
  return IRModule(this->functions, this->source_map, this->attrs, this->global_infos);
}

IRModule IRModule::FromExpr(const RelaxExpr& expr,
                            const tvm::ffi::Map<GlobalVar, BaseFunc>& global_funcs) {
  auto mod = IRModule(global_funcs);
  ffi::String gv_name;

  // All global definitions must be functions.
  BaseFunc func;
  if (auto func_node = expr.as<BaseFunc>()) {
    func = func_node.value();
    if (auto opt = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
      // Function literal has been annotated with it's required global symbol.
      gv_name = opt.value();
    }
  }

  GlobalVar main_gv;
  auto global_var_supply = GlobalVarSupply(mod);
  if (gv_name.empty()) {
    // Bind function to 'main' (though rename if would clash with existing 'main').
    main_gv = global_var_supply->FreshGlobal("main", false);
  } else {
    main_gv = global_var_supply->UniqueGlobalFor(gv_name, false);
  }
  mod->Add(main_gv, func);
  return mod;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ir.IRModule",
           [](tvm::ffi::Map<GlobalVar, BaseFunc> funcs, tvm::ObjectRef attrs,
              ffi::Map<ffi::String, ffi::Array<GlobalInfo>> global_infos) {
             auto dict_attrs = [&attrs]() {
               if (!attrs.defined()) {
                 return DictAttrs();
               } else if (auto* as_dict_attrs = attrs.as<tvm::DictAttrsNode>()) {
                 return ffi::GetRef<tvm::DictAttrs>(as_dict_attrs);
               } else if (attrs.as<ffi::MapObj>()) {
                 return tvm::DictAttrs(Downcast<ffi::Map<ffi::String, Any>>(attrs));
               } else {
                 LOG(FATAL) << "Expected attrs argument to be either DictAttrs or "
                               "ffi::Map<ffi::String,ObjectRef>";
               }
             }();

             return IRModule(funcs, {}, dict_attrs, global_infos);
           })
      .def("ir.Module_Clone",
           [](IRModule mod) -> IRModule {
             IRModule clone = mod;
             clone.CopyOnWrite();
             return clone;
           })
      .def("ir.Module_Add",
           [](IRModule mod, GlobalVar var, ObjectRef val, bool update) -> IRModule {
             ICHECK(val->IsInstance<RelaxExprNode>());
             mod->Add(var, Downcast<BaseFunc>(val), update);
             return mod;
           })
      .def("ir.Module_Remove",
           [](IRModule mod, ffi::Variant<ffi::String, GlobalVar> var) -> IRModule {
             GlobalVar gvar = [&]() {
               if (auto opt = var.as<GlobalVar>()) {
                 return opt.value();
               } else if (auto opt = var.as<ffi::String>()) {
                 return mod->GetGlobalVar(opt.value());
               } else {
                 LOG(FATAL) << "InternalError: "
                            << "Variant didn't contain any of the allowed types";
               }
             }();
             mod->Remove(gvar);
             return mod;
           })
      .def("ir.Module_Contains",
           [](IRModule mod, ffi::Variant<ffi::String, GlobalVar> var) -> bool {
             if (auto opt = var.as<GlobalVar>()) {
               return mod->functions.count(opt.value());
             } else if (auto opt = var.as<ffi::String>()) {
               return mod->global_var_map_.count(opt.value());
             } else {
               LOG(FATAL) << "InternalError: "
                          << "Variant didn't contain any of the allowed types";
             }
           })
      .def_method("ir.Module_GetGlobalVar", &IRModuleNode::GetGlobalVar)
      .def_method("ir.Module_GetGlobalVars", &IRModuleNode::GetGlobalVars)
      .def_method("ir.Module_ContainGlobalVar", &IRModuleNode::ContainGlobalVar)
      .def("ir.Module_Lookup", [](IRModule mod, GlobalVar var) { return mod->Lookup(var); })
      .def("ir.Module_Lookup_str", [](IRModule mod, ffi::String var) { return mod->Lookup(var); })
      .def("ir.Module_FromExpr", &IRModule::FromExpr)
      .def("ir.Module_Update", [](IRModule mod, IRModule from) { mod->Update(from); })
      .def("ir.Module_UpdateFunction",
           [](IRModule mod, GlobalVar gv, BaseFunc func) { mod->Update(gv, func); })
      .def("ir.Module_UpdateGlobalInfo",
           [](IRModule mod, ffi::String name, ffi::Array<GlobalInfo> global_info) {
             mod->UpdateGlobalInfo(name, global_info);
           })
      .def("ir.Module_GetAttrs", [](IRModule mod) -> ObjectRef { return mod->GetAttrs(); })
      .def("ir.Module_WithAttr",
           [](ffi::RValueRef<IRModule> mod, ffi::String key, ffi::Any value) -> IRModule {
             return WithAttr(*std::move(mod), key, value);
           })
      .def("ir.Module_WithoutAttr",
           [](ffi::RValueRef<IRModule> mod, ffi::String key) -> IRModule {
             return WithoutAttr(*std::move(mod), key);
           })
      .def("ir.Module_WithAttrs",
           [](ffi::RValueRef<IRModule> mod, ffi::Map<ffi::String, ffi::Any> attr_map) -> IRModule {
             return WithAttrs(*std::move(mod), attr_map);
           })
      .def("ir.Module_GetAttr",
           [](IRModule mod, ffi::String key) -> ObjectRef { return mod->GetAttr<ObjectRef>(key); });
}

}  // namespace tvm
