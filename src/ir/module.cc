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
#include <tvm/ir/global_var_supply.h>
#include <tvm/ir/module.h>
#include <tvm/ir/type_functor.h>
#include <tvm/node/structural_equal.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <sstream>
#include <unordered_set>

namespace tvm {

IRModule::IRModule(tvm::Map<GlobalVar, BaseFunc> functions,
                   tvm::Map<GlobalTypeVar, TypeData> type_definitions,
                   std::unordered_set<String> import_set, SourceMap source_map, DictAttrs attrs,
                   Map<String, Array<GlobalInfo>> global_infos) {
  auto n = make_object<IRModuleNode>();
  n->functions = std::move(functions);
  n->type_definitions = std::move(type_definitions);
  n->global_type_var_map_ = {};
  n->global_var_map_ = {};
  n->constructor_tag_map_ = {};
  n->import_set_ = std::move(import_set);
  n->source_map = source_map;
  n->attrs = std::move(attrs);
  n->global_infos = std::move(global_infos);

  for (const auto& kv : n->functions) {
    // set global var map
    ICHECK(n->global_var_map_.count(kv.first->name_hint) == 0)
        << "Duplicate global function name " << kv.first->name_hint;
    n->global_var_map_.Set(kv.first->name_hint, kv.first);
  }

  for (const auto& kv : n->type_definitions) {
    // set global typevar map
    ICHECK(n->global_type_var_map_.count(kv.first->name_hint) == 0)
        << "Duplicate global type definition name " << kv.first->name_hint;
    n->global_type_var_map_.Set(kv.first->name_hint, kv.first);
    n->RegisterConstructors(kv.first, kv.second);
  }
  data_ = std::move(n);
}

bool IRModuleNode::SEqualReduce(const IRModuleNode* other, SEqualReducer equal) const {
  if (!equal(this->attrs, other->attrs, [](const auto& path) { return path->Attr("attrs"); })) {
    return false;
  }

  if (this->global_infos.size() != other->global_infos.size()) return false;
  for (const auto& kv : this->global_infos) {
    if (!equal(kv.second, other->global_infos[kv.first])) return false;
  }

  if (functions.size() != other->functions.size()) return false;
  // Update GlobalVar remap
  if (equal.IsPathTracingEnabled()) {
    if ((functions.size() != other->functions.size()) ||
        (type_definitions.size() != other->type_definitions.size())) {
      return false;
    }
  }

  // Define remaps for GlobalVar and GlobalTypeVar based on their
  // string name.  Early bail-out is only performed when path-tracing
  // is disabled, as the later equality checks on the member variables
  // will provide better error messages.
  for (const auto& gv : this->GetGlobalVars()) {
    if (other->ContainGlobalVar(gv->name_hint)) {
      if (!equal.DefEqual(gv, other->GetGlobalVar(gv->name_hint))) return false;
    } else if (!equal.IsPathTracingEnabled()) {
      return false;
    }
  }
  for (const auto& gtv : this->GetGlobalTypeVars()) {
    if (other->ContainGlobalTypeVar(gtv->name_hint)) {
      if (!equal.DefEqual(gtv, other->GetGlobalTypeVar(gtv->name_hint))) return false;
    } else if (!equal.IsPathTracingEnabled()) {
      return false;
    }
  }

  // Checking functions and type definitions
  if (!equal(this->functions, other->functions,
             [](const auto& path) { return path->Attr("functions"); })) {
    return false;
  }
  if (!equal(this->type_definitions, other->type_definitions,
             [](const auto& path) { return path->Attr("type_definitions"); })) {
    return false;
  }

  return true;
}

void IRModuleNode::SHashReduce(SHashReducer hash_reduce) const {
  using KV = std::tuple<std::string, ObjectRef, ObjectRef>;
  // hash the functions.
  std::vector<KV> temp;

  auto reduce_temp = [&]() {
    // sort by the hash key of the keys.
    std::sort(temp.begin(), temp.end(),
              [](const KV& lhs, const KV& rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });

    hash_reduce(static_cast<uint64_t>(temp.size()));
    // Defhash the GlobalVar/GlobalTypeVar
    for (size_t i = 0; i < temp.size(); ++i) {
      hash_reduce.DefHash(std::get<1>(temp[i]));
    }
    // hash the name and content
    for (size_t i = 0; i < temp.size(); ++i) {
      hash_reduce(std::get<0>(temp[i]));
      hash_reduce(std::get<2>(temp[i]));
    }
  };

  for (const auto& kv : this->functions) {
    temp.emplace_back(kv.first->name_hint, kv.first, kv.second);
  }
  reduce_temp();

  temp.clear();
  for (const auto& kv : this->type_definitions) {
    temp.emplace_back(kv.first->name_hint, kv.first, kv.second);
  }
  reduce_temp();
  hash_reduce(this->attrs);
  hash_reduce(this->global_infos);
}

bool IRModuleNode::ContainGlobalVar(const String& name) const {
  return global_var_map_.find(name) != global_var_map_.end();
}

bool IRModuleNode::ContainGlobalTypeVar(const String& name) const {
  return global_type_var_map_.find(name) != global_type_var_map_.end();
}

GlobalVar IRModuleNode::GetGlobalVar(const String& name) const {
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

tvm::Array<GlobalVar> IRModuleNode::GetGlobalVars() const {
  std::vector<GlobalVar> global_vars;
  for (const auto& pair : global_var_map_) {
    global_vars.push_back(pair.second);
  }
  return tvm::Array<GlobalVar>(global_vars);
}

GlobalTypeVar IRModuleNode::GetGlobalTypeVar(const String& name) const {
  ICHECK(global_type_var_map_.defined());
  auto it = global_type_var_map_.find(name);
  ICHECK(it != global_type_var_map_.end())
      << "Cannot find global type var " << name << " in the Module";
  return (*it).second;
}

Constructor IRModuleNode::GetConstructor(const String& adt, const String& cons) const {
  TypeData typeDef = this->LookupTypeDef(adt);
  for (Constructor c : typeDef->constructors) {
    if (cons.compare(c->name_hint) == 0) {
      return c;
    }
  }

  LOG(FATAL) << adt << " does not contain constructor " << cons;
}

tvm::Array<GlobalTypeVar> IRModuleNode::GetGlobalTypeVars() const {
  std::vector<GlobalTypeVar> global_type_vars;
  for (const auto& pair : global_type_var_map_) {
    global_type_vars.push_back(pair.second);
  }
  return tvm::Array<GlobalTypeVar>(global_type_vars);
}

void IRModuleNode::Add(const GlobalVar& var, const BaseFunc& f, bool update) {
  BaseFunc checked_func = f;
  if (const auto* f = runtime::Registry::Get("relay.ir.WarnIfMalformed")) {
    (*f)(GetRef<IRModule>(this), checked_func);
  }
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

void IRModuleNode::RegisterConstructors(const GlobalTypeVar& var, const TypeData& type) {
  // We hash the global type var name to use as a globally unique prefix for tags.
  // The hash will be used as the most significant byte of the tag, with the index of
  // the constructor in the less significant bytes
  size_t hash = std::hash<std::string>()(var->name_hint);
  int32_t prefix = static_cast<int32_t>(hash & 0xff) << 24;
  for (size_t i = 0; i < type->constructors.size(); ++i) {
    type->constructors[i]->tag = prefix | static_cast<int32_t>(i);
    constructor_tag_map_[type->constructors[i]->tag] = type->constructors[i];
  }
}

void IRModuleNode::AddTypeDef(const GlobalTypeVar& var, const TypeData& type, bool update) {
  // TODO(@jroesch): we have temporarily removed kind checking here, and will consolidate
  // to the type checker in follow up PR.
  AddTypeDefUnchecked(var, type, update);
}

void IRModuleNode::AddTypeDefUnchecked(const GlobalTypeVar& var, const TypeData& type,
                                       bool update) {
  this->type_definitions.Set(var, type);
  if (!update) {
    // set global type var map
    ICHECK(global_type_var_map_.count(var->name_hint) == 0)
        << "Duplicate global type definition name " << var;
  }
  global_type_var_map_.Set(var->name_hint, var);
  RegisterConstructors(var, type);
}

void IRModuleNode::Update(const GlobalVar& var, const BaseFunc& func) {
  this->Add(var, func, true);
}

void IRModuleNode::UpdateTypeDef(const GlobalTypeVar& var, const TypeData& type) {
  this->AddTypeDef(var, type, true);
}

void IRModuleNode::UpdateGlobalInfo(const String& name, const Array<GlobalInfo>& info) {
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

BaseFunc IRModuleNode::Lookup(const String& name) const {
  GlobalVar id = this->GetGlobalVar(name);
  return this->Lookup(id);
}

TypeData IRModuleNode::LookupTypeDef(const GlobalTypeVar& var) const {
  auto it = type_definitions.find(var);
  ICHECK(it != type_definitions.end()) << "There is no definition of " << var;
  return (*it).second;
}

TypeData IRModuleNode::LookupTypeDef(const String& name) const {
  GlobalTypeVar id = this->GetGlobalTypeVar(name);
  return this->LookupTypeDef(id);
}

Constructor IRModuleNode::LookupTag(const int32_t tag) {
  auto it = constructor_tag_map_.find(tag);
  ICHECK(it != constructor_tag_map_.end()) << "There is no constructor with the tag " << tag;
  return (*it).second;
}

void IRModuleNode::Update(const IRModule& mod) {
  if (const auto* f = runtime::Registry::Get("relay.ir.IRModuleUpdateWithRenamer")) {
    (*f)(GetRef<IRModule>(this), mod);
    return;
  }
  for (auto pair : mod->functions) {
    // TODO(@jroesch): rename into IRModule.
    this->AddUnchecked(pair.first, pair.second);
  }
}

IRModule IRModuleNode::ShallowCopy() {
  return IRModule(this->functions, this->type_definitions, this->Imports(), this->source_map,
                  this->attrs, this->global_infos);
}

std::pair<IRModule, GlobalVar> IRModule::FromExprInContext(
    const RelayExpr& expr, const tvm::Map<GlobalVar, BaseFunc>& global_funcs,
    const tvm::Map<GlobalTypeVar, TypeData>& type_definitions,
    std::unordered_set<String> import_set) {
  auto mod = IRModule(global_funcs, type_definitions, std::move(import_set));
  String gv_name;

  // All global definitions must be functions.
  BaseFunc func;
  if (auto func_node = expr.as<BaseFunc>()) {
    func = func_node.value();
    if (auto opt = func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
      // Function literal has been annotated with it's required global symbol.
      gv_name = opt.value();
    }
  } else if (const auto* f = runtime::Registry::Get("relay.ir.FunctionFromExprInContext")) {
    func = (*f)(expr, mod);
  } else {
    LOG(FATAL) << "`relay.ir.FunctionFromExprInContext` is not registered";
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
  return {mod, main_gv};
}

IRModule IRModule::FromExpr(const RelayExpr& expr, const Map<GlobalVar, BaseFunc>& global_funcs,
                            const Map<GlobalTypeVar, TypeData>& type_definitions) {
  return FromExprInContext(expr, global_funcs, type_definitions).first;
}

void IRModuleNode::Import(const String& path) {
  static const auto* f = runtime::Registry::Get("relay.parser.ParseModule");
  ICHECK(f != nullptr) << "ValueError: Relay parser is not available";
  if (this->import_set_.count(path) == 0) {
    this->import_set_.insert(path);
    std::fstream src_file(path, std::fstream::in);
    std::string file_contents{std::istreambuf_iterator<char>(src_file),
                              std::istreambuf_iterator<char>()};
    auto mod_to_import = (*f)(path, file_contents, GetRef<IRModule>(this));
    Update(mod_to_import);
  }
}

void IRModuleNode::ImportFromStd(const String& path) {
  auto* f = tvm::runtime::Registry::Get("tvm.relay.std_path");
  ICHECK(f != nullptr) << "The Relay std_path is not set, please register tvm.relay.std_path.";
  std::string std_path = (*f)();
  this->Import(std_path + "/" + path);
}

std::unordered_set<String> IRModuleNode::Imports() const { return this->import_set_; }

IRModule IRModule::FromText(const String& text, const String& source_path) {
  static const auto* f = runtime::Registry::Get("relay.parser.ParseModule");
  ICHECK(f != nullptr) << "ValueError: Relay parser is not available";
  return (*f)(source_path, text, Optional<IRModule>());
}

TVM_REGISTER_NODE_TYPE(IRModuleNode);

TVM_REGISTER_GLOBAL("ir.IRModule")
    .set_body_typed([](tvm::Map<GlobalVar, BaseFunc> funcs, tvm::Map<GlobalTypeVar, TypeData> types,
                       tvm::ObjectRef attrs, Map<String, Array<GlobalInfo>> global_infos) {
      auto dict_attrs = [&attrs]() {
        if (!attrs.defined()) {
          return DictAttrs();
        } else if (auto* as_dict_attrs = attrs.as<tvm::DictAttrsNode>()) {
          return GetRef<tvm::DictAttrs>(as_dict_attrs);
        } else if (attrs.as<tvm::MapNode>()) {
          return tvm::DictAttrs(Downcast<Map<String, ObjectRef>>(attrs));
        } else {
          LOG(FATAL) << "Expected attrs argument to be either DictAttrs or Map<String,ObjectRef>";
        }
      }();

      return IRModule(funcs, types, {}, {}, dict_attrs, global_infos);
    });

TVM_REGISTER_GLOBAL("ir.Module_Clone").set_body_typed([](IRModule mod) -> IRModule {
  IRModule clone = mod;
  clone.CopyOnWrite();
  return clone;
});

TVM_REGISTER_GLOBAL("ir.Module_Add")
    .set_body_typed([](IRModule mod, GlobalVar var, ObjectRef val, bool update) -> IRModule {
      ICHECK(val->IsInstance<RelayExprNode>());
      if (const auto* f = runtime::Registry::Get("relay.ir.IRModuleAdd")) {
        return (*f)(mod, var, val, update);
      }
      mod->Add(var, Downcast<BaseFunc>(val), update);
      return mod;
    });

TVM_REGISTER_GLOBAL("ir.Module_Remove")
    .set_body_typed([](IRModule mod, Variant<String, GlobalVar> var) -> IRModule {
      GlobalVar gvar = [&]() {
        if (auto opt = var.as<GlobalVar>()) {
          return opt.value();
        } else if (auto opt = var.as<String>()) {
          return mod->GetGlobalVar(opt.value());
        } else {
          LOG(FATAL) << "InternalError: "
                     << "Variant didn't contain any of the allowed types";
        }
      }();
      mod->Remove(gvar);
      return mod;
    });

TVM_REGISTER_GLOBAL("ir.Module_Contains")
    .set_body_typed([](IRModule mod, Variant<String, GlobalVar> var) -> bool {
      if (auto opt = var.as<GlobalVar>()) {
        return mod->functions.count(opt.value());
      } else if (auto opt = var.as<String>()) {
        return mod->global_var_map_.count(opt.value());
      } else {
        LOG(FATAL) << "InternalError: "
                   << "Variant didn't contain any of the allowed types";
      }
    });

TVM_REGISTER_GLOBAL("ir.Module_AddDef").set_body_method<IRModule>(&IRModuleNode::AddTypeDef);

TVM_REGISTER_GLOBAL("ir.Module_GetGlobalVar")
    .set_body_method<IRModule>(&IRModuleNode::GetGlobalVar);

TVM_REGISTER_GLOBAL("ir.Module_GetGlobalVars")
    .set_body_method<IRModule>(&IRModuleNode::GetGlobalVars);

TVM_REGISTER_GLOBAL("ir.Module_GetGlobalTypeVars")
    .set_body_method<IRModule>(&IRModuleNode::GetGlobalTypeVars);

TVM_REGISTER_GLOBAL("ir.Module_ContainGlobalVar")
    .set_body_method<IRModule>(&IRModuleNode::ContainGlobalVar);

TVM_REGISTER_GLOBAL("ir.Module_ContainGlobalTypeVar")
    .set_body_method<IRModule>(&IRModuleNode::ContainGlobalTypeVar);

TVM_REGISTER_GLOBAL("ir.Module_GetGlobalTypeVar")
    .set_body_method<IRModule>(&IRModuleNode::GetGlobalTypeVar);

TVM_REGISTER_GLOBAL("ir.Module_Lookup").set_body_typed([](IRModule mod, GlobalVar var) {
  return mod->Lookup(var);
});

TVM_REGISTER_GLOBAL("ir.Module_Lookup_str").set_body_typed([](IRModule mod, String var) {
  return mod->Lookup(var);
});

TVM_REGISTER_GLOBAL("ir.Module_LookupDef").set_body_typed([](IRModule mod, GlobalTypeVar var) {
  return mod->LookupTypeDef(var);
});

TVM_REGISTER_GLOBAL("ir.Module_LookupDef_str").set_body_typed([](IRModule mod, String var) {
  return mod->LookupTypeDef(var);
});

TVM_REGISTER_GLOBAL("ir.Module_LookupTag").set_body_typed([](IRModule mod, int32_t tag) {
  return mod->LookupTag(tag);
});

TVM_REGISTER_GLOBAL("ir.Module_FromExpr").set_body_typed(&IRModule::FromExpr);

TVM_REGISTER_GLOBAL("ir.Module_Update").set_body_typed([](IRModule mod, IRModule from) {
  mod->Update(from);
});

TVM_REGISTER_GLOBAL("ir.Module_UpdateFunction")
    .set_body_typed([](IRModule mod, GlobalVar gv, BaseFunc func) { mod->Update(gv, func); });

TVM_REGISTER_GLOBAL("ir.Module_UpdateGlobalInfo")
    .set_body_typed([](IRModule mod, String name, Array<GlobalInfo> global_info) {
      mod->UpdateGlobalInfo(name, global_info);
    });

TVM_REGISTER_GLOBAL("ir.Module_Import").set_body_typed([](IRModule mod, String path) {
  mod->Import(path);
});

TVM_REGISTER_GLOBAL("ir.Module_ImportFromStd").set_body_typed([](IRModule mod, String path) {
  mod->ImportFromStd(path);
});

TVM_REGISTER_GLOBAL("ir.Module_GetAttrs").set_body_typed([](IRModule mod) -> ObjectRef {
  return mod->GetAttrs();
});

TVM_REGISTER_GLOBAL("ir.Module_WithAttr")
    .set_body_typed([](IRModule mod, String key, ObjectRef value) -> IRModule {
      return WithAttr(mod, key, value);
    });

TVM_REGISTER_GLOBAL("ir.Module_WithoutAttr")
    .set_body_typed([](IRModule mod, String key) -> IRModule { return WithoutAttr(mod, key); });

TVM_REGISTER_GLOBAL("ir.Module_WithAttrs")
    .set_body_typed([](IRModule mod, Map<String, ObjectRef> attr_map) -> IRModule {
      return WithAttrs(mod, attr_map);
    });

TVM_REGISTER_GLOBAL("ir.Module_GetAttr").set_body_typed([](IRModule mod, String key) -> ObjectRef {
  return mod->GetAttr<ObjectRef>(key);
});

}  // namespace tvm
