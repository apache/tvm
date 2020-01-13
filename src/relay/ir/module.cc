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
 * \brief The global module in Relay.
 */
#include <tvm/relay/module.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <sstream>
#include <fstream>
#include <unordered_set>

namespace tvm {
namespace relay {

using tvm::NodePrinter;
using namespace runtime;

Module ModuleNode::make(tvm::Map<GlobalVar, BaseFunc> global_funcs,
                        tvm::Map<GlobalTypeVar, TypeData> global_type_defs,
                        std::unordered_set<std::string> imports) {
  auto n = make_object<ModuleNode>();
  n->functions = std::move(global_funcs);
  n->type_definitions = std::move(global_type_defs);
  n->global_type_var_map_ = {};
  n->global_var_map_ = {};
  n->constructor_tag_map_ = {};
  n->import_set_ = imports;

  for (const auto& kv : n->functions) {
    // set global var map
    CHECK(n->global_var_map_.count(kv.first->name_hint) == 0)
      << "Duplicate global function name " << kv.first->name_hint;
    n->global_var_map_.Set(kv.first->name_hint, kv.first);
  }

  for (const auto& kv : n->type_definitions) {
    // set global typevar map
    CHECK(n->global_type_var_map_.count(kv.first->name_hint) == 0)
      << "Duplicate global type definition name " << kv.first->name_hint;
    n->global_type_var_map_.Set(kv.first->name_hint, kv.first);
    n->RegisterConstructors(kv.first, kv.second);
  }

  return Module(n);
}

bool ModuleNode::ContainGlobalVar(const std::string& name) const {
  return global_var_map_.find(name) != global_var_map_.end();
}

bool ModuleNode::ContainGlobalTypeVar(const std::string& name) const {
  return global_type_var_map_.find(name) != global_type_var_map_.end();
}

GlobalVar ModuleNode::GetGlobalVar(const std::string& name) const {
  auto it = global_var_map_.find(name);
  CHECK(it != global_var_map_.end())
    << "Cannot find global var " << name << " in the Module";
  return (*it).second;
}

tvm::Array<GlobalVar> ModuleNode::GetGlobalVars() const {
  std::vector<GlobalVar> global_vars;
  for (const auto& pair : global_var_map_) {
    global_vars.push_back(pair.second);
  }
  return tvm::Array<GlobalVar>(global_vars);
}

GlobalTypeVar ModuleNode::GetGlobalTypeVar(const std::string& name) const {
  CHECK(global_type_var_map_.defined());
  auto it = global_type_var_map_.find(name);
  CHECK(it != global_type_var_map_.end())
    << "Cannot find global type var " << name << " in the Module";
  return (*it).second;
}

tvm::Array<GlobalTypeVar> ModuleNode::GetGlobalTypeVars() const {
  std::vector<GlobalTypeVar> global_type_vars;
  for (const auto& pair : global_type_var_map_) {
    global_type_vars.push_back(pair.second);
  }
  return tvm::Array<GlobalTypeVar>(global_type_vars);
}

template<typename T>
tvm::Array<T> concat(const tvm::Array<T>& l, const tvm::Array<T>& r) {
  tvm::Array<T> ret(l);
  for (const T& t : r) {
    ret.push_back(t);
  }
  return ret;
}

// helper function to run type check
relay::Function RunTypeCheck(const Module& mod,
                             const GlobalVar& var,
                             relay::Function f) {
  auto func = Downcast<relay::Function>(relay::DeDup(std::move(f)));
  // Type check the item before we add it to the module.
  auto fv = relay::FreeVars(func);
  auto ftv = relay::FreeTypeVars(func, mod);
  if (fv.size() != 0) {
    LOG(WARNING)
        << "There are free variables: "
        << fv
        << " in function: "
        << AsText(func, false)
        << std::endl;
  }
  if (ftv.size() != 0) {
    LOG(WARNING)
        << "There are free type variables: "
        << ftv
        << " in function: "
        << AsText(func, false)
        << std::endl;
  }
  func =
      relay::FunctionNode::make(concat(func->params, fv),
                                func->body,
                                func->ret_type,
                                concat(func->type_params, ftv),
                                func->attrs);
  // Type check the item before we add it to the module.
  relay::Function checked_func = InferType(func, mod, var);
  return checked_func;
}

void ModuleNode::Add(const GlobalVar& var,
                     const BaseFunc& f,
                     bool update) {
  BaseFunc checked_func = f;
  if (auto* ptr = f.as<relay::FunctionNode>()) {
    checked_func = RunTypeCheck(GetRef<Module>(this),
                                var,
                                GetRef<relay::Function>(ptr));
  }

  auto type = checked_func->checked_type();
  CHECK(type.as<relay::IncompleteTypeNode>() == nullptr);

  if (functions.find(var) != functions.end()) {
    CHECK(update)
        << "Already have definition for " << var->name_hint;
    auto old_type = functions[var].as<FunctionNode>()->checked_type();
    CHECK(AlphaEqual(type, old_type))
        << "Module#update changes type, not possible in this mode.";
  }
  var->checked_type_ = type;
  AddUnchecked(var, checked_func);
}

void ModuleNode::AddUnchecked(const GlobalVar& var,
                              const BaseFunc& func) {
  this->functions.Set(var, func);

  auto it = global_var_map_.find(var->name_hint);
  if (it != global_var_map_.end()) {
    CHECK_EQ((*it).second, var);
  } else {
    CHECK(global_var_map_.count(var->name_hint) == 0)
        << "Duplicate global function name " << var->name_hint;
  }

  global_var_map_.Set(var->name_hint, var);
}

void ModuleNode::RegisterConstructors(const GlobalTypeVar& var, const TypeData& type) {
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

void ModuleNode::AddTypeDef(const GlobalTypeVar& var,
                            const TypeData& type,
                            bool update) {
  AddTypeDefUnchecked(var, type, update);
  // need to kind check at the end because the check can look up
  // a definition potentially
  CHECK(relay::KindCheck(type, GetRef<Module>(this)) == Kind::kTypeData)
    << "Invalid or malformed typedata given to module: " << type;
}

void ModuleNode::AddTypeDefUnchecked(const GlobalTypeVar& var,
                                     const TypeData& type,
                                     bool update) {
  this->type_definitions.Set(var, type);
  if (!update) {
    // set global type var map
    CHECK(global_type_var_map_.count(var->name_hint) == 0)
      << "Duplicate global type definition name " << var->name_hint;
  }
  global_type_var_map_.Set(var->name_hint, var);
  RegisterConstructors(var, type);
}

void ModuleNode::Update(const GlobalVar& var,
                        const BaseFunc& func) {
  this->Add(var, func, true);
}

void ModuleNode::UpdateTypeDef(const GlobalTypeVar& var,
                               const TypeData& type) {
  this->AddTypeDef(var, type, true);
}

void ModuleNode::Remove(const GlobalVar& var) {
  auto functions_node = this->functions.CopyOnWrite();
  functions_node->data.erase(var);
  auto gvar_node = global_var_map_.CopyOnWrite();
  gvar_node->data.erase(var->name_hint);
}

BaseFunc ModuleNode::Lookup(const GlobalVar& var) const {
  auto it = functions.find(var);
  CHECK(it != functions.end())
      << "There is no definition of " << var->name_hint;
  return (*it).second;
}

BaseFunc ModuleNode::Lookup(const std::string& name) const {
  GlobalVar id = this->GetGlobalVar(name);
  return this->Lookup(id);
}

TypeData ModuleNode::LookupTypeDef(const GlobalTypeVar& var) const {
  auto it = type_definitions.find(var);
  CHECK(it != type_definitions.end())
    << "There is no definition of " << var->name_hint;
  return (*it).second;
}

TypeData ModuleNode::LookupTypeDef(const std::string& name) const {
  GlobalTypeVar id = this->GetGlobalTypeVar(name);
  return this->LookupTypeDef(id);
}

Constructor ModuleNode::LookupTag(const int32_t tag) {
  auto it = constructor_tag_map_.find(tag);
  CHECK(it != constructor_tag_map_.end())
    << "There is no constructor with the tag " << tag;
  return (*it).second;
}

void ModuleNode::Update(const Module& mod) {
  // add functions and type defs. we add them unchecked first, so all definitions
  // can reference each other, independent of the order in which they were defined.
  for (auto pair : mod->functions) {
    this->AddUnchecked(pair.first, pair.second);
  }
  for (auto pair : mod->type_definitions) {
    this->AddTypeDefUnchecked(pair.first, pair.second);
  }
  for (auto pair : mod->functions) {
    this->Update(pair.first, pair.second);
  }
  for (auto pair : mod->type_definitions) {
    this->UpdateTypeDef(pair.first, pair.second);
  }
}

Module ModuleNode::FromExpr(
  const RelayExpr& expr,
  const tvm::Map<GlobalVar, BaseFunc>& global_funcs,
  const tvm::Map<GlobalTypeVar, TypeData>& type_definitions) {
  auto mod = ModuleNode::make(global_funcs, type_definitions);
  BaseFunc func;
  if (auto* func_node = expr.as<relay::FunctionNode>()) {
    func = GetRef<relay::Function>(func_node);
  } else {
    func = relay::FunctionNode::make(
        relay::FreeVars(expr), expr, Type(),
        relay::FreeTypeVars(expr, mod), {});
  }
  auto main_gv = GlobalVar("main");
  mod->Add(main_gv, func);
  return mod;
}

void ModuleNode::Import(const std::string& path) {
  if (this->import_set_.count(path) == 0) {
    this->import_set_.insert(path);
    DLOG(INFO) << "Importing: " << path;
    std::fstream src_file(path, std::fstream::in);
    std::string file_contents {
      std::istreambuf_iterator<char>(src_file),
      std::istreambuf_iterator<char>() };
    auto mod_to_import = FromText(file_contents, path);
    Update(mod_to_import);
  }
}

void ModuleNode::ImportFromStd(const std::string& path) {
  auto* f = tvm::runtime::Registry::Get("tvm.relay.std_path");
  CHECK(f != nullptr) << "The Relay std_path is not set, please register tvm.relay.std_path.";
  std::string std_path = (*f)();
  return this->Import(std_path + "/" + path);
}

std::unordered_set<std::string> ModuleNode::Imports() const {
  return this->import_set_;
}

Module FromText(const std::string& source, const std::string& source_name) {
  auto* f = tvm::runtime::Registry::Get("relay.fromtext");
  CHECK(f != nullptr) << "The Relay std_path is not set, please register tvm.relay.std_path.";
  Module mod = (*f)(source, source_name);
  return mod;
}

TVM_REGISTER_NODE_TYPE(ModuleNode);

TVM_REGISTER_GLOBAL("relay._make.Module")
.set_body_typed([](tvm::Map<GlobalVar, BaseFunc> funcs,
                   tvm::Map<GlobalTypeVar, TypeData> types) {
  return ModuleNode::make(funcs, types, {});
});

TVM_REGISTER_GLOBAL("relay._module.Module_Add")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  Module mod = args[0];
  GlobalVar var = args[1];
  ObjectRef val = args[2];
  bool update = args[3];
  CHECK(val->IsInstance<ExprNode>());

  if (val->IsInstance<relay::FunctionNode>()) {
    mod->Add(var, Downcast<relay::Function>(val), update);
  } else if (val->IsInstance<GlobalVarNode>()) {
    GlobalVar gv = Downcast<GlobalVar>(val);
    auto mod_copy = Module(make_object<ModuleNode>(*mod.operator->()));
    mod_copy = relay::transform::EtaExpand(
        /* expand_constructor */ false,
        /* expand_global_var */ true)(mod_copy);
    auto func = mod_copy->Lookup(gv->name_hint);
    mod->Add(var, Downcast<relay::Function>(func), update);
  } else {
    auto func = FunctionNode::make({}, Downcast<relay::Expr>(val), Type(nullptr), {});
    mod->Add(var, func, update);
  }
  *ret = mod;
});

TVM_REGISTER_GLOBAL("relay._module.Module_AddDef")
.set_body_method<Module>(&ModuleNode::AddTypeDef);

TVM_REGISTER_GLOBAL("relay._module.Module_GetGlobalVar")
.set_body_method<Module>(&ModuleNode::GetGlobalVar);

TVM_REGISTER_GLOBAL("relay._module.Module_GetGlobalVars")
.set_body_method<Module>(&ModuleNode::GetGlobalVars);

TVM_REGISTER_GLOBAL("relay._module.Module_GetGlobalTypeVars")
.set_body_method<Module>(&ModuleNode::GetGlobalTypeVars);

TVM_REGISTER_GLOBAL("relay._module.Module_ContainGlobalVar")
.set_body_method<Module>(&ModuleNode::ContainGlobalVar);

TVM_REGISTER_GLOBAL("relay._module.Module_GetGlobalTypeVar")
.set_body_method<Module>(&ModuleNode::GetGlobalTypeVar);

TVM_REGISTER_GLOBAL("relay._module.Module_Lookup")
.set_body_typed([](Module mod, GlobalVar var) {
  return mod->Lookup(var);
});

TVM_REGISTER_GLOBAL("relay._module.Module_Lookup_str")
.set_body_typed([](Module mod, std::string var) {
  return mod->Lookup(var);
});

TVM_REGISTER_GLOBAL("relay._module.Module_LookupDef")
.set_body_typed([](Module mod, GlobalTypeVar var) {
  return mod->LookupTypeDef(var);
});

TVM_REGISTER_GLOBAL("relay._module.Module_LookupDef_str")
.set_body_typed([](Module mod, std::string var) {
  return mod->LookupTypeDef(var);
});

TVM_REGISTER_GLOBAL("relay._module.Module_LookupTag")
.set_body_typed([](Module mod, int32_t tag) {
    return mod->LookupTag(tag);
  });

TVM_REGISTER_GLOBAL("relay._module.Module_FromExpr")
.set_body_typed([](RelayExpr e,
                   tvm::Map<GlobalVar, BaseFunc> funcs,
                   tvm::Map<GlobalTypeVar, TypeData> type_defs) {
  return ModuleNode::FromExpr(e, funcs, type_defs);
});

TVM_REGISTER_GLOBAL("relay._module.Module_Update")
.set_body_typed([](Module mod, Module from) {
  mod->Update(from);
});

TVM_REGISTER_GLOBAL("relay._module.Module_Import")
.set_body_typed([](Module mod, std::string path) {
  mod->Import(path);
});

TVM_REGISTER_GLOBAL("relay._module.Module_ImportFromStd")
.set_body_typed([](Module mod, std::string path) {
  mod->ImportFromStd(path);
});;

TVM_STATIC_IR_FUNCTOR(NodePrinter, vtable)
.set_dispatch<ModuleNode>([](const ObjectRef& ref, NodePrinter* p) {
    auto* node = static_cast<const ModuleNode*>(ref.get());
    p->stream << "ModuleNode( " << node->functions << ")";
});

}  // namespace relay
}  // namespace tvm
