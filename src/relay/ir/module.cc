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
 *  Copyright (c) 2018 by Contributors
 * \file  module.cc
 * \brief The global module in Relay.
 */
#include <tvm/relay/module.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <sstream>

namespace tvm {
namespace relay {

using tvm::IRPrinter;
using namespace runtime;

Module ModuleNode::make(tvm::Map<GlobalVar, Function> global_funcs,
                        tvm::Map<GlobalTypeVar, TypeData> global_type_defs) {
  auto n = make_node<ModuleNode>();
  n->functions = std::move(global_funcs);
  n->type_definitions = std::move(global_type_defs);

  for (const auto& kv : n->functions) {
    // set global var map
    CHECK(!n->global_var_map_.count(kv.first->name_hint))
      << "Duplicate global function name " << kv.first->name_hint;
    n->global_var_map_.Set(kv.first->name_hint, kv.first);
  }

  for (const auto& kv : n->type_definitions) {
    // set global typevar map
    CHECK(!n->global_type_var_map_.count(kv.first->var->name_hint))
      << "Duplicate global type definition name " << kv.first->var->name_hint;
    n->global_type_var_map_.Set(kv.first->var->name_hint, kv.first);
    n->RegisterConstructors(kv.first, kv.second);
  }

  return Module(n);
}

bool ModuleNode::ContainGlobalVar(const std::string& name) const {
  return global_var_map_.find(name) != global_var_map_.end();
}

GlobalVar ModuleNode::GetGlobalVar(const std::string& name) const {
  auto it = global_var_map_.find(name);
  CHECK(it != global_var_map_.end())
    << "Cannot find global var " << name << " in the Module";
  return (*it).second;
}

void ModuleNode::AddUnchecked(const GlobalVar& var,
                              const Function& func) {
  auto mod = GetRef<Module>(this);
  this->functions.Set(var, func);

  auto it = global_var_map_.find(var->name_hint);
  if (it != global_var_map_.end()) {
    CHECK_EQ((*it).second, var);
  } else {
    CHECK(!global_var_map_.count(var->name_hint))
        << "Duplicate global function name " << var->name_hint;
  }

  global_var_map_.Set(var->name_hint, var);
}

GlobalTypeVar ModuleNode::GetGlobalTypeVar(const std::string& name) const {
  auto it = global_type_var_map_.find(name);
  CHECK(it != global_type_var_map_.end())
    << "Cannot find global type var " << name << " in the Module";
  return (*it).second;
}

template<typename T>
tvm::Array<T> concat(const tvm::Array<T>& l, const tvm::Array<T>& r) {
  tvm::Array<T> ret(l);
  for (const T& t : r) {
    ret.push_back(t);
  }
  return ret;
}

void ModuleNode::Add(const GlobalVar& var,
                     const Function& f,
                     bool update) {
  Function func = Downcast<Function>(DeDup(f));
  // Type check the item before we add it to the module.
  auto mod = GetRef<Module>(this);
  auto fv = FreeVars(func);
  auto ftv = FreeTypeVars(func, mod);
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
    FunctionNode::make(concat(func->params, fv),
                       func->body,
                       func->ret_type,
                       concat(func->type_params, ftv),
                       func->attrs);
  // Type check the item before we add it to the module.
  Function checked_func = InferType(func, mod, var);
  auto type = checked_func->checked_type();
  CHECK(type.as<IncompleteTypeNode>() == nullptr);
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

void ModuleNode::RegisterConstructors(const GlobalTypeVar& var, const TypeData& type) {
  // We hash the global type var name to use as a globally unique prefix for tags.
  // The hash will be used as the most significant byte of the tag, with the index of
  // the constructor in the less significant bytes
  size_t hash = std::hash<std::string>()(var->var->name_hint);
  int32_t prefix = static_cast<int32_t>(hash & 0xff) << 24;
  for (size_t i = 0; i < type->constructors.size(); ++i) {
    type->constructors[i]->tag = prefix | static_cast<int32_t>(i);
    constructor_tag_map_[type->constructors[i]->tag] = type->constructors[i];
  }
}

void ModuleNode::AddDef(const GlobalTypeVar& var, const TypeData& type) {
  this->type_definitions.Set(var, type);
  // set global type var map
  CHECK(!global_type_var_map_.count(var->var->name_hint))
    << "Duplicate global type definition name " << var->var->name_hint;
  global_type_var_map_.Set(var->var->name_hint, var);
  RegisterConstructors(var, type);

  // need to kind check at the end because the check can look up
  // a definition potentially
  CHECK(KindCheck(type, GetRef<Module>(this)) == Kind::kTypeData)
    << "Invalid or malformed typedata given to module: " << type;
}

void ModuleNode::Update(const GlobalVar& var, const Function& func) {
  this->Add(var, func, true);
}

void ModuleNode::Remove(const GlobalVar& var) {
  auto functions_node = this->functions.CopyOnWrite();
  functions_node->data.erase(var.node_);
  auto gvar_node = global_var_map_.CopyOnWrite();
  gvar_node->data.erase(var->name_hint);
}

Function ModuleNode::Lookup(const GlobalVar& var) const {
  auto it = functions.find(var);
  CHECK(it != functions.end())
      << "There is no definition of " << var->name_hint;
  return (*it).second;
}

Function ModuleNode::Lookup(const std::string& name) const {
  GlobalVar id = this->GetGlobalVar(name);
  return this->Lookup(id);
}

TypeData ModuleNode::LookupDef(const GlobalTypeVar& var) const {
  auto it = type_definitions.find(var);
  CHECK(it != type_definitions.end())
    << "There is no definition of " << var->var->name_hint;
  return (*it).second;
}

TypeData ModuleNode::LookupDef(const std::string& name) const {
  GlobalTypeVar id = this->GetGlobalTypeVar(name);
  return this->LookupDef(id);
}

Constructor ModuleNode::LookupTag(const int32_t tag) {
  auto it = constructor_tag_map_.find(tag);
  CHECK(it != constructor_tag_map_.end())
    << "There is no constructor with the tag " << tag;
  return (*it).second;
}

void ModuleNode::Update(const Module& mod) {
  for (auto pair : mod->functions) {
    this->Update(pair.first, pair.second);
  }
}

Module ModuleNode::FromExpr(
  const Expr& expr,
  const tvm::Map<GlobalVar, Function>& global_funcs,
  const tvm::Map<GlobalTypeVar, TypeData>& type_definitions) {
  auto mod = ModuleNode::make(global_funcs, type_definitions);
  auto func_node = expr.as<FunctionNode>();
  Function func;
  if (func_node) {
    func = GetRef<Function>(func_node);
  } else {
    func = FunctionNode::make(FreeVars(expr), expr, Type(), FreeTypeVars(expr, mod), {});
  }
  auto main_gv = GlobalVarNode::make("main");
  mod->Add(main_gv, func);
  return mod;
}

TVM_REGISTER_NODE_TYPE(ModuleNode);

TVM_REGISTER_API("relay._make.Module")
.set_body_typed(ModuleNode::make);

TVM_REGISTER_API("relay._module.Module_Add")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  Module mod = args[0];
  GlobalVar var = args[1];
  NodeRef val = args[2];
  bool update = args[3];
  CHECK(val->derived_from<ExprNode>());
  if (val->derived_from<FunctionNode>()) {
    mod->Add(var, Downcast<Function>(val), update);
  } else if (val->derived_from<GlobalVarNode>()) {
    GlobalVar gv = Downcast<GlobalVar>(val);
    auto mod_copy = Module(make_node<ModuleNode>(*mod.operator->()));
    mod_copy = transform::EtaExpand()(mod_copy);
    auto func = mod_copy->Lookup(gv->name_hint);
    mod->Add(var, Downcast<Function>(func), update);
  } else {
    auto func = FunctionNode::make({}, Downcast<Expr>(val), Type(nullptr), {});
    mod->Add(var, func, update);
  }
  *ret = mod;
});

TVM_REGISTER_API("relay._module.Module_AddDef")
.set_body_method<Module>(&ModuleNode::AddDef);

TVM_REGISTER_API("relay._module.Module_GetGlobalVar")
.set_body_method<Module>(&ModuleNode::GetGlobalVar);

TVM_REGISTER_API("relay._module.Module_ContainGlobalVar")
.set_body_method<Module>(&ModuleNode::ContainGlobalVar);

TVM_REGISTER_API("relay._module.Module_GetGlobalTypeVar")
.set_body_method<Module>(&ModuleNode::GetGlobalTypeVar);

TVM_REGISTER_API("relay._module.Module_Lookup")
.set_body_typed<Function(Module, GlobalVar)>([](Module mod, GlobalVar var) {
  return mod->Lookup(var);
});

TVM_REGISTER_API("relay._module.Module_Lookup_str")
.set_body_typed<Function(Module, std::string)>([](Module mod, std::string var) {
  return mod->Lookup(var);
});

TVM_REGISTER_API("relay._module.Module_LookupDef")
.set_body_typed<TypeData(Module, GlobalTypeVar)>([](Module mod, GlobalTypeVar var) {
  return mod->LookupDef(var);
});

TVM_REGISTER_API("relay._module.Module_LookupDef_str")
.set_body_typed<TypeData(Module, std::string)>([](Module mod, std::string var) {
  return mod->LookupDef(var);
});

TVM_REGISTER_API("relay._module.Module_LookupTag")
.set_body_typed<Constructor(Module, int32_t)>([](Module mod, int32_t tag) {
    return mod->LookupTag(tag);
  });

TVM_REGISTER_API("relay._module.Module_FromExpr")
.set_body_typed<
  Module(Expr,
         tvm::Map<GlobalVar, Function>,
         tvm::Map<GlobalTypeVar, TypeData>)>([](Expr e,
                                                tvm::Map<GlobalVar, Function> funcs,
                                                tvm::Map<GlobalTypeVar, TypeData> type_defs) {
                                               return ModuleNode::FromExpr(e, funcs, type_defs);
                                             });

TVM_REGISTER_API("relay._module.Module_Update")
.set_body_typed<void(Module, Module)>([](Module mod, Module from) {
  mod->Update(from);
});

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<ModuleNode>(
  [](const ModuleNode *node, tvm::IRPrinter *p) {
    p->stream << "ModuleNode( " << node->functions << ")";
});

}  // namespace relay
}  // namespace tvm
