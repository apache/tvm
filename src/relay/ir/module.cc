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
#include <tvm/relay/pass.h>
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

  n->entry_func = GlobalVarNode::make("main");

  for (const auto& kv : n->type_definitions) {
    // set global typevar map
    CHECK(!n->global_type_var_map_.count(kv.first->var->name_hint))
      << "Duplicate global type definition name " << kv.first->var->name_hint;
    n->global_type_var_map_.Set(kv.first->var->name_hint, kv.first);
  }

  return Module(n);
}

GlobalVar ModuleNode::GetGlobalVar(const std::string& name) {
  auto it = global_var_map_.find(name);
  if (it == global_var_map_.end()) {
    auto gvar = GlobalVarNode::make(name);
    global_var_map_.Set(name, gvar);
    return gvar;
  } else {
    return (*it).second;
  }
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

GlobalTypeVar ModuleNode::GetGlobalTypeVar(const std::string& name) {
  auto it = global_type_var_map_.find(name);
  CHECK(it != global_type_var_map_.end())
    << "Cannot find global type var " << name << " in the Module";
  return (*it).second;
}

void ModuleNode::Add(const GlobalVar& var,
                     const Function& func,
                     bool update) {
  // Type check the item before we add it to the module.
  auto mod = GetRef<Module>(this);
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

void ModuleNode::AddDef(const GlobalTypeVar& var, const TypeData& type) {
  this->type_definitions.Set(var, type);
  // set global type var map
  CHECK(!global_type_var_map_.count(var->var->name_hint))
    << "Duplicate global type definition name " << var->var->name_hint;
  global_type_var_map_.Set(var->var->name_hint, var);
  for (size_t i = 0; i < type->constructors.size(); ++i) {
    type->constructors[i]->tag = i;
  }

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

Function ModuleNode::Lookup(const GlobalVar& var) {
  auto it = functions.find(var);
  CHECK(it != functions.end())
      << "There is no definition of " << var->name_hint;
  return (*it).second;
}

Function ModuleNode::Lookup(const std::string& name) {
  GlobalVar id = this->GetGlobalVar(name);
  return this->Lookup(id);
}

TypeData ModuleNode::LookupDef(const GlobalTypeVar& var) {
  auto it = type_definitions.find(var);
  CHECK(it != type_definitions.end())
    << "There is no definition of " << var->var->name_hint;
  return (*it).second;
}

TypeData ModuleNode::LookupDef(const std::string& name) {
  GlobalTypeVar id = this->GetGlobalTypeVar(name);
  return this->LookupDef(id);
}

void ModuleNode::Update(const Module& mod) {
  for (auto pair : mod->functions) {
    this->Update(pair.first, pair.second);
  }
}

Module ModuleNode::FromExpr(
  const Expr& expr,
  const tvm::Map<GlobalVar, Function>& global_funcs) {
  auto mod = ModuleNode::make(global_funcs, {});
  auto func_node = expr.as<FunctionNode>();
  Function func;
  if (func_node) {
    func = GetRef<Function>(func_node);
  } else {
    func = FunctionNode::make({}, expr, Type(), {}, {});
  }
  mod->Add(mod->entry_func, func);
  return mod;
}

TVM_REGISTER_NODE_TYPE(ModuleNode);

TVM_REGISTER_API("relay._make.Module")
.set_body_typed(ModuleNode::make);

TVM_REGISTER_API("relay._make.Module_Add")
.set_body_method<Module>(&ModuleNode::Add);

TVM_REGISTER_API("relay._module.Module_AddDef")
.set_body_method<Module>(&ModuleNode::AddDef);

TVM_REGISTER_API("relay._module.Module_GetGlobalVar")
.set_body_method<Module>(&ModuleNode::GetGlobalVar);

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

TVM_REGISTER_API("relay._module.Module_FromExpr")
.set_body_typed<Module(Expr)>([](Expr e) {
    return ModuleNode::FromExpr(e);
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
