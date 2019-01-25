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

Module ModuleNode::make(tvm::Map<GlobalVar, Function> global_funcs) {
  auto n = make_node<ModuleNode>();
  n->functions = std::move(global_funcs);

  for (const auto& kv : n->functions) {
    // set gloval var map
    CHECK(!n->global_var_map_.count(kv.first->name_hint))
        << "Duplicate global function name " << kv.first->name_hint;
    n->global_var_map_.Set(kv.first->name_hint, kv.first);
  }
  return Module(n);
}

GlobalVar ModuleNode::GetGlobalVar(const std::string& name) {
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
  AddUnchecked(var, checked_func);
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

void ModuleNode::Update(const Module& mod) {
  for (auto pair : mod->functions) {
    this->Update(pair.first, pair.second);
  }
}

TVM_REGISTER_NODE_TYPE(ModuleNode);

TVM_REGISTER_API("relay._make.Module")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = ModuleNode::make(args[0]);
  });

TVM_REGISTER_API("relay._module.Module_Add")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    Module mod = args[0];
    mod->Add(args[1], args[2], args[3]);
  });

TVM_REGISTER_API("relay._module.Module_GetGlobalVar")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    Module mod = args[0];
    *ret = mod->GetGlobalVar(args[1]);
  });

TVM_REGISTER_API("relay._module.Module_Lookup")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    Module mod = args[0];
    GlobalVar var = args[1];
    *ret = mod->Lookup(var);
  });

TVM_REGISTER_API("relay._module.Module_Lookup_str")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    Module mod = args[0];
    std::string var_name = args[1];
    auto var = mod->GetGlobalVar(var_name);
    *ret = mod->Lookup(var);
  });

TVM_REGISTER_API("relay._module.Module_Update")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    Module mod = args[0];
    mod->Update(args[1]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<ModuleNode>(
    [](const ModuleNode *node, tvm::IRPrinter *p) {
      p->stream << "ModuleNode( " << node->functions << ")";
    });

}  // namespace relay
}  // namespace tvm
