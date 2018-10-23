/*!
 *  Copyright (c) 2018 by Contributors
 * \file  environment.cc
 * \brief The global environment in Relay.
 */
#include <tvm/relay/environment.h>
#include <tvm/relay/pass.h>
#include <sstream>

namespace tvm {
namespace relay {

using tvm::IRPrinter;
using namespace runtime;

Environment EnvironmentNode::make(tvm::Map<GlobalVar, Function> global_funcs) {
  auto n = make_node<EnvironmentNode>();
  n->functions = std::move(global_funcs);

  for (const auto& kv : n->functions) {
    // set gloval var map
    CHECK(!n->global_var_map_.count(kv.first->name_hint))
        << "Duplicate global function name " << kv.first->name_hint;
    n->global_var_map_.Set(kv.first->name_hint, kv.first);
  }
  return Environment(n);
}

GlobalVar EnvironmentNode::GetGlobalVar(const std::string& name) {
  auto it = global_var_map_.find(name);
  CHECK(it != global_var_map_.end())
      << "Cannot find global var " << name << " in the Environment";
  return (*it).second;
}

void EnvironmentNode::Add(const GlobalVar& var,
                          const Function& func,
                          bool update) {
  // Type check the item before we add it to the environment.
  auto env = GetRef<Environment>(this);
  Function checked_func = InferType(func, env, var);
  auto type = checked_func->checked_type();
  CHECK(type.as<IncompleteTypeNode>() == nullptr);
  if (functions.find(var) != functions.end()) {
    CHECK(update)
        << "Already have definition for " << var->name_hint;
    auto old_type = functions[var].as<FunctionNode>()->checked_type();
    CHECK(AlphaEqual(type, old_type))
        << "Environment#update changes type, not possible in this mode.";
  }
  this->functions.Set(var, checked_func);
  // set gloval var map
  CHECK(!global_var_map_.count(var->name_hint))
      << "Duplicate global function name " << var->name_hint;
  global_var_map_.Set(var->name_hint, var);
}

void EnvironmentNode::Update(const GlobalVar& var, const Function& func) {
  this->Add(var, func, true);
}

void EnvironmentNode::Remove(const GlobalVar& var) {
  auto functions_node = this->functions.CopyOnWrite();
  functions_node->data.erase(var.node_);
  auto gvar_node = global_var_map_.CopyOnWrite();
  gvar_node->data.erase(var->name_hint);
}

Function EnvironmentNode::Lookup(const GlobalVar& var) {
  auto it = functions.find(var);
  CHECK(it != functions.end())
      << "There is no definition of " << var->name_hint;
  return (*it).second;
}

Function EnvironmentNode::Lookup(const std::string& name) {
  GlobalVar id = this->GetGlobalVar(name);
  return this->Lookup(id);
}

void EnvironmentNode::Update(const Environment& env) {
  for (auto pair : env->functions) {
    this->Update(pair.first, pair.second);
  }
}

TVM_REGISTER_NODE_TYPE(EnvironmentNode);

TVM_REGISTER_API("relay._make.Environment")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = EnvironmentNode::make(args[0]);
  });

TVM_REGISTER_API("relay._env.Environment_Add")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    Environment env = args[0];
    env->Add(args[1], args[2], false);
  });

TVM_REGISTER_API("relay._env.Environment_GetGlobalVar")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    Environment env = args[0];
    *ret = env->GetGlobalVar(args[1]);
  });

TVM_REGISTER_API("relay._env.Environment_Lookup")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    Environment env = args[0];
    GlobalVar var = args[1];
    *ret = env->Lookup(var);
  });

TVM_REGISTER_API("relay._env.Environment_Lookup_str")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    Environment env = args[0];
    std::string var_name = args[1];
    auto var = env->GetGlobalVar(var_name);
    *ret = env->Lookup(var);
  });

TVM_REGISTER_API("relay._env.Environment_Update")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    Environment env = args[0];
    env->Update(args[1]);
  });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<EnvironmentNode>(
    [](const EnvironmentNode *node, tvm::IRPrinter *p) {
      p->stream << "EnvironmentNode( " << node->functions << ")";
    });

}  // namespace relay
}  // namespace tvm
