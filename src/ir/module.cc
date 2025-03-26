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
#include <tvm/runtime/container/variant.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace tvm {

IRModule::IRModule(tvm::Map<GlobalVar, BaseFunc> functions, SourceMap source_map, DictAttrs attrs,
                   Map<String, Array<GlobalInfo>> global_infos) {
  auto n = make_object<IRModuleNode>();
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
    if (functions.size() != other->functions.size()) {
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

  // Checking functions and type definitions
  if (!equal(this->functions, other->functions,
             [](const auto& path) { return path->Attr("functions"); })) {
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

  hash_reduce(this->attrs);
  hash_reduce(this->global_infos);
}

bool IRModuleNode::ContainGlobalVar(const String& name) const {
  return global_var_map_.find(name) != global_var_map_.end();
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
  std::sort(global_vars.begin(), global_vars.end(), [](const GlobalVar& lhs, const GlobalVar& rhs) {
    return lhs->name_hint < rhs->name_hint;
  });
  return tvm::Array<GlobalVar>(global_vars);
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
                            const tvm::Map<GlobalVar, BaseFunc>& global_funcs) {
  auto mod = IRModule(global_funcs);
  String gv_name;

  // All global definitions must be functions.
  BaseFunc func;
  if (auto func_node = expr.as<BaseFunc>()) {
    func = func_node.value();
    if (auto opt = func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
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

TVM_REGISTER_NODE_TYPE(IRModuleNode);

TVM_REGISTER_GLOBAL("ir.IRModule")
    .set_body_typed([](tvm::Map<GlobalVar, BaseFunc> funcs, tvm::ObjectRef attrs,
                       Map<String, Array<GlobalInfo>> global_infos) {
      auto dict_attrs = [&attrs]() {
        if (!attrs.defined()) {
          return DictAttrs();
        } else if (auto* as_dict_attrs = attrs.as<tvm::DictAttrsNode>()) {
          return GetRef<tvm::DictAttrs>(as_dict_attrs);
        } else if (attrs.as<tvm::MapNode>()) {
          return tvm::DictAttrs(Downcast<Map<String, Any>>(attrs));
        } else {
          LOG(FATAL) << "Expected attrs argument to be either DictAttrs or Map<String,ObjectRef>";
        }
      }();

      return IRModule(funcs, {}, dict_attrs, global_infos);
    });

TVM_REGISTER_GLOBAL("ir.Module_Clone").set_body_typed([](IRModule mod) -> IRModule {
  IRModule clone = mod;
  clone.CopyOnWrite();
  return clone;
});

TVM_REGISTER_GLOBAL("ir.Module_Add")
    .set_body_typed([](IRModule mod, GlobalVar var, ObjectRef val, bool update) -> IRModule {
      ICHECK(val->IsInstance<RelaxExprNode>());
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

TVM_REGISTER_GLOBAL("ir.Module_GetGlobalVar")
    .set_body_method<IRModule>(&IRModuleNode::GetGlobalVar);

TVM_REGISTER_GLOBAL("ir.Module_GetGlobalVars")
    .set_body_method<IRModule>(&IRModuleNode::GetGlobalVars);

TVM_REGISTER_GLOBAL("ir.Module_ContainGlobalVar")
    .set_body_method<IRModule>(&IRModuleNode::ContainGlobalVar);

TVM_REGISTER_GLOBAL("ir.Module_Lookup").set_body_typed([](IRModule mod, GlobalVar var) {
  return mod->Lookup(var);
});

TVM_REGISTER_GLOBAL("ir.Module_Lookup_str").set_body_typed([](IRModule mod, String var) {
  return mod->Lookup(var);
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
    .set_body_typed([](IRModule mod, Map<String, ffi::Any> attr_map) -> IRModule {
      return WithAttrs(mod, attr_map);
    });

TVM_REGISTER_GLOBAL("ir.Module_GetAttr").set_body_typed([](IRModule mod, String key) -> ObjectRef {
  return mod->GetAttr<ObjectRef>(key);
});

}  // namespace tvm
