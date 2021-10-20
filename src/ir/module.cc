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
#include <tvm/ir/module.h>
#include <tvm/node/structural_equal.h>
#include <tvm/runtime/registry.h>
// NOTE: reverse dependency on relay.
// These dependencies do not happen at the interface-level,
// and are only used in minimum cases where they are clearly marked.
//
// Rationale: We calls into relay's analysis module to verify correctness.
#include <tvm/ir/type_functor.h>
#include <tvm/parser/parser.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <fstream>
#include <sstream>
#include <unordered_set>

namespace tvm {

IRModule::IRModule(tvm::Map<GlobalVar, BaseFunc> functions,
                   tvm::Map<GlobalTypeVar, TypeData> type_definitions,
                   std::unordered_set<String> import_set, parser::SourceMap source_map,
                   DictAttrs attrs) {
  auto n = make_object<IRModuleNode>();
  n->functions = std::move(functions);
  n->type_definitions = std::move(type_definitions);
  n->global_type_var_map_ = {};
  n->global_var_map_ = {};
  n->constructor_tag_map_ = {};
  n->import_set_ = std::move(import_set);
  n->source_map = source_map;
  n->attrs = std::move(attrs);

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
  if (functions.size() != other->functions.size()) return false;
  if (!equal(this->attrs, other->attrs)) return false;
  for (const auto& kv : this->functions) {
    if (!other->ContainGlobalVar(kv.first->name_hint)) return false;
    if (!equal(kv.second, other->Lookup(kv.first->name_hint))) return false;
  }
  if (type_definitions.size() != other->type_definitions.size()) return false;
  for (const auto& kv : this->type_definitions) {
    if (!other->ContainGlobalTypeVar(kv.first->name_hint)) return false;
    if (!equal(kv.second, other->LookupTypeDef(kv.first->name_hint))) return false;
  }
  return true;
}

void IRModuleNode::SHashReduce(SHashReducer hash_reduce) const {
  using KV = std::pair<std::string, ObjectRef>;
  // hash the functions.
  std::vector<KV> temp;

  auto reduce_temp = [&]() {
    // sort by the hash key of the keys.
    std::sort(temp.begin(), temp.end(),
              [](const KV& lhs, const KV& rhs) { return lhs.first < rhs.first; });

    hash_reduce(static_cast<uint64_t>(temp.size()));
    // hash the content
    for (size_t i = 0; i < temp.size(); ++i) {
      hash_reduce(temp[i].first);
      hash_reduce(temp[i].second);
    }
  };

  for (const auto& kv : this->functions) {
    temp.emplace_back(kv.first->name_hint, kv.second);
  }
  reduce_temp();

  temp.clear();
  for (const auto& kv : this->type_definitions) {
    temp.emplace_back(kv.first->name_hint, kv.second);
  }
  reduce_temp();
  hash_reduce(this->attrs);
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
  throw std::runtime_error("Constructor Not Found.");
}

tvm::Array<GlobalTypeVar> IRModuleNode::GetGlobalTypeVars() const {
  std::vector<GlobalTypeVar> global_type_vars;
  for (const auto& pair : global_type_var_map_) {
    global_type_vars.push_back(pair.second);
  }
  return tvm::Array<GlobalTypeVar>(global_type_vars);
}

void WarnIfMalformed(const IRModule& mod, relay::Function func) {
  func = Downcast<relay::Function>(relay::DeDup(func));
  // Type check the item before we add it to the module.
  auto fv = relay::FreeVars(func);
  auto ftv = relay::FreeTypeVars(func, mod);
  // TODO(@jroesch): refactor to use diagnostic context
  ICHECK_EQ(fv.size(), 0) << "There are free variables: " << fv << std::endl;
  ICHECK_EQ(ftv.size(), 0) << "There are free type variables: " << fv
                           << " in function: " << AsText(func, false);
}

void IRModuleNode::Add(const GlobalVar& var, const BaseFunc& f, bool update) {
  BaseFunc checked_func = f;
  if (auto* ptr = f.as<relay::FunctionNode>()) {
    WarnIfMalformed(GetRef<IRModule>(this), GetRef<relay::Function>(ptr));
  }

  AddUnchecked(var, checked_func);
}

void IRModuleNode::AddUnchecked(const GlobalVar& var, const BaseFunc& func) {
  this->functions.Set(var, func);

  auto it = global_var_map_.find(var->name_hint);
  if (it != global_var_map_.end()) {
    ICHECK_EQ((*it).second, var);
  } else {
    ICHECK(global_var_map_.count(var->name_hint) == 0)
        << "Duplicate global function name " << var->name_hint;
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
        << "Duplicate global type definition name " << var->name_hint;
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

void IRModuleNode::Remove(const GlobalVar& var) {
  auto functions_node = this->functions.CopyOnWrite();
  functions_node->erase(var);
  auto gvar_node = global_var_map_.CopyOnWrite();
  gvar_node->erase(var->name_hint);
}

BaseFunc IRModuleNode::Lookup(const GlobalVar& var) const {
  auto it = functions.find(var);
  ICHECK(it != functions.end()) << "There is no definition of " << var->name_hint;
  return (*it).second;
}

BaseFunc IRModuleNode::Lookup(const String& name) const {
  GlobalVar id = this->GetGlobalVar(name);
  return this->Lookup(id);
}

TypeData IRModuleNode::LookupTypeDef(const GlobalTypeVar& var) const {
  auto it = type_definitions.find(var);
  ICHECK(it != type_definitions.end()) << "There is no definition of " << var->name_hint;
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

String IRModuleNode::GetUniqueName(const String& name) {
  String result = name;
  int suffix = 0;
  while (true) {
    auto it = global_var_map_.find(result);
    if (it == global_var_map_.end()) {
      return result;
    }
    std::ostringstream os;
    os << name << "_" << ++suffix;
    result = os.str();
  }
}

struct Renamer : relay::ExprMutator, TypeMutator {
  Map<String, GlobalVar> defs;
  Map<String, GlobalTypeVar> types;
  std::unordered_map<int32_t, Constructor> ctors;

  Renamer(Map<String, GlobalVar> defs_one, Map<String, GlobalVar> defs_two,
          Map<String, GlobalTypeVar> types_one, Map<String, GlobalTypeVar> types_two,
          std::unordered_map<int32_t, Constructor> ctors_one,
          std::unordered_map<int32_t, Constructor> ctor_two) {
    for (auto pair : defs_one) {
      defs.Set(pair.first, pair.second);
    }

    for (auto pair : defs_two) {
      auto it = defs.find(pair.first);
      if (it == defs.end()) {
        defs.Set(pair.first, pair.second);
      }
    }

    for (auto pair : types_one) {
      types.Set(pair.first, pair.second);
    }

    for (auto pair : types_two) {
      auto it = types.find(pair.first);
      if (it == types.end()) {
        types.Set(pair.first, pair.second);
      }
    }
  }

  relay::Expr VisitExpr_(const GlobalVarNode* node) override { return defs.at(node->name_hint); }

  Type VisitType_(const GlobalTypeVarNode* node) override { return types.at(node->name_hint); }
};

void IRModuleNode::Update(const IRModule& mod) {
  Renamer renamer(this->global_var_map_, mod->global_var_map_, this->global_type_var_map_,
                  mod->global_type_var_map_, this->constructor_tag_map_, mod->constructor_tag_map_);

  this->global_var_map_ = renamer.defs;
  this->global_type_var_map_ = renamer.types;
  this->constructor_tag_map_ = renamer.ctors;

  for (auto pair : mod->type_definitions) {
    auto tvar = renamer.types.at(pair.first->name_hint);
    auto ty = renamer.ExprMutator::VisitType(pair.second);
    this->AddTypeDefUnchecked(tvar, Downcast<TypeData>(ty), true);
  }

  for (auto pair : mod->functions) {
    if (auto rfn = pair.second.as<relay::FunctionNode>()) {
      auto gvar = renamer.defs.at(pair.first->name_hint);
      auto fn = renamer.VisitExpr(GetRef<relay::Function>(rfn));
      this->AddUnchecked(gvar, Downcast<BaseFunc>(fn));
    } else {
      // TODO(@jroesch): rename into IRModule.
      this->AddUnchecked(pair.first, pair.second);
    }
  }
}

IRModule IRModuleNode::ShallowCopy() {
  return IRModule(this->functions, this->type_definitions, this->Imports(), this->source_map,
                  this->attrs);
}

std::pair<IRModule, GlobalVar> IRModule::FromExprInContext(
    const RelayExpr& expr, const tvm::Map<GlobalVar, BaseFunc>& global_funcs,
    const tvm::Map<GlobalTypeVar, TypeData>& type_definitions,
    std::unordered_set<String> import_set) {
  auto mod = IRModule(global_funcs, type_definitions, std::move(import_set));
  String gv_name;

  // All global definitions must be functions.
  BaseFunc func;
  if (auto* func_node = expr.as<BaseFuncNode>()) {
    func = GetRef<BaseFunc>(func_node);
    if (auto opt = func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
      // Function literal has been annotated with it's required global symbol.
      gv_name = opt.value();
    }
  } else {
    func = relay::Function(relay::FreeVars(expr), expr, Type(), relay::FreeTypeVars(expr, mod), {});
  }

  if (gv_name.empty()) {
    // Bind function to 'main' (though rename if would clash with existing 'main').
    gv_name = mod->GetUniqueName("main");
  }

  GlobalVar main_gv(gv_name);
  mod->Add(main_gv, func);
  return {mod, main_gv};
}

IRModule IRModule::FromExpr(const RelayExpr& expr, const Map<GlobalVar, BaseFunc>& global_funcs,
                            const Map<GlobalTypeVar, TypeData>& type_definitions) {
  return FromExprInContext(expr, global_funcs, type_definitions).first;
}

void IRModuleNode::Import(const String& path) {
  if (this->import_set_.count(path) == 0) {
    this->import_set_.insert(path);
    DLOG(INFO) << "Importing: " << path;
    std::fstream src_file(path, std::fstream::in);
    std::string file_contents{std::istreambuf_iterator<char>(src_file),
                              std::istreambuf_iterator<char>()};
    auto mod_to_import = parser::ParseModule(path, file_contents, GetRef<IRModule>(this));
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
  return tvm::parser::ParseModule(source_path, text);
}

TVM_REGISTER_NODE_TYPE(IRModuleNode);

TVM_REGISTER_GLOBAL("ir.IRModule")
    .set_body_typed([](tvm::Map<GlobalVar, BaseFunc> funcs,
                       tvm::Map<GlobalTypeVar, TypeData> types) {
      return IRModule(funcs, types, {});
    });

TVM_REGISTER_GLOBAL("ir.Module_Add").set_body([](TVMArgs args, TVMRetValue* ret) {
  IRModule mod = args[0];
  GlobalVar var = args[1];
  ObjectRef val = args[2];
  bool update = args[3];
  ICHECK(val->IsInstance<RelayExprNode>());

  if (val->IsInstance<BaseFuncNode>()) {
    mod->Add(var, Downcast<BaseFunc>(val), update);
  } else if (val->IsInstance<GlobalVarNode>()) {
    GlobalVar gv = Downcast<GlobalVar>(val);
    auto mod_copy = IRModule(make_object<IRModuleNode>(*mod.operator->()));
    mod_copy = relay::transform::EtaExpand(
        /* expand_constructor */ false,
        /* expand_global_var */ true)(mod_copy);
    auto func = mod_copy->Lookup(gv->name_hint);
    mod->Add(var, Downcast<relay::Function>(func), update);
  } else {
    auto func = relay::Function({}, Downcast<RelayExpr>(val), Type(nullptr), {});
    mod->Add(var, func, update);
  }
  *ret = mod;
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

TVM_REGISTER_GLOBAL("ir.Module_Import").set_body_typed([](IRModule mod, String path) {
  mod->Import(path);
});

TVM_REGISTER_GLOBAL("ir.Module_ImportFromStd").set_body_typed([](IRModule mod, String path) {
  mod->ImportFromStd(path);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IRModuleNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const IRModuleNode*>(ref.get());
      p->stream << "IRModule(" << node->functions << ")";
    });

}  // namespace tvm
