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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/variant.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/rvalue_ref.h>
#include <tvm/ir/module.h>
#include <tvm/ir/unique_name_supply.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace tvm {

namespace {

BaseFunc RemapModuleGlobals(const BaseFunc& function,
                            const ffi::Map<ffi::String, GlobalVar>& globals) {
  auto remap = [&globals](const GlobalVar& var) {
    auto it = globals.find(var->name_hint);
    return it == globals.end() ? var : (*it).second;
  };
  return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(function, remap).as_or_throw<BaseFunc>();
}

template <ffi::InplaceMode mode>
TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> IRModuleMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const IRModuleNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const IRModuleNode>(value);
  using FunctionMap = ffi::Map<GlobalVar, BaseFunc>;
  using GlobalInfoMap = ffi::Map<ffi::String, ffi::Array<GlobalInfo>>;
  // The name map and symbol types are derived from the resulting function map.
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<FunctionMap>, functions,
                                    mutator->MutateExpected(self->functions, mode));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<SourceMap>, source_map,
                                    mutator->MutateExpected(self->source_map, mode));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<DictAttrs>, attrs,
                                    mutator->MutateExpected(self->attrs, mode));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<GlobalInfoMap>, global_infos,
                                    mutator->MutateExpected(self->global_infos, mode));
  ffi::ObjectPtr<IRModuleNode> copy;
  IRModuleNode* result;
  if constexpr (mode == ffi::InplaceMode::kDisallow) {
    if (functions.UnchangedOrSameAs(self->functions) &&
        source_map.UnchangedOrSameAs(self->source_map) && attrs.UnchangedOrSameAs(self->attrs) &&
        global_infos.UnchangedOrSameAs(self->global_infos)) {
      return ffi::Unchanged();
    }
    copy = ffi::make_object<IRModuleNode>(*self);
    result = copy.get();
  } else {
    result = const_cast<IRModuleNode*>(self);
  }
  result->functions = std::move(functions).ValueOrUnchanged(std::move(result->functions));
  result->source_map = std::move(source_map).ValueOrUnchanged(std::move(result->source_map));
  result->attrs = std::move(attrs).ValueOrUnchanged(std::move(result->attrs));
  result->global_infos = std::move(global_infos).ValueOrUnchanged(std::move(result->global_infos));
  try {
    result->UpdateGlobalVarTypes();
  } catch (const ffi::Error& error) {
    return ffi::Unexpected(error);
  }
  if constexpr (mode == ffi::InplaceMode::kDisallow) {
    return ffi::Any(std::move(copy));
  } else {
    return ffi::Unchanged();
  }
}

}  // namespace

IRModule::IRModule(tvm::ffi::Map<GlobalVar, BaseFunc> functions, SourceMap source_map,
                   DictAttrs attrs, ffi::Map<ffi::String, ffi::Array<GlobalInfo>> global_infos) {
  auto n = ffi::make_object<IRModuleNode>();
  n->functions = std::move(functions);
  n->global_var_map_ = {};
  n->source_map = source_map;
  n->attrs = std::move(attrs);
  n->global_infos = std::move(global_infos);

  n->UpdateGlobalVarTypes();

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

int64_t IRModuleNode::SHash(int64_t init_hash,
                            ffi::TypedFunction<int64_t(AnyView, int64_t, bool)> hash) const {
  int64_t hash_value = init_hash;
  hash_value = hash(this->attrs, hash_value, false);
  hash_value = hash(this->global_infos, hash_value, false);

  // hash the functions.
  using KV = std::tuple<std::string, ffi::ObjectRef, ffi::ObjectRef>;
  std::vector<KV> temp;
  for (const auto& kv : this->functions) {
    temp.emplace_back(kv.first->name_hint, kv.first, kv.second);
  }
  // sort by the hash key of the keys.
  std::sort(temp.begin(), temp.end(),
            [](const KV& lhs, const KV& rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });
  uint64_t temp_size = static_cast<uint64_t>(temp.size());
  hash_value = hash(static_cast<int64_t>(temp_size), hash_value, false);
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

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  IRModuleNode::RegisterReflection();
  refl::TypeAttrDef<IRModuleNode>()
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&IRModuleMutate<ffi::InplaceMode::kDisallow>>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&IRModuleMutate<ffi::InplaceMode::kAllow>>());

  refl::GlobalDef().def(
      "ir.IRModule", [](tvm::ffi::Map<GlobalVar, BaseFunc> funcs, tvm::ffi::ObjectRef attrs,
                        ffi::Map<ffi::String, ffi::Array<GlobalInfo>> global_infos) {
        auto dict_attrs = [&attrs]() {
          if (!attrs.defined()) {
            return DictAttrs();
          } else if (auto* as_dict_attrs = attrs.as<tvm::DictAttrsNode>()) {
            return ffi::GetRef<tvm::DictAttrs>(as_dict_attrs);
          } else if (attrs.as<ffi::MapObj>()) {
            return tvm::DictAttrs(attrs.as_or_throw<ffi::Map<ffi::String, Any>>());
          } else {
            TVM_FFI_THROW(InternalError) << "Expected attrs argument to be either DictAttrs or "
                                            "ffi::Map<ffi::String,ObjectRef>";
          }
        }();

        return IRModule(funcs, {}, dict_attrs, global_infos);
      });
}

bool IRModuleNode::ContainGlobalVar(const ffi::String& name) const {
  return global_var_map_.find(name) != global_var_map_.end();
}

GlobalVar IRModuleNode::GetGlobalVar(const ffi::String& name) const {
  auto it = global_var_map_.find(name);
  if (it == global_var_map_.end()) {
    std::ostringstream msg;
    msg << "Cannot find global var \"" << name << "\" in the Module\n"
        << "candidates are: [";
    int counter = 0;
    for (auto kv : global_var_map_) {
      if (counter++ != 0) {
        msg << ", ";
      }
      msg << "\"" << kv.first << "\"";
    }
    msg << "]";
    TVM_FFI_THROW(ValueError) << msg.str();
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
  auto it = global_var_map_.find(var->name_hint);
  TVM_FFI_ICHECK(it == global_var_map_.end() || (*it).second.same_as(var))
      << "Duplicate global function name " << var->name_hint;
  // Replacements may reuse a caller from before a callee signature changed.
  // New definitions can deliberately retain old callees until a pass rewrites
  // their call arguments or results, so preserve those staged references.
  BaseFunc canonical_func =
      it == global_var_map_.end() ? func : RemapModuleGlobals(func, global_var_map_);
  this->functions.Set(var, canonical_func);
  if (var->ty.as<MissingType>().has_value()) var->ty = canonical_func->ty;
  if (canonical_func->ty.as<MissingType>().has_value() || var->ty.same_as(canonical_func->ty)) {
    global_var_map_.Set(var->name_hint, var);
    return;
  }
  UpdateGlobalVarTypes();
}

void IRModuleNode::UpdateGlobalVarTypes() {
  ffi::Map<ffi::String, GlobalVar> globals;
  bool remap = false;
  bool names_changed = global_var_map_.size() != functions.size();
  for (const auto& [var, function] : functions) {
    TVM_FFI_ICHECK_EQ(globals.count(var->name_hint), 0)
        << "Duplicate global function name " << var->name_hint;
    GlobalVar canonical = var;
    if (!function->ty.as<MissingType>().has_value()) {
      if (var->ty.as<MissingType>().has_value()) {
        var->ty = function->ty;
      } else if (!var->ty.same_as(function->ty)) {
        canonical = GlobalVar(var->name_hint, var->span);
        canonical->ty = function->ty;
        remap = true;
      }
    }
    auto previous = global_var_map_.find(var->name_hint);
    names_changed |= previous == global_var_map_.end() || !(*previous).second.same_as(canonical);
    if (previous != global_var_map_.end() && !(*previous).second.same_as(canonical)) {
      remap = true;
    }
    globals.Set(canonical->name_hint, canonical);
  }
  if (remap) {
    ffi::Map<GlobalVar, BaseFunc> updated;
    for (const auto& [var, function] : functions) {
      updated.Set(globals[var->name_hint], RemapModuleGlobals(function, globals));
    }
    functions = std::move(updated);
  }
  if (names_changed) global_var_map_ = std::move(globals);
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
  TVM_FFI_ICHECK(it != functions.end()) << "There is no definition of " << var;
  return (*it).second;
}

BaseFunc IRModuleNode::Lookup(const ffi::String& name) const {
  GlobalVar id = this->GetGlobalVar(name);
  return this->Lookup(id);
}

void IRModuleNode::Update(const IRModule& mod) {
  for (auto pair : mod->functions) {
    auto it = global_var_map_.find(pair.first->name_hint);
    GlobalVar canonical = it == global_var_map_.end() ? pair.first : (*it).second;
    this->AddUnchecked(canonical, pair.second);
  }
}

IRModule IRModuleNode::ShallowCopy() {
  return IRModule(this->functions, this->source_map, this->attrs, this->global_infos);
}

IRModule IRModule::FromExpr(const Expr& expr,
                            const tvm::ffi::Map<GlobalVar, BaseFunc>& global_funcs) {
  ffi::String gv_name;

  // All global definitions must be functions.
  BaseFunc func = expr.as_or_throw<BaseFunc>();
  {
    if (auto opt = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
      // Function literal has been annotated with it's required global symbol.
      gv_name = opt.value();
    }
  }

  // Replace a named definition before populating any initially untyped symbol,
  // so a temporary definition does not introduce an artificial signature change.
  if (!gv_name.empty()) {
    for (const auto& [var, function] : global_funcs) {
      if (var->name_hint == gv_name) {
        auto functions = global_funcs;
        functions.Set(var, func);
        return IRModule(std::move(functions));
      }
    }
  }
  auto mod = IRModule(global_funcs);

  UniqueNameSupply global_names(mod->functions.begin(), mod->functions.end(),
                                [](const auto& kv) { return kv.first->name_hint; });
  GlobalVar main_gv{ffi::UnsafeInit{}};
  if (gv_name.empty()) {
    // Bind function to 'main' (though rename if would clash with existing 'main').
    main_gv = GlobalVar(global_names->FreshName("main", false));
  } else if (mod->ContainGlobalVar(gv_name)) {
    main_gv = mod->GetGlobalVar(gv_name);
  } else {
    global_names->ReserveName(gv_name, false);
    main_gv = GlobalVar(gv_name);
  }
  mod->Add(main_gv, func);
  return mod;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ir.Module_Clone",
           [](IRModule mod) -> IRModule {
             IRModule clone = mod;
             clone.CopyOnWrite();
             return clone;
           })
      .def("ir.Module_Add",
           [](IRModule mod, GlobalVar var, ffi::ObjectRef val, bool update) -> IRModule {
             TVM_FFI_ICHECK(val->IsInstance<BaseFuncNode>());
             mod->Add(var, val.as_or_throw<BaseFunc>(), update);
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
                 TVM_FFI_THROW(InternalError) << "Variant didn't contain any of the allowed types";
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
               TVM_FFI_THROW(InternalError) << "Variant didn't contain any of the allowed types";
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
      .def("ir.Module_GetAttrs", [](IRModule mod) -> ffi::ObjectRef { return mod->GetAttrs(); })
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
      .def("ir.Module_GetAttr", [](IRModule mod, ffi::String key) -> ffi::Optional<ffi::ObjectRef> {
        return mod->GetAttr<ffi::ObjectRef>(key);
      });
}

}  // namespace tvm
