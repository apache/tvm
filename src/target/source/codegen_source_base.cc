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
 * \file codegen_source_base.cc
 */
#include "codegen_source_base.h"

#include <algorithm>

namespace tvm {
namespace codegen {

void CodeGenSourceBase::ClearFuncState() {
  name_supply_ = NameSupply();
  ssa_assign_map_.clear();
  var_idmap_.clear();
  scope_mark_.clear();
}

std::string CodeGenSourceBase::SSAGetID(std::string src, DataType t) {
  if (name_supply_->ContainsName(src)) return src;
  auto it = ssa_assign_map_.find(src);
  if (it != ssa_assign_map_.end()) {
    if (scope_mark_.at(it->second.scope_id)) {
      return it->second.vid;
    }
  }
  SSAEntry e;
  // use v_ prefix so it works for most systems
  e.vid = name_supply_->FreshName("v_");
  e.scope_id = static_cast<int>(scope_mark_.size() - 1);
  ssa_assign_map_[src] = e;
  this->PrintIndent();
  PrintSSAAssign(e.vid, src, t);
  return e.vid;
}

std::string CodeGenSourceBase::AllocVarID(const tir::VarNode* v) {
  ICHECK(!var_idmap_.count(v)) << "Need input to be in SSA form dup " << v->name_hint;
  std::string key = v->name_hint;
  std::string vid = name_supply_->FreshName(key);
  std::replace(vid.begin(), vid.end(), ':', '_');
  std::replace(vid.begin(), vid.end(), '-', '_');
  std::replace(vid.begin(), vid.end(), '.', '_');
  var_idmap_[v] = vid;
  return vid;
}

std::string CodeGenSourceBase::GetVarID(const tir::VarNode* v) const {
  auto it = var_idmap_.find(v);
  ICHECK(it != var_idmap_.end()) << "Find undefined Variable " << v->name_hint;
  return it->second;
}

void CodeGenSourceBase::PrintIndent() {
  for (int i = 0; i < indent_; ++i) {
    this->stream << ' ';
  }
}

void CodeGenSourceBase::MarkConst(std::string vid) {
  auto it = ssa_assign_map_.find(vid);
  if (it == ssa_assign_map_.end()) {
    SSAEntry e;
    e.vid = vid;
    e.scope_id = 0;
    ssa_assign_map_[vid] = e;
  } else {
    ICHECK_EQ(it->second.vid, vid);
  }
}

int CodeGenSourceBase::BeginScope() {
  int sid = static_cast<int>(scope_mark_.size());
  scope_mark_.push_back(true);
  indent_ += 2;
  return sid;
}

void CodeGenSourceBase::EndScope(int scope_id) {
  scope_mark_[scope_id] = false;
  indent_ -= 2;
}

void CodeGenSourceBase::PrintType(DataType type, std::ostream& os) {  // NOLINT(*)
  ICHECK_EQ(type.lanes(), 1) << "do not yet support vector types";
  if (type.is_handle()) {
    os << "void*";
    return;
  }
  if (type.is_void()) {
    os << "void";
    return;
  }
  if (type.is_float()) {
    if (type.bits() == 32) {
      os << "float";
      return;
    }
    if (type.bits() == 64) {
      os << "double";
      return;
    }
  } else if (type.is_uint()) {
    switch (type.bits()) {
      case 8:
      case 16:
      case 32:
      case 64: {
        os << "uint" << type.bits() << "_t";
        return;
      }
      case 1:
        os << "int";
        return;
    }
  } else if (type.is_int()) {
    switch (type.bits()) {
      case 8:
      case 16:
      case 32:
      case 64: {
        os << "int" << type.bits() << "_t";
        return;
      }
    }
  }
  LOG(FATAL) << "Cannot convert type " << type << " to C type";
}

void CodeGenSourceBase::PrintType(const Type& type, std::ostream& os) {  // NOLINT(*)
  if (auto* ptr = type.as<PrimTypeNode>()) {
    return PrintType(ptr->dtype, os);
  } else if (auto* ptr = type.as<PointerTypeNode>()) {
    PrintType(ptr->element_type, os);
    os << '*';
  } else if (IsVoidType(type)) {
    os << "void";
  } else {
    LOG(FATAL) << "Type " << type << " does not have a corresponding C Type";
  }
}

}  // namespace codegen
}  // namespace tvm
