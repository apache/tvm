/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_source_base.cc
 */
#include "codegen_source_base.h"

namespace tvm {
namespace codegen {

void CodeGenSourceBase::ClearFuncState() {
  name_alloc_map_.clear();
  ssa_assign_map_.clear();
  var_idmap_.clear();
  scope_mark_.clear();
}

std::string CodeGenSourceBase::GetUniqueName(std::string prefix) {
  for (size_t i = 0; i < prefix.size(); ++i) {
    if (prefix[i] == '.') prefix[i] = '_';
  }
  auto it = name_alloc_map_.find(prefix);
  if (it != name_alloc_map_.end()) {
    while (true) {
      std::ostringstream os;
      os << prefix << (++it->second);
      std::string name = os.str();
      if (name_alloc_map_.count(name) == 0) {
        prefix = name;
        break;
      }
    }
  }
  name_alloc_map_[prefix] = 0;
  return prefix;
}

std::string CodeGenSourceBase::SSAGetID(std::string src, Type t) {
  if (name_alloc_map_.count(src)) return src;
  auto it = ssa_assign_map_.find(src);
  if (it != ssa_assign_map_.end()) {
    if (scope_mark_.at(it->second.scope_id)) {
      return it->second.vid;
    }
  }
  SSAEntry e;
  e.vid = GetUniqueName("_");
  e.scope_id = static_cast<int>(scope_mark_.size() - 1);
  ssa_assign_map_[src] = e;
  this->PrintIndent();
  PrintSSAAssign(e.vid, src, t);
  return e.vid;
}

std::string CodeGenSourceBase::AllocVarID(const Variable* v) {
  CHECK(!var_idmap_.count(v))
      << "Need input to be in SSA form dup " << v->name_hint;
  std::string key = v->name_hint;
  std::string vid = GetUniqueName(key);
  var_idmap_[v] = vid;
  return vid;
}

std::string CodeGenSourceBase::GetVarID(const Variable* v) const {
  auto it = var_idmap_.find(v);
  CHECK(it != var_idmap_.end())
      << "Find undefined Variable " << v->name_hint;
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
    CHECK_EQ(it->second.vid, vid);
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

}  // namespace codegen
}  // namespace tvm
