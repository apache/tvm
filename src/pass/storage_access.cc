/*!
 *  Copyright (c) 2017 by Contributors
 * \file storage_access.cc
 */
#include "./storage_access.h"

namespace tvm {
namespace ir {

void StorageAccessVisitor::Visit_(const Load* op) {
  const Variable* buf = op->buffer_var.as<Variable>();
  StorageScope scope = GetScope(buf);
  if (Enabled(buf, scope)) {
    CHECK(allow_append_);
    AccessEntry e;
    e.threads = env_threads();
    e.buffer = buf;
    e.dtype = op->type.element_of();
    e.touched = arith::IntSet::vector(op->index);
    e.type = kRead;
    e.scope = scope;
    curr_stmt_.access.emplace_back(std::move(e));
  }
  // traverse child
  IRVisitor::Visit_(op);
}

void StorageAccessVisitor::Visit_(const Store* op) {
  allow_append_ = true;
  CHECK_EQ(curr_stmt_.access.size(), 0U);
  curr_stmt_.stmt = op;
  const Variable* buf = op->buffer_var.as<Variable>();
  StorageScope scope = GetScope(buf);
  if (Enabled(buf, scope)) {
    AccessEntry e;
    e.threads = env_threads();
    e.buffer = buf;
    e.dtype = op->value.type().element_of();
    e.touched = arith::IntSet::vector(op->index);
    e.type = kWrite;
    e.scope = scope;
    curr_stmt_.access.emplace_back(std::move(e));
  }
  // traverse child
  IRVisitor::Visit_(op);
  // push to the scope
  scope_.back().push_back(curr_stmt_);
  // clear access entry.
  curr_stmt_.access.clear();
  allow_append_ = false;
}

void StorageAccessVisitor::Visit_(const Evaluate* op) {
  allow_append_ = true;
  CHECK_EQ(curr_stmt_.access.size(), 0U);
  curr_stmt_.stmt = op;
  IRVisitor::Visit_(op);
  // push to the scope
  if (curr_stmt_.access.size() != 0) {
    scope_.back().push_back(curr_stmt_);
    curr_stmt_.access.clear();
  }
  allow_append_ = false;
}

void StorageAccessVisitor::Visit_(const AttrStmt* op) {
  if (op->attr_key == attr::storage_scope) {
    const Variable* buf = op->node.as<Variable>();
    storage_scope_[buf] =
        StorageScope::make(op->value.as<StringImm>()->value);
    IRVisitor::Visit_(op);
  } else if (op->attr_key == attr::thread_extent) {
    IterVar iv(op->node.node_);
    env_threads_.push_back(iv);
    if (!in_device_env_) {
      in_device_env_ = true;
      scope_.push_back(std::vector<StmtEntry>());
      IRVisitor::Visit_(op);
      // no need to take the result as the thread barrier automatically syncs.
      Summarize(std::move(scope_.back()), nullptr);
      in_device_env_ = false;
      scope_.pop_back();
    } else {
      IRVisitor::Visit_(op);
    }
    env_threads_.CopyOnWrite()->data.pop_back();
  } else {
    IRVisitor::Visit_(op);
  }
}

void StorageAccessVisitor::Visit_(const For* op) {
  scope_.push_back(std::vector<StmtEntry>());
  IRVisitor::Visit_(op);
  StmtEntry s;
  s.stmt = op;
  s.access = Summarize(std::move(scope_.back()), op);
  scope_.pop_back();
  if (s.access.size() != 0) {
    // relax the touched set to contain all ranges in the loop.
    std::unordered_map<const Variable*, arith::IntSet> relax_map;
    relax_map[op->loop_var.get()] = arith::IntSet::range(
        Range::make_by_min_extent(op->min, op->extent));
    for (AccessEntry& e : s.access) {
      if (e.buffer != nullptr) {
        CHECK(e.touched.defined());
        e.touched = arith::EvalSet(e.touched, relax_map);
      }
    }
    scope_.back().emplace_back(std::move(s));
  }
}

void StorageAccessVisitor::Visit_(const IfThenElse* op) {
  ++condition_counter_;
  this->Visit(op->condition);
  scope_.push_back(std::vector<StmtEntry>());
  this->Visit(op->then_case);
  StmtEntry s;
  s.stmt = op;
  s.access = Summarize(std::move(scope_.back()), nullptr);
  scope_.pop_back();
  if (op->else_case.defined()) {
    scope_.push_back(std::vector<StmtEntry>());
    auto v = Summarize(std::move(scope_.back()), nullptr);
    scope_.pop_back();
    s.access.insert(s.access.end(), v.begin(), v.end());
  }
  scope_.back().emplace_back(std::move(s));
  --condition_counter_;
}

void StorageAccessVisitor::Visit_(const Call* op) {
  if (op->is_intrinsic(intrinsic::tvm_address_of)) {
    const Load *l = op->args[0].as<Load>();
    IRVisitor::Visit_(l);
  } else if (op->is_intrinsic(intrinsic::tvm_access_ptr)) {
    CHECK_EQ(op->args.size(), 5U);
    Type dtype = op->args[0].type();
    const Variable* buffer = op->args[1].as<Variable>();
    Expr offset = op->args[2];
    Expr extent = op->args[3];
    const IntImm* flag = op->args[4].as<IntImm>();
    StorageScope scope = GetScope(buffer);
    // The buffer scope.
    if (Enabled(buffer, scope)) {
      CHECK(allow_append_);
      AccessEntry e;
      e.threads = env_threads();
      e.dtype = dtype;
      e.buffer = buffer;
      e.touched = arith::IntSet::range(
          Range::make_by_min_extent(offset, extent));
      e.scope = scope;
      if (flag->value & 1) {
        e.type = kRead;
        curr_stmt_.access.emplace_back(e);
      }
      if (flag->value & 2) {
        e.type = kWrite;
        curr_stmt_.access.emplace_back(e);
      }
    }
    IRVisitor::Visit_(op);
  } else if (op->is_intrinsic(intrinsic::tvm_storage_sync)) {
    CHECK(allow_append_);
    const std::string& s = op->args[0].as<StringImm>()->value;
    if (s != "warp") {
      StorageScope scope = StorageScope::make(s);
      AccessEntry e;
      e.threads = env_threads();
      e.type = kSync;
      e.scope = StorageScope::make(s);
      curr_stmt_.access.emplace_back(std::move(e));
    }
  } else {
    IRVisitor::Visit_(op);
  }
}

StorageScope StorageAccessVisitor::GetScope(const Variable* buf) const {
  auto it = storage_scope_.find(buf);
  StorageScope s; s.rank = 0;
  if (it == storage_scope_.end()) return s;
  return it->second;
}
}  // namespace ir
}  // namespace tvm
