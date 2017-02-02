/*!
 *  Copyright (c) 2016 by Contributors
 * \file storage_flatten.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <unordered_map>
#include "./ir_util.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace ir {

// key of function buffer
struct TensorKey {
  FunctionRef f;
  int value_index;

  inline bool operator==(const TensorKey& other) const {
    return f == other.f && value_index == other.value_index;
  }
  inline std::string GetName() const {
    if (f->num_outputs() == 1) return f->func_name();
    std::ostringstream os;
    os << f->func_name() << ".v" << value_index;
    return os.str();
  }
};

}  // namespace ir
}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::ir::TensorKey> {
  std::size_t operator()(const ::tvm::ir::TensorKey& k) const {
    size_t lhs = k.f.hash();
    size_t rhs = static_cast<size_t>(k.value_index);
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
  }
};
}  // namespace std

namespace tvm {
namespace ir {

using Halide::Internal::Region;
using runtime::StorageScope;
using runtime::ThreadScope;

class StorageFlattener : public IRMutator {
 public:
  explicit StorageFlattener(Map<Tensor, Buffer> extern_buffer) {
    for (auto kv : extern_buffer) {
      BufferEntry e;
      e.buffer = kv.second;
      e.external = true;
      buf_map_[TensorKey{kv.first->op, kv.first->value_index}] = e;
    }
  }

  Stmt Flatten(Stmt stmt) {
    stmt = this->Mutate(stmt);
    StorageScope key; key.rank = 0;
    if (move_alloc_out_) {
      StorageScope key; key.rank = 0;
      stmt = MergeNest(allocs_[key], stmt);
    }
    return stmt;
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->type_key == "realize_scope") {
      storage_scope_[op->node.get()] = op->value.as<StringImm>()->value;
      return this->Mutate(op->body);
    } else if (op->type_key == "scope") {
      IterVar iv(op->node.node_);
      if (iv->thread_tag.length() != 0) {
        ThreadScope ts = ThreadScope::make(iv->thread_tag);
        curr_thread_scope_.push_back(ts);
        Stmt stmt = IRMutator::Mutate_(op, s);
        curr_thread_scope_.pop_back();
        op = stmt.as<AttrStmt>();

        bool first_scope = true;
        for (const ThreadScope& t : curr_thread_scope_) {
          if (t.rank == ts.rank) first_scope = false;
        }
        if (first_scope && move_alloc_out_) {
          StorageScope key;
          key.rank = ts.rank + 1;
          std::vector<Stmt>& vec = allocs_[key];
          if (vec.size() != 0) {
            Stmt body = MergeNest(vec, op->body);
            vec.clear();
            return AttrStmt::make(
                op->node, op->type_key, op->value, body);
          }
        }
        return stmt;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Provide>();
    TensorKey key{op->func, op->value_index};
    auto it = buf_map_.find(key);
    CHECK(it != buf_map_.end())
        << "Cannot find allocated buffer for " << key.f;
    const BufferEntry& e = it->second;
    CHECK(!e.released)
        << "Read a buffer that is already out of scope";
    return e.buffer.MakeStore(e.RelIndex(op->args), op->value);
  }

  Stmt Mutate_(const Realize* op, const Stmt& s) final {
    TensorKey key{op->func, op->value_index};
    if (buf_map_.count(key)) {
      CHECK(buf_map_.at(key).external);
      return this->Mutate(op->body);
    } else {
      // create a buffer entry
      // TODO(tqchen) allow permutation and inference of index dimension.
      BufferEntry e;
      e.bounds = op->bounds;
      Array<Expr> shape;
      for (auto r : e.bounds) {
        shape.push_back(r->extent);
      }
      e.buffer = Buffer(shape, op->type, key.GetName());

      buf_map_[key] = e;
      Stmt body = this->Mutate(op->body);
      buf_map_[key].released = true;
      // deduce current storage scope.
      auto it = storage_scope_.find(op->func.get());
      CHECK(it != storage_scope_.end());
      StorageScope key; key.rank = 0;
      const std::string& skey = it->second;
      if (skey.length() == 0) {
        if (curr_thread_scope_.size() != 0) {
          key.rank = curr_thread_scope_.back().rank + 1;
        }
      } else {
        key = StorageScope::make(skey);
      }

      if (move_alloc_out_) {
        allocs_[key].push_back(
            AttrStmt::make(
                e.buffer->data, "storage_scope",
                StringImm::make(key.to_string()),
                Evaluate::make(0)));
        allocs_[key].push_back(
            Allocate::make(
                e.buffer->data, e.buffer->dtype, e.buffer->shape,
                make_const(Bool(e.buffer->dtype.lanes()), true),
                Evaluate::make(0)));
        return body;
      } else {
        Stmt ret = Allocate::make(
            e.buffer->data, e.buffer->dtype, e.buffer->shape,
            make_const(Bool(e.buffer->dtype.lanes()), true), body);
        ret = AttrStmt::make(
            e.buffer->data, "storage_scope",
            StringImm::make(key.to_string()), ret);
        return ret;
      }
    }
  }

  Expr Mutate_(const Call* op, const Expr& olde) final {
    Expr expr = IRMutator::Mutate_(op, olde);
    op = expr.as<Call>();
    if (op != nullptr && op->call_type == Call::Halide) {
      TensorKey key{op->func, op->value_index};
      auto it = buf_map_.find(key);
      CHECK(it != buf_map_.end())
          << "Cannot find allocated buffer for " << key.f;
      const BufferEntry& e = it->second;
      CHECK(!e.released)
          << "Read a buffer that is already out of scope";
      return e.buffer.MakeLoad(e.RelIndex(op->args));
    } else {
      return expr;
    }
  }

 private:
  // The buffer entry in the flatten map
  struct BufferEntry {
    // the buffer of storage
    Buffer buffer;
    // the bounds of realization, can be null
    Region bounds;
    // Whether the buffer is external
    bool external{false};
    // Whether we are out of allocation bounds and buffer get released.
    bool released{false};
    // TODO(tqchen) allow permutation and inference of index dimension.
    // relative index
    inline Array<Expr> RelIndex(Array<Expr> args) const {
      if (bounds.size() != 0) {
        Array<Expr> index;
        CHECK_EQ(bounds.size(), args.size());
        for (size_t i = 0; i < bounds.size(); ++i) {
          index.push_back(args[i] - bounds[i]->min);
        }
        return index;
      } else {
        return args;
      }
    }
  };
  // whether move allocation to the outmost scope as possible.
  bool move_alloc_out_{true};
  // The buffer assignment map
  std::unordered_map<TensorKey, BufferEntry> buf_map_;
  std::unordered_map<const Node*, std::string> storage_scope_;
  // The current thread scope.
  std::vector<ThreadScope> curr_thread_scope_;
  // The allocations by rank
  std::unordered_map<StorageScope, std::vector<Stmt> > allocs_;
};

Stmt StorageFlatten(Stmt stmt,
                    Map<Tensor, Buffer> extern_buffer) {
  stmt = StorageFlattener(extern_buffer).Flatten(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace tvm
