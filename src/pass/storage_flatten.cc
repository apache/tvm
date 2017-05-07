/*!
 *  Copyright (c) 2016 by Contributors
 * \file storage_flatten.cc
 */
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/buffer.h>
#include <tvm/operation.h>
#include <unordered_map>
#include "../runtime/thread_storage_scope.h"

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
  Stmt Mutate_(const Store* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Store>();
    auto it = extern_buf_remap_.find(op->buffer_var.get());
    if (it != extern_buf_remap_.end()) {
      return Store::make(it->second, op->value, op->index, op->predicate);
    } else {
      return stmt;
    }
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr::realize_scope) {
      storage_scope_[op->node.get()] = op->value.as<StringImm>()->value;
      return this->Mutate(op->body);
    } else if (op->attr_key == attr::thread_extent) {
      IterVar iv(op->node.node_);
      ThreadScope ts = ThreadScope::make(iv->thread_tag);
      curr_thread_scope_.push_back(ts);
      Stmt stmt = IRMutator::Mutate_(op, s);
      curr_thread_scope_.pop_back();
      return stmt;
    } else if (op->attr_key == attr::extern_op_scope) {
      return HandleExternOp(op);
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
      CHECK(it != storage_scope_.end())
          << "Cannot find storage scope of " << op->func
          << " value_index=" << op->value_index;
      StorageScope skey;
      const std::string& strkey = it->second;
      if (strkey.length() == 0) {
        if (curr_thread_scope_.size() != 0) {
          skey.rank = curr_thread_scope_.back().rank + 1;
        }
      } else {
        skey = StorageScope::make(strkey);
      }
      Stmt ret = Allocate::make(
          e.buffer->data, e.buffer->dtype, e.buffer->shape,
          make_const(Bool(e.buffer->dtype.lanes()), true), body);
      ret = AttrStmt::make(
          e.buffer->data, attr::storage_scope,
          StringImm::make(skey.to_string()), ret);
      return ret;
    }
  }

  Expr Mutate_(const Load* op, const Expr& e) final {
    Expr expr = IRMutator::Mutate_(op, e);
    op = expr.as<Load>();
    auto it = extern_buf_remap_.find(op->buffer_var.get());
    if (it != extern_buf_remap_.end()) {
      return Load::make(op->type, it->second, op->index, op->predicate);
    } else {
      return expr;
    }
  }

  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = extern_buf_remap_.find(op);
    if (it != extern_buf_remap_.end()) {
      return it->second;
    } else {
      return e;
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
  Stmt HandleExternOp(const AttrStmt* op) {
    const ExternOpNode* ext_op = op->node.as<ExternOpNode>();
    CHECK(ext_op);
    Operation func(op->node.node_);
    CHECK_EQ(extern_buf_remap_.size(), 0U);
    for (size_t i = 0; i < ext_op->output_placeholders.size(); ++i) {
      TensorKey key{func, static_cast<int>(i)};
      CHECK(buf_map_.count(key));
      extern_buf_remap_[ext_op->output_placeholders[i]->data.get()] =
          buf_map_.at(key).buffer->data;
    }
    for (size_t i = 0; i < ext_op->inputs.size(); ++i) {
      TensorKey key{ext_op->inputs[i]->op, ext_op->inputs[i]->value_index};
      CHECK(buf_map_.count(key));
      extern_buf_remap_[ext_op->input_placeholders[i]->data.get()] =
          buf_map_.at(key).buffer->data;
    }
    Stmt ret = Mutate(op->body);
    extern_buf_remap_.clear();
    return ret;
  }

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
  // The buffer assignment map
  std::unordered_map<const Variable*, Var> extern_buf_remap_;
  std::unordered_map<TensorKey, BufferEntry> buf_map_;
  std::unordered_map<const Node*, std::string> storage_scope_;
  // The current thread scope.
  std::vector<ThreadScope> curr_thread_scope_;
};

Stmt StorageFlatten(Stmt stmt,
                    Map<Tensor, Buffer> extern_buffer) {
  stmt = StorageFlattener(extern_buffer).Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace tvm
