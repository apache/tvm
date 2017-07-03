/*!
 *  Copyright (c) 2016 by Contributors
 * \file storage_flatten.cc
 */
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/buffer.h>
#include <unordered_map>
#include "../arithmetic/compute_expr.h"
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
    auto it = var_remap_.find(op->buffer_var.get());
    if (it != var_remap_.end() &&
        !it->second.same_as(op->buffer_var)) {
      CHECK(it->second.as<Variable>());
      VarExpr buf_var(it->second.node_);
      return Store::make(buf_var, op->value, op->index, op->predicate);
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
    } else if (op->attr_key == attr::buffer_bind_scope) {
      return HandleBufferBindScope(op);
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
      e.buffer = decl_buffer(shape, op->type, key.GetName());

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
    auto it = var_remap_.find(op->buffer_var.get());
    if (it != var_remap_.end() &&
        !it->second.same_as(op->buffer_var)) {
      CHECK(it->second.as<Variable>());
      VarExpr buf_var(it->second.node_);
      return Load::make(op->type, buf_var, op->index, op->predicate);
    } else {
      return expr;
    }
  }

  Expr Mutate_(const Variable* op, const Expr& e) final {
    auto it = var_remap_.find(op);
    if (it != var_remap_.end()) {
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
  // Bind the symbol sym to value if it is a Variable
  // send a sequence of asserts if it is a constant constrant.
  // hint_name: used for error message
  // add_keys: a list of newly binded keys
  // add_asserts: a list of asserts during the bind
  void BindSymbol(Expr sym,
                  Expr value,
                  std::string hint_name,
                  std::vector<const Variable*>* add_keys,
                  std::vector<Stmt>* add_asserts) {
    if (const Variable* v = sym.as<Variable>()) {
      auto it = var_remap_.find(v);
      if (it == var_remap_.end()) {
        add_keys->push_back(v);
        var_remap_[v] = value;
        return;
      }
    }
    // add assertions
    std::ostringstream os;
    os << "BufferBind constaint fail " << hint_name;
    add_asserts->emplace_back(
        AssertStmt::make(sym == value, os.str()));
  }
  // Start bind
  Stmt HandleBufferBindScope(const AttrStmt* op) {
    Array<NodeRef> arr(op->node.node_);
    CHECK_EQ(arr.size(), 2U);
    const BufferNode* buffer = arr[0].as<BufferNode>();
    const TensorNode* tensor = arr[1].as<TensorNode>();
    const Call* tuple = op->value.as<Call>();
    CHECK(buffer && tensor);
    CHECK(tuple && tuple->is_intrinsic(intrinsic::tvm_tuple));
    TensorKey key{tensor->op, tensor->value_index};
    CHECK(buf_map_.count(key));
    const BufferEntry& be = buf_map_.at(key);
    CHECK(!be.released);
    CHECK_EQ(tuple->args.size(), be.buffer->shape.size() * 2);
    Array<Expr> begins, extents;
    if (be.bounds.size() != 0) {
      CHECK_EQ(tuple->args.size(), be.bounds.size() * 2);
      for (size_t i = 0; i < be.buffer->shape.size(); ++i) {
        begins.push_back(
            arith::ComputeExpr<Sub>(tuple->args[2 * i], be.bounds[i]->min));
        extents.push_back(tuple->args[2 * i + 1]);
      }
    } else {
      for (size_t i = 0; i < tuple->args.size(); i += 2) {
        begins.push_back(tuple->args[i]);
        extents.push_back(tuple->args[i + 1]);
      }
    }
    Buffer slice = be.buffer.MakeSlice(begins, extents);
    if (buffer->strides.size() == 0) {
      CHECK_EQ(slice->strides.size(), 0U)
          << "Trying to bind compact buffer to strided one";
    } else {
      slice = slice.MakeStrideView();
    }
    CHECK_EQ(slice->strides.size(), buffer->strides.size());
    // start binding
    std::vector<const Variable*> keys;
    std::vector<Stmt> asserts;
    BindSymbol(buffer->data, slice->data,
               buffer->name + ".data",
               &keys, &asserts);
    for (size_t i = 0; i < buffer->shape.size(); ++i) {
      std::ostringstream field_name;
      field_name << buffer->name << ".shape[" << i << ']';
      BindSymbol(buffer->shape[i], slice->shape[i],
                 field_name.str(),
                 &keys, &asserts);
    }
    for (size_t i = 0; i < buffer->strides.size(); ++i) {
      std::ostringstream field_name;
      field_name << buffer->name << ".strides[" << i << ']';
      BindSymbol(buffer->strides[i], slice->strides[i],
                 field_name.str(),
                 &keys, &asserts);
    }
    BindSymbol(buffer->elem_offset, slice->elem_offset,
               buffer->name + ".elem_offset",
               &keys, &asserts);
    CHECK_EQ(buffer->scope, slice->scope)
        << "Buffer bind scope mismatch";
    // Apply the remaps
    Stmt body = this->Mutate(op->body);
    for (size_t i = 0; i < asserts.size(); ++i) {
      Stmt ret = Simplify(this->Mutate(asserts[i]));
      if (const AssertStmt* assert_op = ret.as<AssertStmt>()) {
        if (!is_zero(assert_op->condition)) {
          body = Block::make(ret, body);
        } else {
          LOG(FATAL) << "BindBuffer have unmet assertion: " << ret;
        }
      }
    }
    // remove the binds
    for (const Variable* op : keys) {
      var_remap_.erase(op);
    }
    return body;
  }

  // The buffer entry in the flatten map
  struct BufferEntry {
    // the buffer of storage
    Buffer buffer;
    // the bounds of realization, can be null, means everything
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
  // Variable remap
  std::unordered_map<const Variable*, Expr> var_remap_;
  // Buffer map
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
