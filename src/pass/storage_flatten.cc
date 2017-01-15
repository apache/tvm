/*!
 *  Copyright (c) 2016 by Contributors
 * \file storage_flatten.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <unordered_map>

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

// inliner to inline a function
// the result may not be SSA,
// ConvertSSA need to be applied after this pass
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
  Expr Mutate(Expr expr) final {
    expr = IRMutator::Mutate(expr);
    const Call* op = expr.as<Call>();
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

  Stmt Mutate(Stmt stmt) final {
    const Realize* realize = stmt.as<Realize>();
    if (realize != nullptr) {
      return HandleRealize(realize);
    } else if (stmt.as<Provide>()) {
      return HandleProvide(stmt);
    } else {
      return IRMutator::Mutate(stmt);
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

  // The buffer assignment map
  std::unordered_map<TensorKey, BufferEntry> buf_map_;

  Stmt HandleRealize(const Realize* op) {
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

      return Allocate::make(
          e.buffer->ptr, e.buffer->dtype, e.buffer->shape,
          make_const(Bool(e.buffer->dtype.lanes()), true), body);
    }
  }

  Stmt HandleProvide(Stmt stmt) {
    stmt = IRMutator::Mutate(stmt);
    const Provide* op = stmt.as<Provide>();
    TensorKey key{op->func, op->value_index};
    auto it = buf_map_.find(key);
    CHECK(it != buf_map_.end())
        << "Cannot find allocated buffer for " << key.f;
    const BufferEntry& e = it->second;
    CHECK(!e.released)
        << "Read a buffer that is already out of scope";
    return e.buffer.MakeStore(e.RelIndex(op->args), op->value);
  }
};


Stmt StorageFlatten(Stmt stmt,
                    Map<Tensor, Buffer> extern_buffer) {
  stmt = StorageFlattener(extern_buffer).Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace tvm
