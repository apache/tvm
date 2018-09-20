/*!
 *  Copyright (c) 2018 by Contributors
 * \file remap_thread_axis.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <unordered_map>


namespace tvm {
namespace ir {

// Mutator to change the read pattern
class ThreadAxisRewriter : private IRMutator {
 public:
  explicit ThreadAxisRewriter(
      const std::unordered_map<std::string, IterVar>& tmap)
      : tmap_(tmap) {
  }

  Stmt Rewrite(Stmt stmt) {
    return Mutate(stmt);
  }

 private:
  Stmt Mutate_(const AttrStmt* op, const Stmt& stmt) final {
    if (op->attr_key == attr::thread_extent) {
      IterVar iv(op->node.node_);
      CHECK_NE(iv->thread_tag.length(), 0U);
      auto it = tmap_.find(iv->thread_tag);
      if (it != tmap_.end()) {
        const IterVar& new_iv = it->second;
        const Variable* v = iv->var.get();
        if (!vmap_.count(v)) {
          vmap_[v] = new_iv->var;
        } else {
          CHECK(vmap_[v].same_as(new_iv->var));
        }
        Stmt body = this->Mutate(op->body);
        return AttrStmt::make(
            new_iv, op->attr_key, op->value, body);
      }
    }
    return IRMutator::Mutate_(op, stmt);
  }

  Expr Mutate_(const Variable* op, const Expr& expr) final {
    auto it = vmap_.find(op);
    if (it != vmap_.end()) return it->second;
    return IRMutator::Mutate_(op, expr);
  }
  // The thread map
  const std::unordered_map<std::string, IterVar>& tmap_;
  // variable map
  std::unordered_map<const Variable*, Var> vmap_;
};

LoweredFunc
RemapThreadAxis(LoweredFunc f, Map<Expr, IterVar> thread_map) {
  std::unordered_map<std::string, IterVar> tmap;
  for (const auto& kv : thread_map) {
    const StringImm* str = kv.first.as<StringImm>();
    CHECK(str != nullptr);
    tmap[str->value] = kv.second;
  }

  CHECK_EQ(f->func_type, kDeviceFunc);
  auto n = make_node<LoweredFuncNode>(*f.operator->());
  // replace the thread axis
  for (size_t i = 0; i < n->thread_axis.size(); ++i) {
    auto it = tmap.find(n->thread_axis[i]->thread_tag);
    if (it != tmap.end()) {
      n->thread_axis.Set(i, it->second);
    }
  }
  n->body = ThreadAxisRewriter(tmap).Rewrite(n->body);
  return LoweredFunc(n);
}

}  // namespace ir
}  // namespace tvm
