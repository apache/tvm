/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Replace certain copy with copy intrinsics.
 * \file copy_intrin_rewrite.cc
 */
#include <tvm/ir.h>
#include <tvm/packed_func_ext.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>

namespace tvm {
namespace ir {

using runtime::PackedFunc;

class CopyIntrinInjector : public IRMutator {
 public:
  CopyIntrinInjector(const std::string& pragma_key,
                     const PackedFunc& flower_copy_fromto)
      : pragma_key_(pragma_key),
        flower_copy_fromto_(flower_copy_fromto) {
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr::storage_scope) {
      const Variable* buf = op->node.as<Variable>();
      storage_scope_[buf] = op->value.as<StringImm>()->value;
    } else if (op->attr_key == ir::attr::pragma_scope) {
      const std::string& pname = op->value.as<StringImm>()->value;
      if (pname == pragma_key_) {
        Stmt ret;
        CHECK(MatchCopyPattern(op->body, &ret))
            << "Cannot match copy pattern of " << op->body;
        return ret;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool MatchCopyPattern(Stmt stmt, Stmt *out) {
    Stmt body = stmt;
    bool is_single_point_copy = false;

    // strip the loops
    std::vector<const For*> loops;
    while (const For* op = body.as<For>()) {
      if (!is_zero(op->min)) return false;
      loops.push_back(op);
      body = op->body;
    }
    const Store* store = body.as<Store>();
    if (store == nullptr) return false;
    const Select* select = store->value.as<Select>();
    const Cast* cast = store->value.as<Cast>();
    const Load* load = store->value.as<Load>();
    if (0 == loops.size()) {
      is_single_point_copy = true;
      CHECK(select == nullptr);
    }
    // for now only support true condition matching
    if (select != nullptr) {
      load = select->true_value.as<Load>();
    }
    // cast can be part of the pattern
    if (cast != nullptr) {
      load = cast->value.as<Load>();
    }
    if (load == nullptr) return false;
    if (load->type.lanes() != 1) return false;
    Array<Var> loop_vars;
    for (const For* op : loops) {
      loop_vars.push_back(Var(op->loop_var.node_));
    }
    Array<Expr> store_strides =
        arith::DetectLinearEquation(store->index, loop_vars);
    Array<Expr> load_strides =
        arith::DetectLinearEquation(load->index, loop_vars);
    if (load_strides.size()  == 0 || store_strides.size() == 0) return false;
    Array<Expr> dst_shape;
    auto loop_var_size = loop_vars.size();
    if (is_single_point_copy) {
      loop_var_size = 1;
      dst_shape.push_back(make_const(Int(32), 1));
    } else {
      for (const For* op : loops) {
        dst_shape.push_back(op->extent);
      }
    }
    Array<Expr> src_shape = dst_shape;
    Array<Expr> pad_before, pad_after;
    Expr pad_value;
    Expr src_elem_offset = load_strides[loop_var_size];
    if (select != nullptr) {
      Array<Expr> clip_bound =
          arith::DetectClipBound(select->condition, loop_vars);
      pad_value = select->false_value;
      if (clip_bound.size() == 0) return false;
      CHECK_EQ(src_shape.size(), loop_vars.size());
      CHECK_EQ(clip_bound.size(), loop_vars.size() * 2);
      for (size_t i = 0; i < src_shape.size(); ++i) {
        Expr min_value = clip_bound[2 * i];
        Expr max_value = clip_bound[2 * i + 1];
        Type t = loop_vars[i].type();
        Expr svalue = src_shape[i];
        if (min_value.defined()) {
          Expr pbefore = Simplify(Max::make(min_value, make_zero(t)));
          src_elem_offset = src_elem_offset + pbefore * load_strides[i];
          svalue = svalue - pbefore;
          pad_before.push_back(pbefore);
        } else {
          pad_before.push_back(make_zero(t));
        }
        if (max_value.defined()) {
          Expr pafter = Simplify(Max::make(loops[i]->extent - max_value - make_const(t, 1),
                                           make_zero(t)));
          svalue = svalue - pafter;
          pad_after.push_back(pafter);
        } else {
          pad_after.push_back(make_zero(t));
        }
        src_shape.Set(i, Simplify(svalue));
      }
      src_elem_offset = Simplify(src_elem_offset);
    }
    CHECK_EQ(load_strides.size(), store_strides.size());
    CHECK_EQ(load_strides.size(), loop_var_size + 1);
    Array<Expr> src_strides(load_strides.begin(), load_strides.begin() + loop_var_size);
    Array<Expr> dst_strides(store_strides.begin(), store_strides.begin() + loop_var_size);
    Buffer dst = BufferNode::make(
        Var(store->buffer_var.node_),
        store->value.type(),
        dst_shape,
        dst_strides,
        store_strides[loop_var_size],
        store->buffer_var->name_hint,
        GetStorageScope(store->buffer_var.get()),
        0, 0);
    Buffer src = BufferNode::make(
        Var(load->buffer_var.node_),
        load->type,
        src_shape,
        src_strides,
        src_elem_offset,
        load->buffer_var->name_hint,
        GetStorageScope(load->buffer_var.get()),
        0, 0);
    *out = flower_copy_fromto_(src, dst, pad_before, pad_after, pad_value);
    CHECK(out->defined()) << "flower function did not return correct stmt";
    return true;
  }
  // Get storage scope
  std::string GetStorageScope(const Variable* var) const {
    auto it = storage_scope_.find(var);
    if (it != storage_scope_.end()) {
      return it->second;
    } else {
      return "";
    }
  }
  // pragma key
  const std::string& pragma_key_;
  // function to lower copy intrinsics.
  const PackedFunc& flower_copy_fromto_;
  // Storage scope
  std::unordered_map<const Variable*, std::string> storage_scope_;
};

Stmt InjectCopyIntrin(Stmt stmt,
                      const std::string& pragma_key,
                      const PackedFunc& flower_copy_fromto) {
  return CopyIntrinInjector(pragma_key, flower_copy_fromto)
      .Mutate(stmt);
}

}  // namespace ir
}  // namespace tvm
