/*!
 *  Copyright (c) 2017 by Contributors
 *
 * \brief Lift specified AttrStmt scope to outer if
 *   the body contains the same scope.
 * \file lift_attr_scope.cc
 */
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>

namespace tvm {
namespace ir {

// NOTE: this optimization can only be applied
// to a few specified attr keys
class AttrScopeLifter : public IRMutator {
 public:
  explicit AttrScopeLifter(std::string attr_key)
      : attr_key_(attr_key) {}

  Stmt Lift(Stmt stmt) {
    stmt = Mutate(stmt);
    if (attr_node_.defined()) {
      stmt = AttrStmt::make(
          attr_node_, attr_key_, attr_value_, stmt);
    }
    return stmt;
  }

  // do not go beyond
  Stmt Mutate_(const Allocate* op, const Stmt& s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Allocate>();
    if (attr_node_.defined()) {
      Stmt body = AttrStmt::make(
          attr_node_, attr_key_, attr_value_, op->body);
      // undefine them
      attr_node_ = NodeRef();
      attr_value_ = Expr();
      return Allocate::make(
        op->buffer_var, op->type,
        op->extents, op->condition, body,
        op->new_expr, op->free_function);
    } else {
      return stmt;
    }
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr_key_) {
      attr_node_ = op->node;
      attr_value_ = op->value;
      return op->body;
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

  Stmt Mutate_(const Block* op, const Stmt& s) final {
    Stmt first = this->Mutate(op->first);
    NodeRef first_node_;
    Expr first_value_;
    std::swap(first_node_, attr_node_);
    std::swap(first_value_, attr_value_);
    Stmt rest = this->Mutate(op->rest);
    if (attr_node_.defined() &&
        attr_value_.defined() &&
        first_node_.defined() &&
        first_value_.defined() &&
        attr_node_.same_as(first_node_) &&
        attr_value_.same_as(first_value_)) {
      if (first.same_as(op->first) && rest.same_as(op->rest)) {
        return s;
      } else {
        return Block::make(first, rest);
      }
    } else {
      if (first_node_.defined()) {
        first = AttrStmt::make(
            first_node_, attr_key_, first_value_, first);
      }
      if (attr_node_.defined()) {
        rest = AttrStmt::make(
            attr_node_, attr_key_, attr_value_, rest);
        // undefine them
        attr_node_ = NodeRef();
        attr_value_ = Expr();
      }
      if (first.same_as(op->first) && rest.same_as(op->rest)) {
        return s;
      } else {
        return Block::make(first, rest);
      }
    }
  }

  Stmt Mutate_(const IfThenElse* op, const Stmt& s) final {
    if (!op->else_case.defined()) {
      return IRMutator::Mutate_(op, s);
    }
    Stmt then_case = this->Mutate(op->then_case);
    NodeRef first_node_;
    Expr first_value_;
    std::swap(first_node_, attr_node_);
    std::swap(first_value_, attr_value_);
    Stmt else_case = this->Mutate(op->else_case);
    if (attr_node_.defined() &&
        attr_value_.defined() &&
        first_node_.defined() &&
        first_value_.defined() &&
        attr_node_.same_as(first_node_) &&
        attr_value_.same_as(first_value_)) {
      if (then_case.same_as(op->then_case) &&
          else_case.same_as(op->else_case)) {
        return s;
      } else {
        return IfThenElse::make(op->condition, then_case, else_case);
      }
    } else {
      if (first_node_.defined()) {
        then_case = AttrStmt::make(
            first_node_, attr_key_, first_value_, then_case);
      }
      if (attr_node_.defined()) {
        else_case = AttrStmt::make(
            attr_node_, attr_key_, attr_value_, else_case);
        // undefine them
        attr_node_ = NodeRef();
        attr_value_ = Expr();
      }
      if (then_case.same_as(op->then_case) &&
          else_case.same_as(op->else_case)) {
        return s;
      } else {
        return IfThenElse::make(op->condition, then_case, else_case);
      }
    }
  }

 private:
  std::string attr_key_;
  NodeRef attr_node_;
  Expr attr_value_;
};

Stmt LiftAttrScope(Stmt stmt, std::string attr_key) {
  return AttrScopeLifter(attr_key).Lift(stmt);
}

}  // namespace ir
}  // namespace tvm
