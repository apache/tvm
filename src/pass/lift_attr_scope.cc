/*!
 *  Copyright (c) 2017 by Contributors
 *
 * \brief Lift specified AttrStmt scope to outer if
 *   the body contains the same scope.
 * \file lift_attr_scope.cc
 */
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include "./ir_util.h"

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
    std::vector<Stmt> seq;
    FlattenSeq(op->first, &seq);
    FlattenSeq(op->rest, &seq);
    seq = MutateSeq(seq);
    if (seq.size() == 2 &&
        seq[0].same_as(op->first) &&
        seq[1].same_as(op->rest)) {
      return s;
    }
    return MergeSeq(seq);
  }

  Stmt Mutate_(const IfThenElse* op, const Stmt& s) final {
    if (!op->else_case.defined()) {
      return IRMutator::Mutate_(op, s);
    }
    Stmt then_case = this->Mutate(op->then_case);
    NodeRef first_node;
    Expr first_value;
    std::swap(first_node, attr_node_);
    std::swap(first_value, attr_value_);
    Stmt else_case = this->Mutate(op->else_case);
    if (attr_node_.defined() &&
        attr_value_.defined() &&
        first_node.defined() &&
        first_value.defined() &&
        attr_node_.same_as(first_node) &&
        ValueSame(attr_value_, first_value)) {
      if (then_case.same_as(op->then_case) &&
          else_case.same_as(op->else_case)) {
        return s;
      } else {
        return IfThenElse::make(op->condition, then_case, else_case);
      }
    } else {
      if (first_node.defined()) {
        then_case = AttrStmt::make(
            first_node, attr_key_, first_value, then_case);
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
  void FlattenSeq(Stmt s, std::vector<Stmt>* res) {
    if (const Block* op = s.as<Block>()) {
      FlattenSeq(op->first, res);
      FlattenSeq(op->rest, res);
    } else if (const ProducerConsumer* op = s.as<ProducerConsumer>()) {
      if (!op->is_producer) {
        FlattenSeq(op->body, res);
      } else {
        res->emplace_back(s);
      }
    } else {
      res->emplace_back(s);
    }
  }

  std::vector<Stmt> MutateSeq(const std::vector<Stmt>& seq) {
    std::vector<Stmt> res_seq;
    NodeRef curr_node;
    Expr curr_value;
    Stmt curr_stmt;
    for (const Stmt & stmt : seq) {
      attr_node_ = NodeRef();
      attr_value_ = Expr();
      Stmt rest = this->Mutate(stmt);
      if (attr_node_.defined() &&
          attr_value_.defined() &&
          curr_node.defined() &&
          curr_value.defined() &&
          attr_node_.same_as(curr_node) &&
          ValueSame(attr_value_, curr_value)) {
        curr_stmt = Block::make(curr_stmt, rest);
      } else {
        if (curr_stmt.defined()) {
          if (curr_node.defined()) {
            curr_stmt = AttrStmt::make(
                curr_node, attr_key_, curr_value, curr_stmt);
          }
          res_seq.push_back(curr_stmt);
        }
        curr_stmt = rest;
        curr_node = attr_node_;
        curr_value = attr_value_;
      }
    }

    if (curr_stmt.defined()) {
      // keep attr_node_, attr_node_
      if (res_seq.size() == 0) {
        return {curr_stmt};
      }
      if (curr_node.defined()) {
        curr_stmt = AttrStmt::make(
            curr_node, attr_key_, curr_value, curr_stmt);
      }
      res_seq.push_back(curr_stmt);
      // reset
      attr_node_ = NodeRef();
      attr_value_ = Expr();
    }
    return res_seq;
  }

  // value comparison that also compares content of int constant
  static bool ValueSame(const Expr& a, const Expr& b) {
    if (a.same_as(b)) return true;
    if (a->type_key() != b->type_key()) return false;
    if (a.type() != b.type()) return false;
    if (const IntImm* op = a.as<IntImm>()) {
      return op->value == b.as<IntImm>()->value;
    }
    if (const UIntImm* op = a.as<UIntImm>()) {
      return op->value == b.as<UIntImm>()->value;
    }
    return false;
  }

  std::string attr_key_;
  NodeRef attr_node_;
  Expr attr_value_;
};

Stmt LiftAttrScope(Stmt stmt, std::string attr_key) {
  return AttrScopeLifter(attr_key).Lift(stmt);
}

}  // namespace ir
}  // namespace tvm
