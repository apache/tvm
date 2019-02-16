/*!
 *  Copyright (c) 2018 by Contributors
 * \file src/tvm/relay/ir/alpha_equal.cc
 * \brief Alpha equality check by deep comparing two nodes.
 */
#include <tvm/ir_pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/relay/pass.h>
#include "type_functor.h"
#include "../../lang/attr_functor.h"

namespace tvm {
namespace relay {

// Alpha Equal handler for Relay.
class AlphaEqualHandler:
      public AttrsEqualHandler,
      public TypeFunctor<bool(const Type&, const Type&)>,
      public ExprFunctor<bool(const Expr&, const Expr&)>,
      public PatternFunctor<bool(const Pattern&, const Pattern&)> {
 public:
  explicit AlphaEqualHandler(bool map_free_var)
      : map_free_var_(map_free_var) {}

  /*!
   * Check equality of two nodes.
   * \param lhs The left hand operand.
   * \param rhs The right hand operand.
   * \return The comparison result.
   */
  bool Equal(const NodeRef& lhs, const NodeRef& rhs) {
    if (lhs.same_as(rhs)) return true;
    if (!lhs.defined() || !rhs.defined()) return false;
    if (lhs->derived_from<TypeNode>()) {
      if (!rhs->derived_from<TypeNode>()) return false;
      return TypeEqual(Downcast<Type>(lhs), Downcast<Type>(rhs));
    }
    if (lhs->derived_from<ExprNode>()) {
      if (!rhs->derived_from<ExprNode>()) return false;
      return ExprEqual(Downcast<Expr>(lhs), Downcast<Expr>(rhs));
    }
    return AttrEqual(lhs, rhs);
  }

  /*!
   * Check equality of two attributes.
   * \param lhs The left hand operand.
   * \param rhs The right hand operand.
   * \return The comparison result.
   */
  bool AttrEqual(const NodeRef& lhs, const NodeRef& rhs) {
    return AttrsEqualHandler::Equal(lhs, rhs);
  }
  /*!
   * Check equality of two types.
   * \param lhs The left hand operand.
   * \param rhs The right hand operand.
   * \return the comparison result.
   */
  bool TypeEqual(const Type& lhs, const Type& rhs) {
    if (lhs.same_as(rhs)) return true;
    if (!lhs.defined() || !rhs.defined()) return false;
    return this->VisitType(lhs, rhs);
  }
  /*!
   * Check equality of two expressions.
   *
   * \note We run graph structural equality checking when comparing two Exprs.
   *   This means that AlphaEqualHandler can only be used once for each pair.
   *   The equality checker checks data-flow equvalence of the Expr DAG.
   *   This function also runs faster as it memomizes equal_map.
   *
   * \param lhs The left hand operand.
   * \param rhs The right hand operand.
   * \return The comparison result.
   */
  bool ExprEqual(const Expr& lhs, const Expr& rhs) {
    if (lhs.same_as(rhs)) return true;
    if (!lhs.defined() || !rhs.defined()) return false;
    auto it = equal_map_.find(lhs);
    if (it != equal_map_.end()) {
      return it->second.same_as(rhs);
    }
    if (this->VisitExpr(lhs, rhs)) {
      equal_map_[lhs] = rhs;
      return true;
    } else {
      return false;
    }
  }

 protected:
  /*!
   * \brief Check if data type equals each other.
   * \param lhs The left hand operand.
   * \param rhs The right hand operand.
   * \return The compare result.
   */
  bool DataTypeEqual(const DataType& lhs, const DataType& rhs) {
    return lhs == rhs;
  }
  /*!
   * \brief Check Equality of leaf node of the graph.
   *  if map_free_var_ is set to true, try to map via equal node.
   * \param lhs The left hand operand.
   * \param rhs The right hand operand.
   * \return The compare result.
   */
  bool LeafNodeEqual(const NodeRef& lhs, const NodeRef& rhs) {
    if (lhs.same_as(rhs)) return true;
    auto it = equal_map_.find(lhs);
    if (it != equal_map_.end()) {
      return it->second.same_as(rhs);
    } else {
      if (map_free_var_) {
        if (lhs->type_index() != rhs->type_index()) return false;
        equal_map_[lhs] = rhs;
        return true;
      } else {
        return false;
      }
    }
  }
  using AttrsEqualHandler::VisitAttr_;
  bool VisitAttr_(const Variable* lhs, const NodeRef& other) final {
    return LeafNodeEqual(GetRef<NodeRef>(lhs), other);
  }

  // Type equality
  bool VisitType_(const TensorTypeNode* lhs, const Type& other) final {
    if (const TensorTypeNode* rhs = other.as<TensorTypeNode>()) {
      return (lhs->dtype == rhs->dtype &&
              AttrEqual(lhs->shape, rhs->shape));
    } else {
      return false;
    }
  }

  bool VisitType_(const IncompleteTypeNode* lhs, const Type& other) final {
    return LeafNodeEqual(GetRef<NodeRef>(lhs), other);
  }

  bool VisitType_(const TypeVarNode* lhs, const Type& other) final {
    if (const TypeVarNode* rhs = other.as<TypeVarNode>()) {
      if (lhs->kind != rhs->kind) return false;
      return LeafNodeEqual(GetRef<NodeRef>(lhs), other);
    } else {
      return false;
    }
  }

  bool VisitType_(const FuncTypeNode* lhs, const Type& other) final {
    if (const FuncTypeNode* rhs = other.as<FuncTypeNode>()) {
      if (lhs->arg_types.size() != rhs->arg_types.size()) return false;
      if (lhs->type_params.size() != rhs->type_params.size()) return false;
      if (lhs->type_constraints.size() != rhs->type_constraints.size()) return false;
      for (size_t i = 0; i < lhs->type_params.size(); ++i) {
        if (lhs->type_params[i]->kind != rhs->type_params[i]->kind) {
          return false;
        }
        equal_map_[lhs->type_params[i]] = rhs->type_params[i];
        // set up type parameter equal
        if (lhs->type_params[i]->kind == Kind::kShapeVar) {
          // map variable
          equal_map_[lhs->type_params[i]->var] = rhs->type_params[i]->var;
        }
      }
      for (size_t i = 0; i < lhs->arg_types.size(); i++) {
        if (!TypeEqual(lhs->arg_types[i], rhs->arg_types[i])) return false;
      }
      if (!TypeEqual(lhs->ret_type, rhs->ret_type)) return false;
      for (size_t i = 0; i < lhs->type_constraints.size(); i++) {
        if (!TypeEqual(lhs->type_constraints[i],
                       rhs->type_constraints[i])) {
          return false;
        }
      }
      return true;
    } else {
      return false;
    }
  }

  bool VisitType_(const TypeRelationNode* lhs, const Type& other) final {
    if (const TypeRelationNode* rhs = other.as<TypeRelationNode>()) {
      if (lhs->func->name != rhs->func->name) return false;
      if (lhs->num_inputs != rhs->num_inputs) return false;
      if (!this->AttrEqual(lhs->attrs, rhs->attrs)) return false;
      if (lhs->args.size() != rhs->args.size()) return false;
      for (size_t i = 0; i < lhs->args.size(); ++i) {
        if (!TypeEqual(lhs->args[i], rhs->args[i])) return false;
      }
      return true;
    } else {
      return false;
    }
  }

  bool VisitType_(const TupleTypeNode* lhs, const Type& other) final {
    if (const TupleTypeNode* rhs = other.as<TupleTypeNode>()) {
      if (lhs->fields.size() != rhs->fields.size()) return false;
      for (size_t i = 0; i < lhs->fields.size(); ++i) {
        if (!TypeEqual(lhs->fields[i], rhs->fields[i])) return false;
      }
      return true;
    } else {
      return false;
    }
  }

  bool VisitType_(const RefTypeNode* lhs, const Type& other) final {
    if (const RefTypeNode* rhs = other.as<RefTypeNode>()) {
      return TypeEqual(lhs->value, rhs->value);
    }
    return false;
  }

  bool VisitType_(const GlobalTypeVarNode* op, const Type& t2) final {
    return GetRef<Type>(op) == t2;
  }

  bool VisitType_(const TypeCallNode* op, const Type& t2) final {
    const TypeCallNode* pt = t2.as<TypeCallNode>();
    if (pt == nullptr
        || op->args.size() != pt->args.size()
        || !TypeEqual(op->func, pt->func)) {
      return false;
    }

    for (size_t i = 0; i < op->args.size(); ++i) {
      if (!TypeEqual(op->args[i], pt->args[i])) {
        return false;
      }
    }
    return true;
  }

  // Expr equal checking.
  bool NDArrayEqual(const runtime::NDArray& lhs,
                    const runtime::NDArray& rhs) {
    if (lhs.defined() != rhs.defined()) {
      return false;
    } else if (lhs.same_as(rhs)) {
      return true;
    } else {
      auto ldt = lhs->dtype;
      auto rdt = rhs->dtype;
      CHECK_EQ(lhs->ctx.device_type, kDLCPU) << "can only compare CPU tensor";
      CHECK_EQ(rhs->ctx.device_type, kDLCPU) << "can only compare CPU tensor";
      if (ldt.code == rdt.code && ldt.lanes == rdt.lanes && ldt.bits == rdt.bits) {
        size_t data_size = runtime::GetDataSize(*lhs.operator->());
        return std::memcmp(lhs->data, rhs->data, data_size) == 0;
      } else {
        return false;
      }
    }
  }
  // merge declaration of two variables together.
  bool MergeVarDecl(const Var& lhs, const Var& rhs) {
    if (lhs.same_as(rhs)) return true;
    if (!lhs.defined() || !rhs.defined()) return false;
    if (!TypeEqual(lhs->type_annotation,
                   rhs->type_annotation)) return false;
    CHECK(!equal_map_.count(lhs))
        << "Duplicated declaration of variable " <<  lhs;
    equal_map_[lhs] = rhs;
    return true;
  }

  bool VisitExpr_(const VarNode* lhs, const Expr& other) final {
    // This function will only be triggered if we are matching free variables.
    if (const VarNode* rhs = other.as<VarNode>()) {
      if (lhs->name_hint() != rhs->name_hint()) return false;
      if (!TypeEqual(lhs->type_annotation, rhs->type_annotation)) return false;
      return LeafNodeEqual(GetRef<NodeRef>(lhs), other);
    } else {
      return false;
    }
  }

  bool VisitExpr_(const GlobalVarNode* lhs, const Expr& other) final {
    if (const GlobalVarNode* rhs = other.as<GlobalVarNode>()) {
      // use name equality for global var for now.
      return lhs->name_hint == rhs->name_hint;
    }
    return false;
  }

  bool VisitExpr_(const TupleNode* lhs, const Expr& other) final {
    if (const TupleNode* rhs = other.as<TupleNode>()) {
      if (lhs->fields.size() != rhs->fields.size()) return false;
      for (size_t i = 0; i < lhs->fields.size(); ++i) {
        if (!ExprEqual(lhs->fields[i], rhs->fields[i])) return false;
      }
      return true;
    } else {
      return false;
    }
  }

  bool VisitExpr_(const FunctionNode* lhs, const Expr& other) final {
    if (const FunctionNode* rhs = other.as<FunctionNode>()) {
      if (lhs->params.size() != rhs->params.size()) return false;
      if (lhs->type_params.size() != rhs->type_params.size()) return false;
      // map type parameter to be the same
      for (size_t i = 0; i < lhs->type_params.size(); ++i) {
        if (lhs->type_params[i]->kind != rhs->type_params[i]->kind) return false;
        equal_map_[lhs->type_params[i]] = rhs->type_params[i];
      }
      // check parameter type annotations
      for (size_t i = 0; i < lhs->params.size(); ++i) {
        if (!MergeVarDecl(lhs->params[i], rhs->params[i])) return false;
      }
      // check return types.
      if (!TypeEqual(lhs->ret_type, rhs->ret_type)) return false;
      return ExprEqual(lhs->body, rhs->body);
    } else {
      return false;
    }
  }

  bool VisitExpr_(const CallNode* lhs, const Expr& other) final {
    if (const CallNode* rhs = other.as<CallNode>()) {
      if (!ExprEqual(lhs->op, rhs->op)) return false;
      if (lhs->args.size() != rhs->args.size()) return false;
      // skip type_args check for primitive ops.
      bool is_primitive = IsPrimitiveOp(lhs->op);
      if (!is_primitive) {
        if (lhs->type_args.size() != rhs->type_args.size()) {
          return false;
        }
      }
      for (size_t i = 0; i < lhs->args.size(); ++i) {
        if (!ExprEqual(lhs->args[i], rhs->args[i])) {
          return false;
        }
      }

      if (!is_primitive) {
        for (size_t i = 0; i < lhs->type_args.size(); ++i) {
          if (!TypeEqual(lhs->type_args[i], rhs->type_args[i])) return false;
        }
      }
      return AttrEqual(lhs->attrs, rhs->attrs);
    } else {
      return false;
    }
  }

  bool VisitExpr_(const LetNode* lhs, const Expr& other) final {
    if (const LetNode* rhs = other.as<LetNode>()) {
      if (!ExprEqual(lhs->value, rhs->value)) return false;
      if (!MergeVarDecl(lhs->var, rhs->var)) return false;
      return ExprEqual(lhs->body, rhs->body);
    } else {
      return false;
    }
  }

  bool VisitExpr_(const IfNode* lhs, const Expr& other) final {
    if (const IfNode* rhs = other.as<IfNode>()) {
      return ExprEqual(lhs->cond, rhs->cond) &&
          ExprEqual(lhs->true_branch, rhs->true_branch) &&
          ExprEqual(lhs->false_branch, rhs->false_branch);
    } else {
      return false;
    }
  }

  bool VisitExpr_(const OpNode* op, const Expr& other) final {
    return op == other.get();
  }

  bool VisitExpr_(const ConstantNode* lhs, const Expr& other) final {
    if (const ConstantNode* rhs = other.as<ConstantNode>()) {
      return NDArrayEqual(lhs->data, rhs->data);
    } else {
      return false;
    }
  }

  bool VisitExpr_(const TupleGetItemNode* lhs, const Expr& other) final {
    if (const TupleGetItemNode* rhs = other.as<TupleGetItemNode>()) {
      return ExprEqual(lhs->tuple, rhs->tuple) && lhs->index == rhs->index;
    } else {
      return false;
    }
  }

  bool VisitExpr_(const RefCreateNode* op, const Expr& e2) final {
    if (const RefCreateNode* nr = e2.as<RefCreateNode>()) {
      return ExprEqual(op->value, nr->value);
    } else {
      return false;
    }
  }

  bool VisitExpr_(const RefReadNode* op, const Expr& e2) final {
    if (const RefReadNode* r = e2.as<RefReadNode>()) {
      return ExprEqual(op->ref, r->ref);
    } else {
      return false;
    }
  }

  bool VisitExpr_(const RefWriteNode* op, const Expr& e2) final {
    if (const RefWriteNode* r = e2.as<RefWriteNode>()) {
      return ExprEqual(op->ref, r->ref) && ExprEqual(op->value, r->value);
    } else {
      return false;
    }
  }

  bool VisitExpr_(const ConstructorNode* op, const Expr& e2) final {
    return GetRef<Expr>(op) == e2;
  }

  bool ClauseEqual(const Clause& l, const Clause& r) {
    return PatternEqual(l->lhs, r->lhs) && ExprEqual(l->rhs, r->rhs);
  }

  bool PatternEqual(const Pattern& l, const Pattern& r) {
    return VisitPattern(l, r);
  }

  bool VisitPattern_(const PatternWildcardNode* op, const Pattern& r) final {
    return r.as<PatternWildcardNode>();
  }

  bool VisitPattern_(const PatternVarNode* op, const Pattern& e2) final {
    if (const auto* r = e2.as<PatternVarNode>()) {
      return MergeVarDecl(op->var, r->var);
    }
    return false;
  }

  bool VisitPattern_(const PatternConstructorNode* op, const Pattern& e2) final {
    const auto* r = e2.as<PatternConstructorNode>();
    if (r == nullptr
        || !ExprEqual(op->constructor, r->constructor)
        || op->patterns.size() != r->patterns.size()) {
      return false;
    }

    for (size_t i = 0; i < op->patterns.size(); i++) {
      if (!PatternEqual(op->patterns[i], r->patterns[i])) {
        return false;
      }
    }
    return true;
  }

  bool VisitExpr_(const MatchNode* op, const Expr& e2) final {
    const MatchNode* r = e2.as<MatchNode>();

    if (r == nullptr
        || !ExprEqual(op->data, r->data)
        || op->clauses.size() != r->clauses.size()) {
      return false;
    }

    for (size_t i = 0; i < op->clauses.size(); ++i) {
      if (!ClauseEqual(op->clauses[i], r->clauses[i])) {
        return false;
      }
    }
    return true;
  }

 private:
  // whether to map open terms.
  bool map_free_var_{false};
  // renaming of NodeRef to indicate two nodes equals to each other
  std::unordered_map<NodeRef, NodeRef, NodeHash, NodeEqual> equal_map_;
};

bool AlphaEqual(const Type& lhs, const Type& rhs) {
  return AlphaEqualHandler(false).TypeEqual(lhs, rhs);
}

bool AlphaEqual(const Expr& lhs, const Expr& rhs) {
  return AlphaEqualHandler(false).ExprEqual(lhs, rhs);
}

// TODO(@jroesch): move to correct namespace?
TVM_REGISTER_API("relay._make._alpha_equal")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = AlphaEqualHandler(false).Equal(args[0], args[1]);
  });

TVM_REGISTER_API("relay._make._type_alpha_equal")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = AlphaEqualHandler(false).TypeEqual(args[0], args[1]);
  });

TVM_REGISTER_API("relay._make._graph_equal")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = AlphaEqualHandler(true).Equal(args[0], args[1]);
  });
}  // namespace relay
}  // namespace tvm
