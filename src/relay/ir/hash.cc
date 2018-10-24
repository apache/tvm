/*!
 *  Copyright (c) 2018 by Contributors
 * \file src/tvm/relay/ir/hash.cc
 * \brief Hash functions for Relay types and expressions.
 */
#include <tvm/ir_pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/relay/pass.h>
#include "type_functor.h"
#include "../../lang/attr_functor.h"

namespace tvm {
namespace relay {

// Alpha equal handler for relay.
class RelayHashHandler:
      public AttrsHashHandler,
      public TypeFunctor<size_t(const Type&)>,
      public ExprFunctor<size_t(const Expr&)> {
 public:
  explicit RelayHashHandler(bool map_free_var)
      : map_free_var_(map_free_var) {}

  /*!
   * Check equality of two nodes.
   * \param lhs The left hand operand.
   * \param rhs The right hand operand.
   * \return the compare result.
   */
  size_t Hash(const NodeRef& ref) {
    if (!ref.defined()) return ref.hash();

    if (ref->derived_from<TypeNode>()) {
      return TypeHash(Downcast<Type>(ref));
    }
    if (ref->derived_from<ExprNode>()) {
      return ExprHash(Downcast<Expr>(ref));
    }
    return AttrHash(ref);
  }

  /*!
   * Compute hash of the attributes.
   * \param ref The attributes.
   * \return the hash value
   */
  size_t AttrHash(const NodeRef& ref) {
    if (!ref.defined()) { return ref.hash(); }
    return AttrsHashHandler::Hash(ref);
  }
  /*!
   * Compute hash of a Relay type.
   * \param ref The type to hash.
   * \param rhs The right hand operand.
   * \return the hash value.
   */
  size_t TypeHash(const Type& type) {
    if (!type.defined()) { return type.hash(); }
    auto found = hash_map_.find(type);
    if (found != hash_map_.end()) {
      return found->second;
    } else {
      auto hash = this->VisitType(type);
      hash_map_.insert({type, hash});
      return hash;
    }
  }
  /*!
   * Compute the hash of an expression.
   *
   * \note We run graph structural equality checking when comparing two Exprs.
   *   This means that AlphaEqualHandler can only be used once for each pair.
   *   The equality checker checks data-flow equvalence of the Expr DAG.
   *   This function also runs faster as it memomizes equal_map.
   *
   * \param expr The expression to hash.
   * \return the hash value.
   */
  size_t ExprHash(const Expr& expr) {
    if (!expr.defined()) return expr.hash();
    auto found = hash_map_.find(expr);
    if (found != hash_map_.end()) {
      return found->second;
    } else {
      auto hash = this->VisitExpr(expr);
      hash_map_.insert({expr, hash});
      return hash;
    }
  }

 protected:
  /*!
   * \brief Check if data type equals each other.
   * \param lhs The left hand operand.
   * \param rhs The right hand operand.
   * \return the compare result.
   */
  size_t DataTypeHash(const DataType& dtype) {
     return std::hash<int>()(
         static_cast<int>(dtype.code()) |
         (static_cast<int>(dtype.bits()) << 8) |
         (static_cast<int>(dtype.lanes()) << 16));
  }

  /*!
   * \brief Check Equality of leaf node of the graph.
   *  if map_free_var_ is set to true, try to map via equal node.
   * \param lhs The left hand operand.
   * \param rhs The right hand operand.
   * \return the compare result.
   */
  size_t LeafNodeEqual(const NodeRef& lhs, const NodeRef& rhs) {
    return 0;
    // if (lhs.same_as(rhs)) return true;
    // auto it = equal_map_.find(lhs);
    // if (it != equal_map_.end()) {
    //   return it->second.same_as(rhs);
    // } else {
    //   if (map_free_var_) {
    //     if (lhs->type_index() != rhs->type_index()) return false;
    //     equal_map_[lhs] = rhs;
    //     return true;
    //   } else {
    //     return false;
    //   }
    // }
  }

  using AttrsHashHandler::VisitAttr_;
  size_t VisitAttr_(const Variable* lhs) final {
    return 0; // return LeafNodeEqual(GetRef<NodeRef>(lhs), other);
  }

  // Type equality
  size_t VisitType_(const TensorTypeNode* tensor_type) final {
    size_t hash = std::hash<std::string>()(tensor_type->_type_key);
    hash = Combine(hash, DataTypeHash(tensor_type->dtype));
    hash = Combine(hash, Hash(tensor_type->shape));
    return hash;
  }

  size_t VisitType_(const IncompleteTypeNode* incomplete) final {
    return GetRef<IncompleteType>(incomplete);
  }

  size_t VisitType_(const TypeVarNode* lhs) final {
    return 0;
    // if (const TypeVarNode* rhs = other.as<TypeVarNode>()) {
    //   if (lhs->kind != rhs->kind) return false;
    //   return LeafNodeEqual(GetRef<NodeRef>(lhs), other);
    // } else {
    //   return false;
    // }
  }

  size_t VisitType_(const FuncTypeNode* func_type) final {
    size_t hash = std::hash<std::string>()(func_type->_type_key);
    for (auto type_param : func_type->type_params) {
      hash = Combine(hash, TypeHash(type_param));
    }

    for (auto arg : func_type->arg_types) {
      hash = Combine(hash, TypeHash(arg));
    }

    hash = Combine(hash, TypeHash(func_type->ret_type));
    for (auto cs : func_type->type_constraints) {
      hash = Combine(hash, TypeHash(cs));
    }

    return hash;
  }

  size_t VisitType_(const TypeRelationNode* type_rel) final {
    return GetRef<TypeRelation>(type_rel).hash();
    // if (const TypeRelationNode* rhs = other.as<TypeRelationNode>()) {
    //   if (lhs->func->name != rhs->func->name) return false;
    //   if (lhs->num_inputs != rhs->num_inputs) return false;
    //   if (!this->AttrEqual(lhs->attrs, rhs->attrs)) return false;
    //   if (lhs->args.size() != rhs->args.size()) return false;
    //   for (size_t i = 0; i < lhs->args.size(); ++i) {
    //     if (!TypeEqual(lhs->args[i], rhs->args[i])) return false;
    //   }
    //   return true;
    // } else {
    //   return false;
    // }
  }

  size_t VisitType_(const TupleTypeNode* tuple_type) final {
    size_t hash = std::hash<std::string>()(tuple_type->_type_key);
    for (size_t i = 0; i < tuple_type->fields.size(); i++) {
      hash = Combine(hash, TypeHash(tuple_type->fields[i]));
    }
    return hash;
  }

  // Expr equal checking.
  size_t NDArrayHash(const runtime::NDArray& array) {
    return 0;
    // if (lhs.defined() != rhs.defined()) {
    //   return false;
    // } else if (lhs.same_as(rhs)) {
    //   return true;
    // } else {
    //   auto ldt = lhs->dtype;
    //   auto rdt = rhs->dtype;
    //   CHECK_EQ(lhs->ctx.device_type, kDLCPU) << "can only compare CPU tensor";
    //   CHECK_EQ(rhs->ctx.device_type, kDLCPU) << "can only compare CPU tensor";
    //   if (ldt.code == rdt.code && ldt.lanes == rdt.lanes && ldt.bits == rdt.bits) {
    //     size_t data_size = runtime::GetDataSize(*lhs.operator->());
    //     return std::memcmp(lhs->data, rhs->data, data_size) == 0;
    //   } else {
    //     return false;
    //   }
    // }
  }

  int BindVar(const NodeRef& var) {
    var_map_[var] = var_counter;
    return var_counter++;
  }

  size_t VisitExpr_(const VarNode* var) final {
    return std::hash<int>()(var_map_[GetRef<Var>(var)]);
  }

  size_t VisitExpr_(const GlobalVarNode* global) final {
    return GetRef<GlobalVar>(global).hash();
  }

  size_t VisitExpr_(const TupleNode* tuple) final {
    size_t hash = std::hash<std::string>()(tuple->_type_key);
    for (size_t i = 0; i < tuple->fields.size(); i++) {
      hash = Combine(hash, ExprHash(tuple->fields[i]));
    }
    return hash;
  }

  size_t VisitExpr_(const FunctionNode* func) final {
    size_t hash = std::hash<std::string>()(func->_type_key);
    for (auto type_param : func->type_params) {
      hash = Combine(hash, TypeHash(type_param));
    }

    for (auto param : func->params) {
      hash = Combine(hash, std::hash<int>()(BindVar(param)));
    }

    hash = Combine(hash, TypeHash(func->ret_type));
    hash =  Combine(hash, ExprHash(func->body));

    return hash;
  }

  size_t VisitExpr_(const CallNode* call) final {
    size_t hash = std::hash<std::string>()(call->_type_key);
    hash = Combine(hash, ExprHash(call->op));

    for (auto arg : call->args) {
      hash = Combine(hash, ExprHash(arg));
    }

    hash = Combine(hash, AttrHash(call->attrs));

    return hash;
  }

  size_t VisitExpr_(const LetNode* let) final {
    size_t hash = std::hash<std::string>()(let->_type_key);
    hash = Combine(hash, std::hash<int>()(BindVar(let->var)));
    hash = Combine(hash, ExprHash(let->value));
    hash = Combine(hash, ExprHash(let->body));
    return hash;
  }

  size_t VisitExpr_(const IfNode* ite) final {
    size_t hash = std::hash<std::string>()(ite->_type_key);
    hash = Combine(hash, ExprHash(ite->cond));
    hash = Combine(hash, ExprHash(ite->true_branch));
    hash = Combine(hash, ExprHash(ite->false_branch));
    return hash;
  }

  size_t VisitExpr_(const OpNode* op) final {
    return GetRef<Op>(op).hash();
  }

  size_t VisitExpr_(const ConstantNode* rconst) final {
    return NDArrayHash(rconst->data);
  }

  size_t VisitExpr_(const TupleGetItemNode* get_item) final {
    size_t hash = std::hash<std::string>()(get_item->_type_key);
    hash = Combine(hash, ExprHash(get_item->tuple));
    hash = Combine(hash, std::hash<int>()(get_item->index));
    return hash;
  }

 private:
  // whether to map open terms.
  bool map_free_var_{false};
  // renaming of NodeRef to indicate two nodes equals to each other
  std::unordered_map<NodeRef, size_t, NodeHash, NodeEqual> hash_map_;
  std::unordered_map<NodeRef, int, NodeHash, NodeEqual> var_map_;
  int var_counter = 0;
};

size_t HashType(const Type& type) {
  return RelayHashHandler(false).TypeHash(type);
}

size_t HashExpr(const Expr& expr) {
  return RelayHashHandler(false).ExprHash(expr);
}

// TODO(@jroesch): move to correct namespace?
TVM_REGISTER_API("relay._ir_pass._expr_hash")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = static_cast<int64_t>(RelayHashHandler(false).Hash(args[0]));
  });

TVM_REGISTER_API("relay._ir_pass._type_hash")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = static_cast<int64_t>(RelayHashHandler(false).TypeHash(args[0]));
  });

// TVM_REGISTER_API("relay._make._graph_equal")
// .set_body([](TVMArgs args, TVMRetValue* ret) {
//     *ret = AlphaEqualHandler(true).Equal(args[0], args[1]);
//   });

}  // namespace relay
}  // namespace tvm
