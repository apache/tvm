/*!
 *  Copyright (c) 2018 by Contributors
 * \file src/tvm/relay/ir/hash.cc
 * \brief Hash functions for Relay types and expressions.
 */
#include <tvm/ir_pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/relay/pass.h>
#include <tvm/attrs.h>
#include "type_functor.h"
#include "../../lang/attr_functor.h"

namespace tvm {
namespace relay {

// Hash handler for Relay.
class RelayHashHandler:
      public AttrsHashHandler,
      public TypeFunctor<size_t(const Type&)>,
      public ExprFunctor<size_t(const Expr&)> {
 public:
  explicit RelayHashHandler() {}

  /*!
   * Compute hash of a node.
   * \param ref The node to hash.
   * \return the hash value.
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
   * \brief Hash a DataType.
   * \param dtype The dtype to hash.
   * \return the hash value.
   */
  size_t DataTypeHash(const DataType& dtype) {
    return ::tvm::AttrsHash()(dtype);
  }

  using AttrsHashHandler::VisitAttr_;
  size_t VisitAttr_(const Variable* var) final {
    size_t hash = std::hash<std::string>()(Variable::_type_key);
    auto it = hash_map_.find(GetRef<VarExpr>(var));
    if (it != hash_map_.end()) {
      return it->second;
    }
    return Combine(hash, std::hash<std::string>()(var->name_hint));
  }

  // Type hashing
  size_t VisitType_(const TensorTypeNode* tensor_type) final {
    size_t hash = std::hash<std::string>()(TensorTypeNode::_type_key);
    hash = Combine(hash, DataTypeHash(tensor_type->dtype));
    hash = Combine(hash, Hash(tensor_type->shape));
    return hash;
  }

  size_t VisitType_(const IncompleteTypeNode* incomplete) final {
    size_t hash = std::hash<std::string>()(IncompleteTypeNode::_type_key);
    return Combine(hash, std::hash<int>()(incomplete->kind));
  }

  size_t VisitType_(const TypeVarNode* tyvar) final {
    /*
      TypeVar/Var/Variable have two locations where they are hashed:

        The declaration site of a function, let, or function type.
        The first occurence in the term.

      We will only reach this code if the TypeVar itself is unbound, we assign
      a free variable index to it, meaning this hashing function implements
      structural equality for both open (i.e graph equality) and closed terms
      (i.e alpha_equality).
    */
    return BindVar(GetRef<TypeVar>(tyvar));
  }

  size_t VisitType_(const FuncTypeNode* func_type) final {
    size_t hash = std::hash<std::string>()(FuncTypeNode::_type_key);

    for (auto type_param : func_type->type_params) {
      hash = Combine(hash, BindVar(type_param));
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
    size_t hash = std::hash<std::string>()(TypeRelationNode::_type_key);
    hash = Combine(hash, std::hash<std::string>()(type_rel->func->name));
    hash = Combine(hash, AttrHash(type_rel->attrs));

    for (auto arg : type_rel->args) {
      hash = Combine(hash, TypeHash(arg));
    }

    return hash;
  }

  size_t VisitType_(const TupleTypeNode* tuple_type) final {
    size_t hash = std::hash<std::string>()(TupleTypeNode::_type_key);
    for (size_t i = 0; i < tuple_type->fields.size(); i++) {
      hash = Combine(hash, TypeHash(tuple_type->fields[i]));
    }
    return hash;
  }

  // Expr hashing.
  size_t NDArrayHash(const runtime::NDArray& array) {
    size_t hash = std::hash<uint8_t>()(array->dtype.code);
    hash = Combine(hash, std::hash<uint8_t>()(array->dtype.bits));
    hash = Combine(hash, std::hash<uint16_t>()(array->dtype.lanes));
    CHECK_EQ(array->ctx.device_type, kDLCPU) << "can only compare CPU tensor";
    size_t data_size = runtime::GetDataSize(*array.operator->());
    uint8_t * data = reinterpret_cast<uint8_t*>(array->data);
    for (size_t i = 0; i < data_size; i++) {
      hash = Combine(hash, std::hash<uint8_t>()(data[i]));
    }
    return hash;
  }

  size_t BindVar(const NodeRef& var) {
    size_t hash = std::hash<int>()(var_counter++);
    CHECK_EQ(hash_map_.count(var), 0);
    hash_map_[var] = hash;

    const auto* ty_param = var.as<TypeVarNode>();
    if (ty_param && ty_param->kind == TypeVarNode::Kind::kShapeVar) {
      hash_map_[ty_param->var] = hash;
    }
    return hash;
  }

  size_t VisitExpr_(const VarNode* var) final {
    size_t name_hash = std::hash<std::string>()(var->name_hint);
    return Combine(name_hash, TypeHash(var->type_annotation));
  }

  size_t VisitExpr_(const GlobalVarNode* global) final {
    return std::hash<std::string>()(global->name_hint);
  }

  size_t VisitExpr_(const TupleNode* tuple) final {
    size_t hash = std::hash<std::string>()(TupleNode::_type_key);
    for (size_t i = 0; i < tuple->fields.size(); i++) {
      hash = Combine(hash, ExprHash(tuple->fields[i]));
    }
    return hash;
  }

  size_t VisitExpr_(const FunctionNode* func) final {
    size_t hash = std::hash<std::string>()(FunctionNode::_type_key);
    for (auto type_param : func->type_params) {
      hash = Combine(hash, BindVar(type_param));
    }

    for (auto param : func->params) {
      hash = Combine(hash, BindVar(param));
    }

    hash = Combine(hash, TypeHash(func->ret_type));
    hash =  Combine(hash, ExprHash(func->body));

    return hash;
  }

  size_t VisitExpr_(const CallNode* call) final {
    size_t hash = std::hash<std::string>()(CallNode::_type_key);
    hash = Combine(hash, ExprHash(call->op));

    for (auto arg : call->args) {
      hash = Combine(hash, ExprHash(arg));
    }

    hash = Combine(hash, AttrHash(call->attrs));

    return hash;
  }

  size_t VisitExpr_(const LetNode* let) final {
    size_t hash = std::hash<std::string>()(LetNode::_type_key);
    hash = Combine(hash, BindVar(let->var));
    hash = Combine(hash, ExprHash(let->value));
    hash = Combine(hash, ExprHash(let->body));
    return hash;
  }

  size_t VisitExpr_(const IfNode* ite) final {
    size_t key = std::hash<std::string>()(IfNode::_type_key);
    size_t hash = key;
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
    size_t hash = std::hash<std::string>()(TupleGetItemNode::_type_key);
    hash = Combine(hash, ExprHash(get_item->tuple));
    hash = Combine(hash, std::hash<int>()(get_item->index));
    return hash;
  }

 private:
  // renaming of NodeRef to indicate two nodes equals to each other
  std::unordered_map<NodeRef, size_t, NodeHash, NodeEqual> hash_map_;
  int var_counter = 0;
};

size_t StructuralHash::operator()(const Type& type) const {
  return RelayHashHandler().TypeHash(type);
}

size_t StructuralHash::operator()(const Expr& expr) const {
  return RelayHashHandler().ExprHash(expr);
}

TVM_REGISTER_API("relay._ir_pass._expr_hash")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = static_cast<int64_t>(RelayHashHandler().Hash(args[0]));
  });

TVM_REGISTER_API("relay._ir_pass._type_hash")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = static_cast<int64_t>(RelayHashHandler().TypeHash(args[0]));
  });

}  // namespace relay
}  // namespace tvm
