/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2018 by Contributors
 * \file src/tvm/relay/ir/hash.cc
 * \brief Hash functions for Relay types and expressions.
 */
#include <tvm/ir_pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/relay/analysis.h>
#include <tvm/attrs.h>
#include "type_functor.h"
#include "../../lang/attr_functor.h"

namespace tvm {
namespace relay {

// Hash handler for Relay.
class RelayHashHandler:
    public AttrsHashHandler,
    public TypeFunctor<size_t(const Type&)>,
    public ExprFunctor<size_t(const Expr&)>,
    public PatternFunctor<size_t(const Pattern&)> {
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

  size_t VisitType_(const RefTypeNode* rtn) final {
    size_t hash = std::hash<std::string>()(RefTypeNode::_type_key);
    hash = Combine(hash, TypeHash(rtn->value));
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
    if (auto var_node = var.as<VarNode>()) {
      hash = Combine(hash, TypeHash(var_node->type_annotation));
    }
    hash_map_[var] = hash;

    const auto* ty_param = var.as<TypeVarNode>();
    if (ty_param && ty_param->kind == Kind::kShapeVar) {
      hash_map_[ty_param->var] = hash;
    }
    return hash;
  }

  size_t VisitExpr_(const VarNode* var) final {
    // hash free variable
    size_t name_hash = std::hash<const Node*>()(var->vid.get());
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
    hash = Combine(hash, ExprHash(func->body));

    return hash;
  }

  size_t VisitExpr_(const CallNode* call) final {
    size_t hash = std::hash<std::string>()(CallNode::_type_key);
    hash = Combine(hash, ExprHash(call->op));

    for (auto arg : call->args) {
      hash = Combine(hash, ExprHash(arg));
    }

    for (auto t : call->type_args) {
      CHECK(t.defined());
      hash = Combine(hash, TypeHash(t));
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

  size_t VisitExpr_(const RefCreateNode* rn) final {
    size_t hash = std::hash<std::string>()(RefCreateNode::_type_key);
    hash = Combine(hash, ExprHash(rn->value));
    return hash;
  }

  size_t VisitExpr_(const RefReadNode* rn) final {
    size_t hash = std::hash<std::string>()(RefReadNode::_type_key);
    hash = Combine(hash, ExprHash(rn->ref));
    return hash;
  }

  size_t VisitExpr_(const RefWriteNode* rn) final {
    size_t hash = std::hash<std::string>()(RefWriteNode::_type_key);
    hash = Combine(hash, ExprHash(rn->ref));
    hash = Combine(hash, ExprHash(rn->value));
    return hash;
  }

  size_t VisitExpr_(const MatchNode* mn) final {
    size_t hash = std::hash<std::string>()(MatchNode::_type_key);
    hash = Combine(hash, ExprHash(mn->data));
    for (const auto& c : mn->clauses) {
      hash = Combine(hash, PatternHash(c->lhs));
      hash = Combine(hash, ExprHash(c->rhs));
    }
    return hash;
  }

  size_t VisitExpr_(const ConstructorNode* cn) final {
    size_t hash = std::hash<std::string>()(ConstructorNode::_type_key);
    hash = Combine(hash, std::hash<std::string>()(cn->name_hint));
    return hash;
  }

  size_t VisitType_(const TypeCallNode* tcn) final {
    size_t hash = std::hash<std::string>()(TypeCallNode::_type_key);
    hash = Combine(hash, TypeHash(tcn->func));
    for (const auto& t : tcn->args) {
      hash = Combine(hash, TypeHash(t));
    }
    return hash;
  }

  size_t VisitType_(const TypeDataNode* tdn) final {
    size_t hash = std::hash<std::string>()(TypeDataNode::_type_key);
    hash = Combine(hash, TypeHash(tdn->header));
    for (const auto& tv : tdn->type_vars) {
      hash = Combine(hash, TypeHash(tv));
    }
    for (const auto& cn : tdn->constructors) {
      hash = Combine(hash, ExprHash(cn));
    }
    return hash;
  }

  size_t VisitType_(const GlobalTypeVarNode* tvn) final {
    return BindVar(GetRef<GlobalTypeVar>(tvn));
  }

  size_t PatternHash(const Pattern& p) {
    return VisitPattern(p);
  }

  size_t VisitPattern_(const PatternConstructorNode* pcn) final {
    size_t hash = std::hash<std::string>()(PatternConstructorNode::_type_key);
    hash = Combine(hash, ExprHash(pcn->constructor));
    for (const auto& p : pcn->patterns) {
      hash = Combine(hash, PatternHash(p));
    }
    return hash;
  }

  size_t VisitPattern_(const PatternVarNode* pvn) final {
    size_t hash = std::hash<std::string>()(PatternVarNode::_type_key);
    hash = Combine(hash, BindVar(pvn->var));
    return hash;
  }

  size_t VisitPattern_(const PatternWildcardNode* pwn) final {
    size_t hash = std::hash<std::string>()(PatternWildcardNode::_type_key);
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

TVM_REGISTER_API("relay._analysis._expr_hash")
.set_body_typed<int64_t(NodeRef)>([](NodeRef ref) {
  return static_cast<int64_t>(RelayHashHandler().Hash(ref));
});

TVM_REGISTER_API("relay._analysis._type_hash")
.set_body_typed<int64_t(Type)>([](Type type) {
  return static_cast<int64_t>(RelayHashHandler().TypeHash(type));
});

}  // namespace relay
}  // namespace tvm
