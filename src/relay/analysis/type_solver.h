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
 * \file type_solver.h
 * \brief Solver logic for type inference.
 */
#ifndef TVM_RELAY_ANALYSIS_TYPE_SOLVER_H_
#define TVM_RELAY_ANALYSIS_TYPE_SOLVER_H_

#include <tvm/relay/analysis.h>
#include <tvm/relay/error.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../support/arena.h"

namespace tvm {
namespace relay {

using support::LinkedList;
using support::LinkNode;

/*!
 * \brief Interface of type solver used in type inference.
 *
 * TypeSolver works on a list of constraints among incomplete types.
 * The user will populate the constraints by AddConstraint and Assign.
 * Then we can call Solve to trying to resolve the unknown.
 *
 * This can be viewed as "type program(computational graph)" of types, where
 * the type constraint are operators of the graph and the incomplete
 * types are intermediate value of the graph.
 * If all the input types are concretely known, we should be able to
 * just run a forward pass on the "type program" to get all the types.
 *
 * The list of constraints representation means we are storing it as a bipartite
 * graph instead of a DAG. This is because some constraints might go both direction.
 * TypeSolver could take advantage of bidirectional constraints to deduce input
 * value given output ones. Never-the-less, we should keep in mind that
 * there is a "forward direction" that the TypeSolver should take advantage of.
 */
class TypeSolver {
 public:
  TypeSolver(const GlobalVar& current_func, DiagnosticContext diag_ctx);
  ~TypeSolver();
  /*!
   * \brief Add a type constraint to the solver.
   * \param constraint The constraint to be added.
   * \param location The location at which the constraint was incurred.
   */
  void AddConstraint(const TypeConstraint& constraint, const Span& span);
  /*!
   * \brief Resolve type to the solution type in the solver.
   * \param type The type to be resolved.
   * \return The resolved type.
   */
  Type Resolve(const Type& type);
  /*!
   * \brief Start to solve the types using the current known information.
   * \return Whether all the incomplete types has been fully resolved.
   */
  bool Solve();
  /*!
   * \brief Unify lhs and rhs.
   * \param lhs The left operand.
   * \param rhs The right operand
   * \param location The location at which the unification problem arose.
   */
  Type Unify(const Type& lhs, const Type& rhs, const Span& span, bool assign_lhs = true,
             bool assign_rhs = true);
  /*!
   * \brief Report a diagnostic.
   * \param diag The diagnostic to report.
   */
  void Emit(const Diagnostic& diag) { diag_ctx_.Emit(diag); }

 private:
  class AnyChecker;
  class OccursChecker;
  class Unifier;
  class Resolver;
  class Propagator;
  class Merger;
  class Reporter;
  struct TypeNode;
  struct RelationNode;
  // Internally the solver maintains a bipartite graph of Relation and Types.
  // All the object in the structure is managed by a arena allocator
  // which releases the memory upon distruction of the type solver.
  /*!
   * \brief type node struct
   *  TypeNode implements a union-find data structure(via parent)
   *  that can unifies the same types to the name resolved_type.
   *
   *  It also contains collection of links to related Relations,
   *  which is stored in rel_set.
   */
  struct TypeNode {
    /*! \brief The final resolved type */
    Type resolved_type;
    /*! \brief type node in the union find algorithm */
    TypeNode* parent{nullptr};
    /*! \brief set of relations that is related to this type node */
    std::unordered_set<RelationNode*> rel_set;

    /*!
     * \brief Find the root type node, perform path compression
     * \return The root type node.
     */
    TypeNode* FindRoot() {
      // fast path
      if (this->parent == nullptr) return this;
      // slow path with path compression.
      TypeNode* root = this;
      while (root->parent != nullptr) {
        root = root->parent;
      }
      for (TypeNode* p = this; p != root;) {
        TypeNode* parent = p->parent;
        p->parent = root;
        p = parent;
      }
      return root;
    }
  };

  /*! \brief relation node */
  struct RelationNode {
    /*! \brief Whether the relation is in the queue to be solved */
    bool inqueue{false};
    /*! \brief Whether the relation is resolved */
    bool resolved{false};
    /*! \brief The corresponding type relation */
    TypeRelation rel;
    /*! \brief list types to this relation */
    LinkedList<TypeNode*> type_list;
    /*! \brief The location this type relation originated from. */
    Span span;
  };

  /*! \brief A simple union find between shapes. */
  tvm::Map<IndexExpr, IndexExpr> shape_uf_;
  /*! \brief List of all allocated type nodes */
  std::vector<TypeNode*> type_nodes_;
  /*! \brief List of all allocated relation nodes */
  std::vector<RelationNode*> rel_nodes_;
  /*! \brief Number of resolved relations */
  size_t num_resolved_rels_{0};
  /*! \brief map from types to type nodes. */
  std::unordered_map<Type, TypeNode*, ObjectPtrHash, ObjectPtrEqual> tmap_;
  /*! \brief Internal queue to update the relation */
  std::queue<RelationNode*> update_queue_;
  /*! \brief allocator of all the internal node obhect*/
  support::Arena arena_;
  /*! \brief Reporter that reports back to self */
  TypeReporter reporter_;
  /*! \brief The global representing the current function. */
  GlobalVar current_func_;
  /*! \brief The diagnostic context. */
  DiagnosticContext diag_ctx_;
  /*! \brief The module. */
  IRModule module_;

  /*!
   * \brief GetTypeNode that is corresponds to t.
   * if it do not exist, create a new one.
   * \return The type node.
   */
  TypeNode* GetTypeNode(const Type& t) {
    auto it = tmap_.find(t);
    if (it != tmap_.end()) {
      return it->second->FindRoot();
    } else {
      TypeNode* n = arena_.make<TypeNode>();
      type_nodes_.push_back(n);
      n->resolved_type = t;
      tmap_[t] = n;
      return n;
    }
  }
  /*!
   * \brief Add relation node rel to the update queue
   * \param rel The relation node
   */
  void AddToQueue(RelationNode* rel) {
    if (rel->inqueue) return;
    ICHECK(!rel->resolved);
    rel->inqueue = true;
    update_queue_.push(rel);
  }

  /*!
   * \brief Merge rhs type node to lhs
   * \param src The source operand
   * \param dst The dst operand.
   */
  void MergeFromTo(TypeNode* src, TypeNode* dst);
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ANALYSIS_TYPE_SOLVER_H_
