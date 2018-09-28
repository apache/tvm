/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_solver.h
 * \brief Solver logic for type inference.
 */
#ifndef TVM_RELAY_PASS_TYPE_SOLVER_H_
#define TVM_RELAY_PASS_TYPE_SOLVER_H_

#include <tvm/relay/type.h>
#include <tvm/relay/pass.h>
#include <vector>
#include <queue>
#include "../../common/arena.h"


namespace tvm {
namespace relay {

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
  TypeSolver();
  ~TypeSolver();
  /*!
   * \brief Add a type constraint to the solver.
   * \param constraint The constraint to be added.
   */
  void AddConstraint(const TypeConstraint& constraint);
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
   */
  Type Unify(const Type& lhs, const Type& rhs);

 private:
  class Reporter;
  struct TypeNode;
  struct RelationNode;
  // Internally the solver maintains a bipartite graph of Relation and Types.
  // All the object in the structure is managed by a arena allocator
  // which releases the memory upon distruction of the type solver.
  /*!
   * \brief Link list node
   * \tparam T the content data type
   */
  template<typename T>
  struct LinkNode {
    /*! \brief The content value */
    T value;
    /*! \brief pointer to the next location */
    LinkNode<T>* next{nullptr};
  };
  /*!
   * \brief LinkedList structure
   * \tparam T the content data type
   */
  template<typename T>
  struct LinkedList {
    /*! \brief Head pointer */
    LinkNode<T>* head{nullptr};
    /*! \brief Tail pointer */
    LinkNode<T>* tail{nullptr};
    /*!
     * \brief Push a new node to the end of the linked list.
     * \param node The node to be pushed.
     */
    void Push(LinkNode<T>* node) {
      node->next = nullptr;
      if (this->tail != nullptr) {
        this->tail->next = node;
        this->tail = node;
      } else {
        head = tail = node;
      }
    }
  };
  /*!
   * \brief type node struct
   *  TypeNode implements a union-find data structure(via parent)
   *  that can unifies the same types to the name resolved_type.
   *
   *  It also contains collection of links to related Relations,
   *  which is stored in rel_list.
   */
  struct TypeNode {
    /*! \brief The final resolved type */
    Type resolved_type;
    /*! \brief type node in the union find algorithm */
    TypeNode* parent{nullptr};
    /*! \brief list of relations that is related to this type node */
    LinkedList<RelationNode*> rel_list;
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
  };
  /*! \brief List of all allocated type nodes */
  std::vector<TypeNode*> type_nodes_;
  /*! \brief List of all allocated relation nodes */
  std::vector<RelationNode*> rel_nodes_;
  /*! \brief Number of resolved relations */
  size_t num_resolved_rels_{0};
  /*! \brief map from type node to types. */
  std::unordered_map<Type, TypeNode*, NodeHash, NodeEqual> tmap_;
  /*! \breif Internal queue to update the relation */
  std::queue<RelationNode*> update_queue_;
  /*! \brief allocator of all the internal node obhect*/
  common::Arena arena_;
  /*! \brief Reporter that reports back to self */
  TypeReporter reporter_;
  /*!
   * \brief Create function to create a new node ptr via arena
   * \tparam The type parameter
   * \return The node pointer.
   */
  template<typename T>
  T* make() {
    T* ptr = arena_.Alloc<T>();
    // call constructor
    new (ptr) T();
    return ptr;
  }
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
      TypeNode* n = make<TypeNode>();
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
    CHECK(!rel->resolved);
    rel->inqueue = true;
    update_queue_.push(rel);
  }
  /*!
   * \brief Merge rhs type node to lhs
   * \param src The source operand
   * \param dst The dst operand.
   */
  void MergeFromTo(TypeNode* src, TypeNode* dst) {
    if (src == dst) return;
    src->parent = dst;
    // move the link to the to dst
    for (auto* rlink = src->rel_list.head; rlink != nullptr;) {
      // store next pointer first before rlink get moved
      auto* next = rlink->next;
      // if the relation is not yet resolved
      // send the relation to the new
      if (!rlink->value->resolved) {
        this->AddToQueue(rlink->value);
        dst->rel_list.Push(rlink);
      }
      rlink = next;
    }
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PASS_TYPE_SOLVER_H_
