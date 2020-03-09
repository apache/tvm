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
 * \file tvm/relay/pass/subgraph_set.h
 * \brief Define data structures to extract and manipulate subgraphs from
 * a relay function. Subgraphs are denoted by subgraph_begin and subgraph_end
 * annotations that exist on all the input and output edges of the subgraph.
 */

#ifndef TVM_RELAY_ANALYSIS_SUBGRAPH_SET_H_
#define TVM_RELAY_ANALYSIS_SUBGRAPH_SET_H_

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/ir/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <list>

namespace tvm {
namespace relay {

struct Subgraph;
class SubgraphSet;

struct SubgraphNode : public Object {
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("id", &id);
    Array<Expr> nodes_array(nodes.begin(), nodes.end());
    v->Visit("nodes", &nodes_array);
    Array<Expr> args_array(args.begin(), args.end());
    v->Visit("args", &args_array);
    Array<Expr> rets_array(rets.begin(), rets.end());
    v->Visit("rets", &rets_array);
  }

  /*! \brief The subgraph ID. */
  int id{-1};

  /*! \brief The input arguments to this subgraph. */
  std::list<Expr> args;

  /*! \brief The return values of this subgraph */
  std::list<Expr> rets;

  /*! \brief Nodes in this subgraph. */
  std::unordered_set<Expr, ObjectHash, ObjectEqual> nodes;

  static constexpr const char* _type_key = "relay.Subgraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(SubgraphNode, Object);
};

/*!
 * \brief An object to hold the properties of a subgraph as used by the
 * SubgraphSet class.
*/
struct Subgraph : public ObjectRef {
  Subgraph() {
    auto n = make_object<SubgraphNode>();
    data_ = std::move(n);
  }

  /*!
 * \brief Construct from an object pointer.
 * \param n The object pointer.
 */
  explicit Subgraph(ObjectPtr<Object> n) : ObjectRef(n) {}

  /*! \return Mutable pointers to the node. */
  SubgraphNode* operator->() const {
    auto* ptr = get_mutable();
    CHECK(ptr != nullptr);
    return static_cast<SubgraphNode*>(ptr);
  }
};

class SubgraphSetNode : public Object {
  using UnorderedSubgraphSet =
  std::unordered_set<Subgraph, ObjectHash, ObjectEqual>;
  // Create iterator alias for a CallGraph object.
  using iterator = UnorderedSubgraphSet::iterator;
  using const_iterator = UnorderedSubgraphSet::const_iterator;

 public:
  /*! \brief Default constructor. */
  SubgraphSetNode() = default;

  /*! \return The begin iterator */
  iterator begin() {
    return subgraphs_.begin();
  }
  /*! \return The end iterator */
  iterator end() {
    return subgraphs_.end();
  }
  /*! \return The const begin iterator */
  const_iterator begin() const {
    return subgraphs_.begin();
  }
  /*! \return The const end iterator */
  const_iterator end() const {
    return subgraphs_.end();
  }

  /*!
   * \brief Get the subgraph that an expression belongs to.
   *
   * \param expr Which expr to get the subgraph for.
   *
   * \return A pointer to the subgraph, nullptr if the expression
   * doesn't belong to a subgraph.
   */
  Subgraph GetSubgraph(const Expr& expr) const;

  /*!
   * \brief Merge subgraph 1 into subgraph 2.
   *
   * \param subgraph1 A subgraph to merge.
   * \param subgraph2 A subgraph to merge.
   */
  void MergeSubgraph(Subgraph subgraph1, Subgraph subgraph2);

  /*!
   * \brief Add an expression to a subgraph.
   *
   * \param subgraph The subgraph to add the expression to.
   * \param expr The expression.
   */
  void AddToSubgraph(Subgraph subgraph, const Expr& expr);

  /*!
   * \brief Make a new subgraph.
   *
   * \return The new subgraph.
   */
  Subgraph MakeSubgraph();

  void VisitAttrs(AttrVisitor* v) {
    Array<Subgraph> subgraphs_array(subgraphs_.begin(), subgraphs_.end());
    v->Visit("subgraphs", &subgraphs_array);
  }

  static constexpr const char* _type_key = "relay.SubgraphSet";
  TVM_DECLARE_FINAL_OBJECT_INFO(SubgraphSetNode, Object);

 private:
  std::unordered_set<Subgraph, ObjectHash, ObjectEqual> subgraphs_;
};

/*!
 * \brief A class to hold a set of subgraphs produced from a relay expression
 * that contains 'subgraph_begin' and 'subgraph_end' style annotations. The
 * subgraphs should be disjoint. The class provides both a method to construct
 * the subgraph set of a given relay expression as well as additional methods
 * to update and query subgraphs.
 */
class SubgraphSet : public ObjectRef {
  using UnorderedSubgraphSet =
    std::unordered_set<Subgraph, ObjectHash, ObjectEqual>;
  // Create iterator alias for a CallGraph object.
  using iterator = UnorderedSubgraphSet::iterator;
  using const_iterator = UnorderedSubgraphSet::const_iterator;

 public:
  SubgraphSet() {
    auto n = make_object<SubgraphSetNode>();
    data_ = std::move(n);
  }

  /*!
 * \brief Construct from an object pointer.
 *
 * \param n The object pointer.
 */
  explicit SubgraphSet(ObjectPtr<Object> n) : ObjectRef(n) {}

  /*! \return The begin iterator. */
  iterator begin() {
    auto* n = operator->();
    CHECK(n);
    return n->begin();
  }
  /*! \return The end iterator. */
  iterator end() {
    auto* n = operator->();
    CHECK(n);
    return n->end();
  }
  /*! \return The begin iterator. */
  const_iterator begin() const {
    const auto* n = operator->();
    CHECK(n);
    return n->begin();
  }
  /*! \return The end iterator. */
  const_iterator end() const {
    const auto *n = operator->();
    CHECK(n);
    return n->end();
  }

    /*! \return mutable pointers to the node. */
  SubgraphSetNode* operator->() const {
    auto* ptr = get_mutable();
    CHECK(ptr != nullptr);
    return static_cast<SubgraphSetNode*>(ptr);
  }

  /*! \brief Create a SubgraphSet from a relay expression.
   *
   * \param expr The relay expr from which to construct the set.
   * \param begin Subgraph begin annotation operator.
   * \param end Subgraph end annotation operator.
   *
   * \return The created SubgraphSet for the expression.
   */
  static SubgraphSet Create(const Expr& expr,
                            const Op& begin,
                            const Op& end);

 private:
  /*! \brief Helper class to construct a SubgraphSet from an expr.*/
  class Creator;
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_ANALYSIS_SUBGRAPH_SET_H_
