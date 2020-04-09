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
 * \file tvm/relay/pass/annotated_region_set.h
 * \brief Define data structures to extract and manipulate regions from
 * a relay function. Regions are denoted by region_begin and region_end
 * annotations that exist on all the input and output edges of the region.
 */

#ifndef TVM_RELAY_ANALYSIS_ANNOTATED_REGION_SET_H_
#define TVM_RELAY_ANALYSIS_ANNOTATED_REGION_SET_H_

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/ir/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/container.h>
#include <tvm/relay/transform.h>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <list>

namespace tvm {
namespace relay {

class AnnotatedRegion;
class AnnotatedRegionSet;

class AnnotatedRegionNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("id", &id_);
    v->Visit("target", &target_);
    Array<Expr> nodes_array(nodes_.begin(), nodes_.end());
    v->Visit("nodes", &nodes_array);
    Array<Expr> args_array(ins_.begin(), ins_.end());
    v->Visit("args", &args_array);
    Array<Expr> rets_array(outs_.begin(), outs_.end());
    v->Visit("rets", &rets_array);
  }

  /*! \brief Get the region ID. */
  int GetID() const {
    return id_;
  }

  /*! \brief Get the region target. */
  std::string GetTarget() const {
    return target_;
  }

  /*! \brief Get the region's inputs. */
  std::list<Expr> GetInputs() const {
    return ins_;
  }

  /*! \brief Get the region's outputs. */
  std::list<Expr> GetOutputs() const {
    return outs_;
  }

  /*! \brief Get the region's nodes. */
  std::unordered_set<Expr, ObjectHash, ObjectEqual> GetNodes() const {
    return nodes_;
  }

  static constexpr const char* _type_key = "relay.AnnotatedRegion";
  TVM_DECLARE_FINAL_OBJECT_INFO(AnnotatedRegionNode, Object);

 protected:
  /*! \brief The region ID. */
  int id_{-1};
  /*! \brief The target for this region. */
  std::string target_ = "default";
  /*! \brief The inputs to this region. */
  std::list<Expr> ins_;
  /*! \brief The outputs of this region */
  std::list<Expr> outs_;
  /*! \brief Nodes in this region. */
  std::unordered_set<Expr, ObjectHash, ObjectEqual> nodes_;

  friend class AnnotatedRegionSet;
  friend class AnnotatedRegionSetNode;
};

/*!
 * \brief An object to hold the properties of a region as used by the
 * AnnotatedRegionSet class. This should be considered read-only.
*/
class AnnotatedRegion : public ObjectRef {
 public:
  AnnotatedRegion() {
    auto n = make_object<AnnotatedRegionNode>();
    data_ = std::move(n);
  }

  /*!
 * \brief Construct from an object pointer.
 * \param n The object pointer.
 */
  explicit AnnotatedRegion(ObjectPtr<Object> n) : ObjectRef(n) {}

  /*! \return Mutable pointers to the node. */
  AnnotatedRegionNode* operator->() const {
    auto* ptr = get_mutable();
    CHECK(ptr != nullptr);
    return static_cast<AnnotatedRegionNode*>(ptr);
  }
};

class AnnotatedRegionSetNode : public Object {
  using UnorderedRegionSet =
  std::unordered_set<AnnotatedRegion, ObjectHash, ObjectEqual>;
  // Create iterator alias for a RegionSet object.
  using iterator = UnorderedRegionSet::iterator;
  using const_iterator = UnorderedRegionSet::const_iterator;

 public:
  /*! \brief Default constructor. */
  AnnotatedRegionSetNode() = default;

  /*! \return The begin iterator */
  iterator begin() {
    return regions_.begin();
  }
  /*! \return The end iterator */
  iterator end() {
    return regions_.end();
  }
  /*! \return The const begin iterator */
  const_iterator begin() const {
    return regions_.begin();
  }
  /*! \return The const end iterator */
  const_iterator end() const {
    return regions_.end();
  }

  /*!
   * \brief Get the region that an expression belongs to.
   *
   * \param expr Which expr to get the region for.
   *
   * \return A pointer to the region, nullptr if the expression
   * doesn't belong to a region.
   */
  AnnotatedRegion GetRegion(const Expr& expr) const;

  /*!
 * \brief Merge src region into dest region.
 *
 * \param src The region to merge - will be erased.
 * \param dest The region into which src will be merged.
 */
  void MergeRegions(AnnotatedRegion src, AnnotatedRegion dest);

  void VisitAttrs(AttrVisitor* v) {
    Array<AnnotatedRegion> regions_array(regions_.begin(), regions_.end());
    v->Visit("regions", &regions_array);
  }

  static constexpr const char* _type_key = "relay.AnnotatedRegionSet";
  TVM_DECLARE_FINAL_OBJECT_INFO(AnnotatedRegionSetNode, Object);

 private:
  /*!
   * \brief Add an expression to a region.
   *
   * \param dest The region to add the expression to.
   * \param expr The expression.
   */
  void AddToRegion(AnnotatedRegion dest, const Expr& expr);

  /*!
   * \brief Make a new region for a target.
   *
   * \return The new region.
   */
  AnnotatedRegion MakeRegion(const std::string& target);

  std::unordered_set<AnnotatedRegion, ObjectHash, ObjectEqual> regions_;
  /*! \brief The next region ID to assign. */
  int region_id_{0};

  friend class AnnotatedRegionSet;
};

/*!
 * \brief A class to hold a set of regions produced from a relay expression
 * that contains 'region_begin' and 'region_end' style annotations. The
 * regions should be disjoint. The class provides both a method to construct
 * the region set of a given relay expression as well as additional methods
 * to update and query regions.
 */
class AnnotatedRegionSet : public ObjectRef {
  using UnorderedRegionSet =
    std::unordered_set<AnnotatedRegion, ObjectHash, ObjectEqual>;
  // Create iterator alias for a RegionSet object.
  using iterator = UnorderedRegionSet::iterator;
  using const_iterator = UnorderedRegionSet::const_iterator;

 public:
  AnnotatedRegionSet() {
    auto n = make_object<AnnotatedRegionSetNode>();
    data_ = std::move(n);
  }

  /*!
 * \brief Construct from an object pointer.
 *
 * \param n The object pointer.
 */
  explicit AnnotatedRegionSet(ObjectPtr<Object> n) : ObjectRef(n) {}

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
  AnnotatedRegionSetNode* operator->() const {
    auto* ptr = get_mutable();
    CHECK(ptr != nullptr);
    return static_cast<AnnotatedRegionSetNode*>(ptr);
  }

  /*! \return The region an expression belongs to. */
  AnnotatedRegion operator[](const Expr& expr) {
    const auto *n = operator->();
    CHECK(n);
    return n->GetRegion(expr);
  }

  /*! \brief Create a RegionSet from a relay expression.
   *
   * \param expr The relay expr from which to construct the set.
   * \param begin Region begin annotation operator.
   * \param end Region end annotation operator.
   *
   * \return The created RegionSet for the expression.
   */
  static AnnotatedRegionSet Create(const Expr& expr,
                                   const Op& begin,
                                   const Op& end);

 private:
  /*! \brief Helper class to construct a RegionSet from an expr.*/
  class Creator;
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_ANALYSIS_ANNOTATED_REGION_SET_H_
