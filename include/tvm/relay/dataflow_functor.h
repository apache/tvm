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
 * \file tvm/relay/dataflow_matcher.h
 * \brief A pattern matcher for matching dataflow properties.
 */
#ifndef TVM_RELAY_DATAFLOW_FUNCTOR_H_
#define TVM_RELAY_DATAFLOW_FUNCTOR_H_

#include <tvm/relay/dataflow_pattern.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {

/*!
 * \brief A dynamical functor that dispatches on in the first DFPattern argument.
 *
 * \tparam FType function signiture
 *  This type is only defined for FType with function signature R(const DFPattern&,
 * Args...)
 */
template <typename FType>
class DFPatternFunctor;

// functions to be overriden.
#define DFPATTERN_FUNCTOR_DEFAULT \
  { return VisitDFPatternDefault_(op, std::forward<Args>(args)...); }

#define RELAY_DFPATTERN_FUNCTOR_DISPATCH(OP)                                                    \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {          \
    return self->VisitDFPattern_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

template <typename R, typename... Args>
class DFPatternFunctor<R(const DFPattern& n, Args...)> {
 private:
  using TSelf = DFPatternFunctor<R(const DFPattern& n, Args...)>;
  using FType = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self, Args...)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~DFPatternFunctor() {}
  /*!
   * \brief Same as call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  R operator()(const DFPattern& n, Args... args) {
    return VisitDFPattern(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
  virtual R VisitDFPattern(const DFPattern& n, Args... args) {
    CHECK(n.defined());
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
  // Functions that can be overriden by subclass
  virtual R VisitDFPattern_(const AltPatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const AttrPatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const CallPatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const DominatorPatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const ExprPatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const TupleGetItemPatternNode* op,
                            Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const TuplePatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const TypePatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const VarPatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPattern_(const WildcardPatternNode* op, Args... args) DFPATTERN_FUNCTOR_DEFAULT;
  virtual R VisitDFPatternDefault_(const Object* op, Args...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    throw;
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;
    // Set dispatch
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(AltPatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(AttrPatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(CallPatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(DominatorPatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(ExprPatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(TupleGetItemPatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(TuplePatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(TypePatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(VarPatternNode);
    RELAY_DFPATTERN_FUNCTOR_DISPATCH(WildcardPatternNode);
    return vtable;
  }
};

/*!
 * \brief A simple visitor wrapper around DFPatternFunctor.
 *  Recursively visit the content.
 *
 *  DFPatternVisitor treats the Pattern as dataflow graph,and only visit each Expr node once.
 */
class DFPatternVisitor : public DFPatternFunctor<void(const DFPattern&)> {
 public:
  void VisitDFPattern(const DFPattern& pattern) override;
  void VisitDFPattern_(const AltPatternNode* op) override;
  void VisitDFPattern_(const AttrPatternNode* op) override;
  void VisitDFPattern_(const CallPatternNode* op) override;
  void VisitDFPattern_(const DominatorPatternNode* op) override;
  void VisitDFPattern_(const ExprPatternNode* op) override;
  void VisitDFPattern_(const TupleGetItemPatternNode* op) override;
  void VisitDFPattern_(const TuplePatternNode* op) override;
  void VisitDFPattern_(const TypePatternNode* op) override;
  void VisitDFPattern_(const VarPatternNode* op) override;
  void VisitDFPattern_(const WildcardPatternNode* op) override;

 protected:
  // set of already-visited nodes
  std::unordered_set<const Object*> visited_;
};

/*!
 * \brief A Wrapper around a templated graph type
 *  Holds a forward-backward indexed representation of the graph and a dominator tree representation
 * of the graph
 *
 *  Class is Templated and the implementaiton is in the header file so we can analyis both DFPattern
 * and Expr with the same infrastructure.
 *
 *  IndexedGraph should be instantiated thorught the CreateIndexedGraph utilities.
 */
template <typename T>
class IndexedGraph {
 public:
  /*! \brief A Node that wraps the input type and represents the indexed graph and dominator tree */
  struct Node {
    /*! \brief Node Constructor
     *  \param ref The input graph node
     *  \param index The index of the node in toplogoical order
     */
    Node(const T& ref, const size_t index) : ref_(ref), index_(index) {}

    /*! \brief The input node */
    const T ref_;
    /*! \brief The topological order index */
    const size_t index_;

    /*! \brief A boolean to determine if this node is external to the graph */
    bool is_external_ = false;
    /*! \brief The forward outputs/users of the node */
    std::vector<std::shared_ptr<Node>> outputs_;

    /*! \brief The depth of the node in the dominator tree */
    size_t depth_;
    /*! \brief The dominator parent/final user of the outputs of this node */
    std::shared_ptr<Node> dominator_parent_;
  };
  /*! \brief Construct the domination create of the index graph */
  void PostDom() {
    for (size_t i = topological_order_.size(); i != 0; --i) {
      size_t index = i - 1;
      auto current = topological_order_[index];
      if (current->is_external_) {
        current->depth_ = 1;
        current->dominator_parent_ = nullptr;
      } else {
        auto parent = LeastCommonAncestor(current->outputs_);
        current->depth_ = parent ? parent->depth_ + 1 : 1;
        current->dominator_parent_ = parent;
      }
    }
  }
  /*! \brief Map of input nodes to IndexedGraph Nodes */
  std::unordered_map<T, std::shared_ptr<Node>, ObjectHash, ObjectEqual> node_map_;
  /*! \brief Topological IndexedGraph Nodes */
  std::vector<std::shared_ptr<Node>> topological_order_;

 protected:
  /*! \brief Find the least common ancestor of all outputs of a node */
  std::shared_ptr<Node> LeastCommonAncestor(const std::vector<std::shared_ptr<Node>>& outputs) {
    if (outputs.size() == 0) {
      return nullptr;
    }
    auto parent = outputs.at(0);
    for (size_t i = 1; i < outputs.size(); ++i) {
      parent = LeastCommonAncestor(parent, outputs.at(i));
    }
    return parent;
  }

  /*! \brief Find the least common ancestor of two nodes */
  std::shared_ptr<Node> LeastCommonAncestor(std::shared_ptr<Node> lhs, std::shared_ptr<Node> rhs) {
    if (lhs == nullptr || rhs == nullptr) {
      return nullptr;
    }
    while (lhs != rhs) {
      if (lhs->depth_ < rhs->depth_) {
        rhs = rhs->dominator_parent_;
      } else if (lhs->depth_ > rhs->depth_) {
        lhs = lhs->dominator_parent_;
      } else {
        rhs = rhs->dominator_parent_;
        lhs = lhs->dominator_parent_;
      }
    }
    return lhs;
  }
};

/*! \brief Create an Indexed Graph based on an Expr */
IndexedGraph<Expr> CreateIndexedGraph(const Expr& expr);
/*! \brief Create an Indexed Graph based on an DFPattern */
IndexedGraph<DFPattern> CreateIndexedGraph(const DFPattern& pattern);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_DATAFLOW_FUNCTOR_H_
