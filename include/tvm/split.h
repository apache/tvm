/*!
 *  Copyright (c) 2016 by Contributors
 * \file split.h
 * \brief Define a split over Domain or RDomain
 */
#ifndef TVM_SPLIT_H_
#define TVM_SPLIT_H_

#include "./base.h"
#include "./expr.h"

namespace tvm {

// internal node container for split.
class SplitNode;

/*! \brief Split over input domain */
class Split : public NodeRef {
 public:
  /*! \brief default constructor  */
  Split() {}
  explicit Split(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const SplitNode* operator->() const;
};

/*!
 * \brief base class of split node,
 *  specifies a split over domain
 *  split also defines how to generate
 */
class SplitNode : public Node {
 public:
  /*! \brief the variable to be splitted on */
  Var var;
};

/*! \brief simple split node that splits over one dimension */
class DimSplitNode : public SplitNode {
 public:
  /*! \brief The factor of the split */
  Expr factor;
  /*! \brief constructor */
  DimSplitNode() {}

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("var", &var);
    v->Visit("factor", &factor);
  }
  static Split make(Var var, Expr factor);

  static constexpr const char* _type_key = "DimSplit";
  TVM_DECLARE_NODE_TYPE_INFO(DimSplitNode);
};

// Implementations of inline functions
inline const SplitNode* Split::operator->() const {
  return static_cast<const SplitNode*>(node_.get());
}

}  // namespace tvm
#endif  // TVM_SPLIT_H_
