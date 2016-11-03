/*!
 *  Copyright (c) 2016 by Contributors
 * \file split.h
 * \brief Define a split over Domain or RDomain
 */
#ifndef TVM_SPLIT_H_
#define TVM_SPLIT_H_

#include "./base.h"
#include "./domain.h"

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
  /*! \brief whether the split is over reduction domain*/
  bool split_over_rdom{false};
};

/*! \brief simple split node that splits over one dimension */
class DimSplitNode : public SplitNode {
 public:
  /*! \brief The dimension to split on */
  int dim_index;
  /*! \brief The factor of the split */
  Expr factor;
  /*! \brief constructor */
  DimSplitNode() {}
  const char* type_key() const final {
    return "DimSplit";
  }
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("split_over_rdom", &split_over_rdom);
    v->Visit("dim_index", &dim_index);
    v->Visit("factor", &factor);
  }
  static Split make(int dim_index,
                    Expr factor,
                    bool over_rdom);
};

// Implementations of inline functions
inline const SplitNode* Split::operator->() const {
  return static_cast<const SplitNode*>(node_.get());
}

}  // namespace tvm
#endif  // TVM_SPLIT_H_
