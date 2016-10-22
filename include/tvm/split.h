/*!
 *  Copyright (c) 2016 by Contributors
 * \file split.h
 * \brief Define a split over Domain or RDomain
 */
#ifndef TVM_SPLIT_H_
#define TVM_SPLIT_H_

#include "./base.h"
#include "./array.h"
#include "./domain.h"

namespace tvm {

// internal node container for split.
class SplitNode;

/*! \brief Split over input domain */
class Split : public NodeRef {
 public:
  /*! \brief default constructor  */
  Split() {}
  /*! \return  Whether the split is over RDomain or not */
  inline bool is_over_rdom() const;
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
  int split_over_rdom{0};
  /*!
   * \brief given the output domain, infer input domain
   * \param split_index The index to be splitted on
   * \param out_domain The outer domain
   * \return The inferred inner domain.
   */
  virtual Domain InferInnerDomain(Expr split_index, Domain out_domain) const = 0;
};

/*! \brief simple split node that splits over one dimension */
class DimSplitNode : public SplitNode {
 public:
  /*! \brief The dimension to split on */
  int64_t dim_index;
  /*! \brief The factor of the split */
  Expr factor;
  /*! \brief constructor */
  DimSplitNode() {}
  const char* type_key() const override {
    return "DimSplitNode";
  }
  void VisitAttrs(AttrVisitor* visitor) override {
    visitor->Visit("split_over_rdom", &split_over_rdom);
  }
  void VisitNodeRefFields(FNodeRefVisit fvisit) override {
    fvisit("factor", &factor);
  }
  Domain InferInnerDomain(Expr split_index, Domain out_domain) const override {
    LOG(FATAL) << "not implemented";
    return Domain();
  }
};

// Implementations of inline functions
inline const SplitNode* Split::operator->() const {
  return static_cast<const SplitNode*>(node_.get());
}

inline bool Split::is_over_rdom() const {
  return (*this)->split_over_rdom != 0;
}

}  // namespace tvm
#endif  // TVM_SPLIT_H_
