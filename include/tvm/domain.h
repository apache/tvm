/*!
 *  Copyright (c) 2016 by Contributors
 * \file domain.h
 * \brief Defines the AST
 */
#ifndef TVM_DOMAIN_H_
#define TVM_DOMAIN_H_

#include <memory>
#include "./base.h"
#include "./array.h"
#include "./expr.h"

namespace tvm {

// Internal node container of Range
class RangeNode;
// Internal node container of RDomain
class RDomainNode;

/*! \brief Node range */
class Range : public NodeRef {
 public:
  /*! \brief constructor */
  Range() {}
  /*!
   * \brief constructor
   * \param begin start of the range.
   * \param end end of the range.
   */
  Range(Expr begin, Expr end);
  /*! \return The extent of the range */
  Expr extent() const;
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const RangeNode* operator->() const;
  /*! \return the begining of the range */
  inline const Expr& begin() const;
  /*! \return the end  of the range */
  inline const Expr& end() const;
  // overload print function
  friend std::ostream& operator<<(std::ostream &os, const Range& r) {  // NOLINT(*)
    os << '[' << r.begin() << ", " << r.end() <<')';
    return os;
  }
};

/*! \brief Domain is a multi-dimensional range */
using Domain = Array<Range>;

/*! \brief reduction domain */
class RDomain : public NodeRef {
 public:
  /*! \brief constructor*/
  RDomain() {}
  /*!
   * constructor by domain
   * \param domain The domain of reduction.
   */
  explicit RDomain(Domain domain);
  /*!
   * \brief constructor by list of ranges
   * \param domain The reduction domain
   */
  explicit RDomain(std::initializer_list<Range> domain)
      : RDomain(Domain(domain)) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const RDomainNode* operator->() const;
  /*! \return The dimension of the RDomain */
  inline size_t ndim() const;
  /*!
   * \param i the index.
   * \return i-th index variable in the RDomain
   */
  inline Var index(size_t i) const;
  /*! \return the 0-th index of the domain */
  inline Var i0() const {
    return index(0);
  }
  /*!
   * \return The domain of the reduction.
   */
  inline const Domain& domain() const;
  // overload print function
  friend std::ostream& operator<<(std::ostream &os, const RDomain& r){  // NOLINT(*)
    os << "rdomain(" << r.domain() << ")";
    return os;
  }
};

/*! \brief use RDom as alias of RDomain */
using RDom = RDomain;

/*! \brief range over one dimension */
class RangeNode : public Node {
 public:
  /*! \brief beginning of the node */
  Expr begin;
  /*! \brief end of the node */
  Expr end;
  /*! \brief constructor */
  RangeNode() {}
  RangeNode(Expr && begin, Expr && end)
      : begin(std::move(begin)), end(std::move(end)) {
  }
  const char* type_key() const override {
    return "RangeNode";
  }
  void VisitNodeRefFields(FNodeRefVisit fvisit) override {
    fvisit("begin", &begin);
    fvisit("end", &end);
  }
  void VisitAttrs(AttrVisitor* visitor) override {}
};

/*! \brief reduction domain node */
class RDomainNode : public Node {
 public:
  /*! \brief internal index */
  Array<Var> index;
  /*! \brief The inernal domain */
  Domain domain;
  /*! \brief constructor */
  RDomainNode() {}
  RDomainNode(Array<Var> && index, Domain && domain)
      : index(std::move(index)), domain(std::move(domain)) {
  }
  const char* type_key() const override {
    return "RDomainNode";
  }
  void VisitNodeRefFields(FNodeRefVisit fvisit) override {
    fvisit("index", &index);
    fvisit("domain", &domain);
  }
  void VisitAttrs(AttrVisitor* visitor) override {}
};

// implements of inline functions
inline const RangeNode* Range::operator->() const {
  return static_cast<const RangeNode*>(node_.get());
}

inline const Expr& Range::begin() const {
  return (*this)->begin;
}

inline const Expr& Range::end() const {
  return (*this)->end;
}

inline const RDomainNode* RDomain::operator->() const {
  return static_cast<const RDomainNode*>(node_.get());
}

inline size_t RDomain::ndim() const {
  return (*this)->index.size();
}

inline Var RDomain::index(size_t i) const {
  return (*this)->index[i];
}

inline const Domain& RDomain::domain() const {
  return (*this)->domain;
}

}  // namespace tvm

#endif  // TVM_DOMAIN_H_
