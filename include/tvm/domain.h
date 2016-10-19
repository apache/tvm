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
  /*! \return the begining of the range */
  inline const Expr& begin() const {
    return static_cast<const RangeNode*>(node_.get())->begin;
  }
  /*! \return the end  of the range */
  inline const Expr& end() const {
    return static_cast<const RangeNode*>(node_.get())->end;
  }
  friend std::ostream& operator<<(std::ostream &os, const Range& r) {  // NOLINT(*)
    os << '[' << r.begin() << ", " << r.end() <<')';
    return os;
  }
};

/*! \brief Domain is a multi-dimensional range */
using Domain = Array<Range>;

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
   * \brief constructor from node pointer
   * \param nptr Another node shared pointer
   */
  explicit RDomain(std::shared_ptr<Node>&& nptr) : NodeRef(std::move(nptr)) {
    CHECK(node_.get() != nullptr);
    CHECK(node_->is_type<RDomainNode>());
  }
  /*! \return The dimension of the RDomain */
  inline size_t ndim() const {
    return static_cast<const RDomainNode*>(node_.get())->index.size();
  }
  /*! \return the 0-th index of the domain */
  inline Var i0() const {
    return index(0);
  }
  /*!
   * \param i the index.
   * \return i-th index variable in the RDomain
   */
  inline Var index(size_t i) const {
    return static_cast<const RDomainNode*>(node_.get())->index[i];
  }
  /*!
   * \return The domain of the reduction.
   */
  inline const Domain& domain() const {
    return static_cast<const RDomainNode*>(node_.get())->domain;
  }
  friend std::ostream& operator<<(std::ostream &os, const RDomain& r) {  // NOLINT(*)
    os << "rdomain(" << r.domain() << ")";
    return os;
  }
};

/*! \brief use RDom as alias of RDomain */
using RDom = RDomain;

}  // namespace tvm

#endif  // TVM_DOMAIN_H_
