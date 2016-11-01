/*!
 *  Copyright (c) 2016 by Contributors
 * \file domain.h
 * \brief Defines the domain in AST
 */
#ifndef TVM_DOMAIN_H_
#define TVM_DOMAIN_H_

#include <ir/Range.h>
#include <memory>
#include "./base.h"
#include "./expr.h"

namespace tvm {

/*! \brief container class of reduction domain */
class RDomainNode;

/*!
 * \brief same as Halide::IR::Range
 *  except it provide an constructor with (begin, end)
 *
 *  \note Traditional Halide's Range have a constructor with
 *   (begin, extent), which does not match the convention in e.g. python.
 *   We decided to correct it by removing the constructor in HalideIR,
 *   and add it back in TVM's range.
 */
class Range : public Halide::IR::Range {
 public:
  /*! \brief constructor */
  Range() {}
  explicit Range(std::shared_ptr<Node> n) : Halide::IR::Range(n) {}
  /*!
   * \brief constructor by begin and end
   * \param begin The begin of the range.
   * \param end The end of the range.
   */
  Range(Expr begin, Expr end);
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
};

/*! \brief use RDom as alias of RDomain */
using RDom = RDomain;

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
    return "RDomain";
  }
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("index", &index);
    v->Visit("domain", &domain);
  }
};

inline const RDomainNode* RDomain::operator->() const {
  return static_cast<const RDomainNode*>(node_.get());
}

inline size_t RDomain::ndim() const {
  return (*this)->index.size();
}

inline Var RDomain::index(size_t i) const {
  return (*this)->index[i];
}

// overload print function
inline std::ostream& operator<<(std::ostream &os, const RDomain& r){  // NOLINT(*)
  os << "rdomain(" << r->domain << ")";
  return os;
}

}  // namespace tvm

#endif  // TVM_DOMAIN_H_
