/*!
 *  Copyright (c) 2016 by Contributors
 * \file domain.cc
 */
#include <tvm/base.h>
#include <tvm/domain.h>

namespace tvm {

Range::Range(Expr begin, Expr end)
    : Range(std::make_shared<Halide::IR::RangeNode>(begin, end - begin)) {
  // TODO(tqchen) add simplify to end - begin
}

Range Range::make_with_min_extent(Expr min, Expr extent) {
  return Range(std::make_shared<Halide::IR::RangeNode>(min, extent));
}

RDomain::RDomain(Domain domain) {
  std::vector<Var> index;
  for (size_t i = 0; i < domain.size(); ++i) {
    std::ostringstream os;
    os << "reduction_index" << i;
    index.push_back(Var(os.str()));
  }
  Array<Var> idx(index);
  node_ = std::make_shared<RDomainNode>(
      std::move(idx), std::move(domain));
}

RDomain RDomain::make(Array<Var> index, Domain domain) {
  return RDomain(std::make_shared<RDomainNode>(index, domain));
}

TVM_REGISTER_NODE_TYPE(RDomainNode);

}  // namespace tvm
