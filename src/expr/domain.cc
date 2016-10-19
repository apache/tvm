/*!
 *  Copyright (c) 2016 by Contributors
 * \file domain.cc
 */
#include <tvm/domain.h>
#include <tvm/op.h>
#include <tvm/expr_node.h>
#include <tvm/expr_util.h>

namespace tvm {

Range::Range(Expr begin, Expr end) {
  node_ = std::make_shared<RangeNode>(
      std::move(begin), std::move(end));
}

Expr Range::extent() const {
  return Simplify(end() - begin());
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

TVM_REGISTER_NODE_TYPE(RangeNode);
TVM_REGISTER_NODE_TYPE(ArrayNode);
TVM_REGISTER_NODE_TYPE(RDomainNode);

}  // namespace tvm
