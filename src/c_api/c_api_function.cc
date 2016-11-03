/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions
 * \file c_api_impl.cc
 */
#include <tvm/expr.h>
#include <tvm/domain.h>
#include <tvm/tensor.h>
#include <ir/IROperator.h>
#include "./c_api_registry.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::tvm::APIFunctionReg);
}  // namespace dmlc

namespace tvm {

using ArgStack = const std::vector<APIVariantValue>;
using RetValue = APIVariantValue;

TVM_REGISTER_API(_format_str)
.set_body([](const ArgStack& args,  RetValue *ret) {
    using Halide::Internal::BaseExprNode;
    using Halide::Internal::BaseStmtNode;

    CHECK(args.at(0).type_id == kNodeHandle);
    std::ostringstream os;
    auto& sptr = args.at(0).sptr;
    if (sptr->is_type<TensorNode>()) {
      os << args.at(0).operator Tensor();
    } else if (sptr->is_type<RDomainNode>()) {
      os << args.at(0).operator RDomain();
    } else if (dynamic_cast<const BaseExprNode*>(sptr.get())) {
      os << args.at(0).operator Expr();
    } else if (dynamic_cast<const BaseStmtNode*>(sptr.get())) {
      os << args.at(0).operator Stmt();
    } else {
      LOG(FATAL) << "don't know how to print input NodeBaseType";
    }
    *ret = os.str();
  })
.add_argument("expr", "Node", "expression to be printed");

TVM_REGISTER_API(_raw_ptr)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_id == kNodeHandle);
    *ret = reinterpret_cast<int64_t>(args.at(0).sptr.get());
  })
.add_argument("src", "NodeBase", "the node base");

}  // namespace tvm
