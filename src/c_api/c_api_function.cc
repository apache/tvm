/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions
 * \file c_api_impl.cc
 */
#include <tvm/expr.h>
#include <ir/IROperator.h>
#include "./c_api_registry.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::tvm::APIFunctionReg);
}  // namespace dmlc

namespace tvm {

using namespace Halide::Internal;

using ArgStack = const std::vector<APIVariantValue>;
using RetValue = APIVariantValue;

TVM_REGISTER_API(_const)
.set_body([](const ArgStack& args,  RetValue *ret) {
    if (args.at(0).type_id == kLong) {
      *ret = make_const(args.at(1), args.at(0).operator int64_t());
    } else if (args.at(0).type_id == kDouble) {
      *ret = make_const(args.at(1), args.at(0).operator double());
    } else {
      LOG(FATAL) << "only accept int or float";
    }
  })
.add_argument("src", "Number", "source number")
.add_argument("dtype", "str", "data type");

TVM_REGISTER_API(format_str)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_id == kNodeHandle);
    std::ostringstream os;
    auto& sptr = args.at(0).sptr;
    if (dynamic_cast<const BaseExprNode*>(sptr.get())) {
      os << args.at(0).operator Expr();
    } else if (dynamic_cast<const BaseStmtNode*>(sptr.get())) {
      os << args.at(0).operator Stmt();
    }
    *ret = os.str();
  })
.add_argument("expr", "Node", "expression to be printed");

}  // namespace tvm
