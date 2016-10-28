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
    } else {
      LOG(FATAL) << "don't know how to print input NodeBaseType";
    }
    *ret = os.str();
  })
.add_argument("expr", "Node", "expression to be printed");

TVM_REGISTER_API(_Array)
.set_body([](const ArgStack& args,  RetValue *ret) {
    std::vector<std::shared_ptr<Node> > data;
    for (size_t i = 0; i < args.size(); ++i) {
      CHECK(args.at(i).type_id == kNodeHandle);
      data.push_back(args.at(i).sptr);
    }
    auto node = std::make_shared<ArrayNode>();
    node->data = std::move(data);
    ret->type_id = kNodeHandle;
    ret->sptr = node;
  });

TVM_REGISTER_API(_ArrayGetItem)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_id == kNodeHandle);
    int64_t i = args.at(1);
    auto& sptr = args.at(0).sptr;
    CHECK(sptr->is_type<ArrayNode>());
    auto* n = static_cast<const ArrayNode*>(sptr.get());
    CHECK_LT(static_cast<size_t>(i), n->data.size())
        << "out of bound of array";
    ret->sptr = n->data[i];
    ret->type_id = kNodeHandle;
  });

TVM_REGISTER_API(_ArraySize)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_id == kNodeHandle);
    auto& sptr = args.at(0).sptr;
    CHECK(sptr->is_type<ArrayNode>());
    *ret = static_cast<int64_t>(
        static_cast<const ArrayNode*>(sptr.get())->data.size());
  });

TVM_REGISTER_API(_Range)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = Range(args.at(0), args.at(1));
  })
.add_argument("min", "Expr", "beginning of the range.")
.add_argument("extent", "Expr", "extent of the range");

}  // namespace tvm
