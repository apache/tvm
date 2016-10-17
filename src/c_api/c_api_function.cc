/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions
 * \file c_api_impl.cc
 */
#include <tvm/expr.h>
#include <tvm/op.h>
#include "./c_api_registry.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::tvm::APIFunctionReg);
}  // namespace dmlc

namespace tvm {

using ArgStack = const std::vector<APIVariantValue>;
using RetValue = APIVariantValue;

TVM_REGISTER_API(Var)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = Var(args.at(0), static_cast<DataType>(static_cast<int>(args.at(1))));
  })
.add_argument("name", "str", "name of the var")
.add_argument("dtype", "int", "data type of var");


TVM_REGISTER_API(max)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = max(args.at(0), args.at(1));
  })
.add_argument("lhs", "Expr", "left operand")
.add_argument("rhs", "Expr", "right operand");

TVM_REGISTER_API(min)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = min(args.at(0), args.at(1));
  })
.add_argument("lhs", "Expr", "left operand")
.add_argument("rhs", "Expr", "right operand");

TVM_REGISTER_API(format_str)
.set_body([](const ArgStack& args,  RetValue *ret) {
    std::ostringstream os;
    os << Expr(args.at(0));
    *ret = os.str();
  })
.add_argument("expr", "Expr", "expression to be printed");

}  // namespace tvm
