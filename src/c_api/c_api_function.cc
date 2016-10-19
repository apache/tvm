/*!
 *  Copyright (c) 2016 by Contributors
 *  Implementation of API functions
 * \file c_api_impl.cc
 */
#include <tvm/expr.h>
#include <tvm/op.h>
#include <tvm/tensor.h>
#include "./c_api_registry.h"

namespace dmlc {
DMLC_REGISTRY_ENABLE(::tvm::APIFunctionReg);
}  // namespace dmlc

namespace tvm {

using ArgStack = const std::vector<APIVariantValue>;
using RetValue = APIVariantValue;

// expression logic x
TVM_REGISTER_API(_Var)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = Var(args.at(0),
               static_cast<DataType>(static_cast<int>(args.at(1))));
  })
.add_argument("name", "str", "name of the var")
.add_argument("dtype", "int", "data type of var");

TVM_REGISTER_API(constant)
.set_body([](const ArgStack& args,  RetValue *ret) {
    if (args.at(0).type_id == kLong) {
      *ret = IntConstant(args.at(0));
    } else if (args.at(0).type_id == kDouble) {
      *ret = FloatConstant(args.at(0));
    } else {
      LOG(FATAL) << "only accept int or float";
    }
  })
.add_argument("src", "Number", "source number");

TVM_REGISTER_API(binary_op)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_id == kStr);
    *ret = (*BinaryOp::Get(args.at(0).str.c_str()))(args.at(1), args.at(2));
  })
.add_argument("op", "str", "operator")
.add_argument("lhs", "Expr", "left operand")
.add_argument("rhs", "Expr", "right operand");

TVM_REGISTER_API(_raw_ptr)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_id == kNodeHandle);
    *ret = reinterpret_cast<int64_t>(args.at(0).sptr.get());
  })
.add_argument("src", "NodeBase", "the node base");

TVM_REGISTER_API(Range)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = Range(args.at(0), args.at(1));
  })
.add_argument("begin", "Expr", "beginning of the range.")
.add_argument("end", "Expr", "end of the range");

TVM_REGISTER_API(_TensorInput)
.set_body([](const ArgStack& args,  RetValue *ret) {
    *ret = Tensor(
        static_cast<Array<Expr> >(args.at(0)),
        static_cast<std::string>(args.at(1)),
        static_cast<DataType>(static_cast<int>(args.at(1))));
  });

// transformations
TVM_REGISTER_API(format_str)
.set_body([](const ArgStack& args,  RetValue *ret) {
    CHECK(args.at(0).type_id == kNodeHandle);
    std::ostringstream os;
    auto& sptr = args.at(0).sptr;
    if (sptr->is_type<TensorNode>()) {
      os << args.at(0).operator Tensor();
    } else if (sptr->is_type<RDomainNode>()) {
      os << args.at(0).operator RDomain();
    } else if (sptr->is_type<RangeNode>()) {
      os << args.at(0).operator Range();
    } else {
      os << args.at(0).operator Expr();
    }
    *ret = os.str();
  })
.add_argument("expr", "Expr", "expression to be printed");

}  // namespace tvm
