/*!
 *  Copyright (c) 2016 by Contributors
 *  Exposre of pass functions.
 * \file c_api_pass.cc
 */
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include "./c_api_registry.h"

namespace tvm {
namespace ir {
using ArgStack = const std::vector<APIVariantValue>;
using RetValue = APIVariantValue;

TVM_REGISTER_API(_pass_Simplify)
.set_body([](const ArgStack& args, RetValue *ret) {
    if (dynamic_cast<Stmt::ContainerType*>(args.at(0).sptr.get())) {
      *ret = Simplify(args.at(0).operator Stmt());
    } else {
      *ret = Simplify(args.at(0).operator Expr());
    }
  });

TVM_REGISTER_API(_pass_Equal)
.set_body([](const ArgStack& args, RetValue *ret) {
    if (dynamic_cast<Stmt::ContainerType*>(args.at(0).sptr.get())) {
      CHECK(args.at(1).type_id == kNodeHandle);
      *ret = Equal(args.at(0).operator Stmt(), args.at(1).operator Stmt());
    } else {
      Expr a = args.at(0).operator Expr();
      Expr b = args.at(1).operator Expr();
      *ret = Equal(a, b);
    }
  });

// make from two arguments
#define REGISTER_PASS1(PassName)                                  \
  TVM_REGISTER_API(_pass_## PassName)                             \
  .set_body([](const ArgStack& args,  RetValue *ret) {            \
      *ret = PassName(args.at(0));                                \
    })                                                            \

#define REGISTER_PASS2(PassName)                                  \
  TVM_REGISTER_API(_pass_## PassName)                             \
  .set_body([](const ArgStack& args,  RetValue *ret) {            \
      *ret = PassName(args.at(0), args.at(1));                    \
    })                                                            \

#define REGISTER_PASS4(PassName)                                        \
  TVM_REGISTER_API(_pass_## PassName)                                   \
  .set_body([](const ArgStack& args,  RetValue *ret) {                  \
      *ret = PassName(args.at(0), args.at(1), args.at(2), args.at(3));  \
    })                                                                  \

REGISTER_PASS1(ConvertSSA);
REGISTER_PASS1(VerifySSA);
REGISTER_PASS4(Inline);
REGISTER_PASS2(ScheduleOps);
REGISTER_PASS2(StorageFlatten);

}  // namespace ir
}  // namespace tvm
