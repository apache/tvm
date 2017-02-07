/*!
 *  Copyright (c) 2017 by Contributors
 *  Exposre of pass functions.
 * \file api_pass.cc
 */
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/api_registry.h>

namespace tvm {
namespace ir {

TVM_REGISTER_API(_pass_Simplify)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args[0].IsNodeType<Stmt>()) {
      *ret = Simplify(args[0].operator Stmt());
    } else {
      *ret = Simplify(args[0].operator Expr());
    }
  });

TVM_REGISTER_API(_pass_Equal)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args[0].IsNodeType<Stmt>()) {
      *ret = Equal(args[0].operator Stmt(), args[1].operator Stmt());
    } else {
      *ret = Equal(args[0].operator Expr(), args[1].operator Expr());
    }
  });

TVM_REGISTER_API(_pass_PostOrderVisit)
.set_body([](TVMArgs args, TVMRetValue *ret) {
    PackedFunc f = args[1];
    ir::PostOrderVisit(args[0], [f](const NodeRef& n) {
        f(n);
      });
  });

// make from two arguments
#define REGISTER_PASS1(PassName)                                  \
  TVM_REGISTER_API(_pass_## PassName)                             \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {                 \
      *ret = PassName(args[0]);                                   \
    })                                                            \

#define REGISTER_PASS2(PassName)                                  \
  TVM_REGISTER_API(_pass_## PassName)                             \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {                 \
      *ret = PassName(args[0], args[1]);                          \
    })                                                            \

#define REGISTER_PASS4(PassName)                                        \
  TVM_REGISTER_API(_pass_## PassName)                                   \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {                       \
      *ret = PassName(args[0], args[1], args[2], args[3]);              \
    })                                                                  \

REGISTER_PASS1(ConvertSSA);
REGISTER_PASS1(VerifySSA);
REGISTER_PASS1(CanonicalSimplify);
REGISTER_PASS4(Inline);
REGISTER_PASS2(StorageFlatten);
REGISTER_PASS1(VectorizeLoop);
REGISTER_PASS2(UnrollLoop);
REGISTER_PASS2(StorageSync);
REGISTER_PASS4(MakeAPI);
REGISTER_PASS1(SplitHostDevice);

}  // namespace ir
}  // namespace tvm
