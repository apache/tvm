/*!
 *  Copyright (c) 2017 by Contributors
 *  Exposre of pass functions.
 * \file api_pass.cc
 */
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/api_registry.h>

namespace tvm {
namespace ir {

TVM_REGISTER_API("ir_pass.Simplify")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args[0].IsNodeType<Stmt>()) {
      if (args.size() > 1) {
        *ret = Simplify(args[0].operator Stmt(), args[1]);
      } else {
        *ret = Simplify(args[0].operator Stmt());
      }
    } else {
      if (args.size() > 1) {
        *ret = Simplify(args[0].operator Expr(), args[1]);
      } else {
        *ret = Simplify(args[0].operator Expr());
      }
    }
  });

TVM_REGISTER_API("ir_pass.CanonicalSimplify")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args[0].IsNodeType<Stmt>()) {
      if (args.size() > 1) {
        *ret = CanonicalSimplify(args[0].operator Stmt(), args[1]);
      } else {
        *ret = CanonicalSimplify(args[0].operator Stmt());
      }
    } else {
      if (args.size() > 1) {
        *ret = CanonicalSimplify(args[0].operator Expr(), args[1]);
      } else {
        *ret = CanonicalSimplify(args[0].operator Expr());
      }
    }
  });

TVM_REGISTER_API("ir_pass.Substitute")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args[0].IsNodeType<Stmt>()) {
      *ret = Substitute(args[0].operator Stmt(), args[1].operator Map<Var, Expr>());
    } else {
      *ret = Substitute(args[0].operator Expr(), args[1].operator Map<Var, Expr>());
    }
  });

TVM_REGISTER_API("ir_pass.Equal")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args[0].IsNodeType<Stmt>()) {
      *ret = Equal(args[0].operator Stmt(), args[1].operator Stmt());
    } else {
      *ret = Equal(args[0].operator Expr(), args[1].operator Expr());
    }
  });

TVM_REGISTER_API("ir_pass.ExprUseVar")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = ExprUseVar(args[0].operator Expr(), args[1].operator Var());
  });

TVM_REGISTER_API("ir_pass.PostOrderVisit")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    PackedFunc f = args[1];
    ir::PostOrderVisit(args[0], [f](const NodeRef& n) {
        f(n);
      });
  });

// make from two arguments
#define REGISTER_PASS1(PassName)                                  \
  TVM_REGISTER_API("ir_pass."#PassName)                           \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {                 \
      *ret = PassName(args[0]);                                   \
    })                                                            \

#define REGISTER_PASS2(PassName)                                  \
  TVM_REGISTER_API("ir_pass."#PassName)                           \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {                 \
      *ret = PassName(args[0], args[1]);                          \
    })                                                            \

#define REGISTER_PASS3(PassName)                                        \
  TVM_REGISTER_API("ir_pass."#PassName)                                 \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {                       \
      *ret = PassName(args[0], args[1], args[2]);                       \
    })                                                                  \

#define REGISTER_PASS4(PassName)                                        \
  TVM_REGISTER_API("ir_pass."#PassName)                                 \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {                       \
      *ret = PassName(args[0], args[1], args[2], args[3]);              \
    })                                                                  \

#define REGISTER_PASS5(PassName)                                        \
  TVM_REGISTER_API("ir_pass."#PassName)                                 \
  .set_body([](TVMArgs args,  TVMRetValue *ret) {                       \
      *ret = PassName(args[0], args[1], args[2], args[3], args[4]);     \
    })                                                                  \

REGISTER_PASS1(ConvertSSA);
REGISTER_PASS1(VerifySSA);
REGISTER_PASS1(RewriteUnsafeSelect);
REGISTER_PASS4(Inline);
REGISTER_PASS3(StorageFlatten);
REGISTER_PASS4(IRTransform);
REGISTER_PASS1(VectorizeLoop);
REGISTER_PASS5(UnrollLoop);
REGISTER_PASS3(InjectCopyIntrin);
REGISTER_PASS2(ThreadSync);
REGISTER_PASS5(MakeAPI);
REGISTER_PASS2(BindDeviceType);
REGISTER_PASS1(SplitHostDevice);
REGISTER_PASS1(StorageRewrite);
REGISTER_PASS1(CoProcSync);
REGISTER_PASS1(LowerStorageAccessInfo);
REGISTER_PASS1(InjectVirtualThread);
REGISTER_PASS1(InjectPrefetch);
REGISTER_PASS2(InjectDoubleBuffer);
REGISTER_PASS2(LoopPartition);
REGISTER_PASS1(RemoveNoOp);
REGISTER_PASS2(SplitPipeline);
REGISTER_PASS2(LiftAttrScope);
REGISTER_PASS1(NarrowChannelAccess);
REGISTER_PASS2(LowerThreadAllreduce);
REGISTER_PASS2(LowerWarpMemory);
REGISTER_PASS2(RemapThreadAxis);
REGISTER_PASS2(LowerIntrin);
REGISTER_PASS1(LowerTVMBuiltin);
REGISTER_PASS1(CombineContextCall);
REGISTER_PASS2(VerifyMemory);
REGISTER_PASS2(VerifyGPUCode);
}  // namespace ir
}  // namespace tvm
