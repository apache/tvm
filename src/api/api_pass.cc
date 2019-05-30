/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2017 by Contributors
 *  Exposre of pass functions.
 * \file api_pass.cc
 */
#include <tvm/expr.h>
#include <tvm/ir.h>
#include <tvm/attrs.h>
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

TVM_REGISTER_API("ir_pass.StorageFlatten")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args.size() <= 3) {
      *ret = StorageFlatten(args[0], args[1], args[2]);
    } else {
      *ret = StorageFlatten(args[0], args[1], args[2], args[3]);
    }
  });

TVM_REGISTER_API("ir_pass.AttrsEqual")
.set_body_typed<bool(const NodeRef&, const NodeRef&)>([](const NodeRef& lhs, const NodeRef& rhs) {
    return AttrsEqual()(lhs, rhs);
  });

TVM_REGISTER_API("ir_pass.AttrsHash")
.set_body_typed<int64_t(const NodeRef&)>([](const NodeRef &node) {
    return AttrsHash()(node);
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
#define REGISTER_PASS(PassName)                                   \
  TVM_REGISTER_API("ir_pass."#PassName)                           \
  .set_body_typed(PassName);                                     \


REGISTER_PASS(ConvertSSA);
REGISTER_PASS(VerifySSA);
REGISTER_PASS(RewriteUnsafeSelect);
REGISTER_PASS(Inline);
REGISTER_PASS(IRTransform);
REGISTER_PASS(VectorizeLoop);
REGISTER_PASS(SkipVectorize);
REGISTER_PASS(UnrollLoop);
REGISTER_PASS(InjectCopyIntrin);
REGISTER_PASS(ThreadSync);
REGISTER_PASS(MakeAPI);
REGISTER_PASS(BindDeviceType);
REGISTER_PASS(SplitHostDevice);
REGISTER_PASS(StorageRewrite);
REGISTER_PASS(CoProcSync);
REGISTER_PASS(LowerStorageAccessInfo);
REGISTER_PASS(InjectVirtualThread);
REGISTER_PASS(InjectPrefetch);
REGISTER_PASS(InjectDoubleBuffer);
REGISTER_PASS(LoopPartition);
REGISTER_PASS(RemoveNoOp);
REGISTER_PASS(SplitPipeline);
REGISTER_PASS(LiftAttrScope);
REGISTER_PASS(NarrowChannelAccess);
REGISTER_PASS(LowerThreadAllreduce);
REGISTER_PASS(LowerWarpMemory);
REGISTER_PASS(RemapThreadAxis);
REGISTER_PASS(LowerIntrin);
REGISTER_PASS(LowerCustomDatatypes);
REGISTER_PASS(LowerTVMBuiltin);
REGISTER_PASS(CombineContextCall);
REGISTER_PASS(VerifyMemory);
REGISTER_PASS(VerifyGPUCode);
REGISTER_PASS(DecorateDeviceScope);
REGISTER_PASS(InstrumentBoundCheckers);
}  // namespace ir
}  // namespace tvm
