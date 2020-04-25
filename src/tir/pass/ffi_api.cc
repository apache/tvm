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
 *  Exposure of pass functions.
 * \file ffi_api.cc
 */
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/ir/attrs.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace tir {

TVM_REGISTER_GLOBAL("ir_pass.Simplify")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args[0].IsObjectRef<Stmt>()) {
      if (args.size() > 1) {
        *ret = Simplify(args[0].operator Stmt(), args[1]);
      } else {
        *ret = Simplify(args[0].operator Stmt());
      }
    } else {
      if (args.size() > 1) {
        *ret = Simplify(args[0].operator PrimExpr(), args[1]);
      } else {
        *ret = Simplify(args[0].operator PrimExpr());
      }
    }
  });

TVM_REGISTER_GLOBAL("ir_pass.CanonicalSimplify")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args[0].IsObjectRef<Stmt>()) {
      if (args.size() > 1) {
        *ret = CanonicalSimplify(args[0].operator Stmt(), args[1]);
      } else {
        *ret = CanonicalSimplify(args[0].operator Stmt());
      }
    } else {
      if (args.size() > 1) {
        *ret = CanonicalSimplify(args[0].operator PrimExpr(), args[1]);
      } else {
        *ret = CanonicalSimplify(args[0].operator PrimExpr());
      }
    }
  });

TVM_REGISTER_GLOBAL("ir_pass.Substitute")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args[0].IsObjectRef<Stmt>()) {
      *ret = Substitute(args[0].operator Stmt(), args[1].operator Map<Var, PrimExpr>());
    } else {
      *ret = Substitute(args[0].operator PrimExpr(), args[1].operator Map<Var, PrimExpr>());
    }
  });

TVM_REGISTER_GLOBAL("ir_pass.StorageFlatten")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args.size() <= 3) {
      *ret = StorageFlatten(args[0], args[1], args[2]);
    } else {
      *ret = StorageFlatten(args[0], args[1], args[2], args[3]);
    }
  });

TVM_REGISTER_GLOBAL("ir_pass.RewriteForTensorCore")
.set_body_typed
  ([](const Stmt& stmt,
      const te::Schedule& schedule,
      const Map<te::Tensor, Buffer>& extern_buffer) {
      return RewriteForTensorCore(stmt, schedule, extern_buffer);
  });

TVM_REGISTER_GLOBAL("ir_pass.ExprUseVar")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = ExprUseVar(args[0].operator PrimExpr(), args[1].operator Var());
  });

TVM_REGISTER_GLOBAL("ir_pass.PostOrderVisit")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    PackedFunc f = args[1];
    tir::PostOrderVisit(args[0], [f](const ObjectRef& n) {
        f(n);
      });
  });


// make from two arguments
#define REGISTER_PASS(PassName)                                   \
  TVM_REGISTER_GLOBAL("ir_pass."#PassName)                        \
  .set_body_typed(PassName);                                      \


REGISTER_PASS(ConvertSSA);
REGISTER_PASS(VerifySSA);
REGISTER_PASS(RewriteUnsafeSelect);
REGISTER_PASS(Inline);
REGISTER_PASS(IRTransform);
REGISTER_PASS(VectorizeLoop);
REGISTER_PASS(SkipVectorize);
REGISTER_PASS(UnrollLoop);
REGISTER_PASS(InjectCopyIntrin);
REGISTER_PASS(StorageRewrite);
REGISTER_PASS(CoProcSync);
REGISTER_PASS(LowerStorageAccessInfo);
REGISTER_PASS(InjectVirtualThread);
REGISTER_PASS(InjectPrefetch);
REGISTER_PASS(InjectDoubleBuffer);
REGISTER_PASS(LoopPartition);
REGISTER_PASS(RemoveNoOp);
REGISTER_PASS(LiftAttrScope);
REGISTER_PASS(VerifyGPUCode);
REGISTER_PASS(DecorateDeviceScope);
REGISTER_PASS(InstrumentBoundCheckers);
REGISTER_PASS(VerifyCompactBuffer);
REGISTER_PASS(HoistIfThenElse);
REGISTER_PASS(NarrowDataType);
}  // namespace tir
}  // namespace tvm
