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

#include <gtest/gtest.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/rvalue_ref.h>
#include <tvm/ir/module.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/target/target.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/transform.h>

#include <utility>

TEST(LowerWarpMemory, RetainsBufferVariablesDuringRewrite) {
  using namespace tvm;
  using namespace tvm::tirx;

  auto make_function = []() {
    auto i32 = PrimType::Int(32);
    BufferVar buffer("warp_buffer", BufferType("warp", i32, {64}, {}, 0, 64, 1));
    PrimVar thread("tx", i32);
    IterVar axis(Range(), thread, IterVarType::kThreadIndex, "threadIdx.x");
    PrimVar value("value", i32);
    PrimExpr index = thread * 2;
    Stmt access =
        SeqStmt({BufferStore(buffer, value, {index}), Evaluate(BufferLoad(buffer, {index}))});
    Stmt body = SeqStmt({AllocBuffer(buffer), AttrStmt(axis, "thread_extent", 32, access)});
    Target target(ffi::Map<ffi::String, ffi::Any>{{"kind", "cuda"}, {"arch", "sm_80"}});
    DictAttrs attrs(ffi::Map<ffi::String, ffi::Any>{{"target", target}});
    return PrimFunc({value}, body, VoidType(), attrs);
  };

  for (bool shared : {false, true}) {
    SCOPED_TRACE(shared ? "shared module" : "unique module");
    // No separate function/body/buffer handle may keep the input alive in the unique case.
    IRModule module = IRModule::FromExpr(make_function());
    IRModule retained = shared ? module : IRModule(nullptr);
    IRModule lowered =
        ffi::Function::GetGlobalRequired("transform.RunPass")(
            tirx::transform::LowerWarpMemory(), ffi::RValueRef<IRModule>(std::move(module)))
            .cast<IRModule>();
    auto function = lowered->Lookup("main").as_or_throw<PrimFunc>();
    auto body = function->body.as_or_throw<SeqStmt>();
    auto alloc = body->seq[0].as_or_throw<AllocBuffer>();
    EXPECT_EQ(alloc->buffer.scope(), "local");
    ASSERT_EQ(alloc->buffer->shape.size(), 1);
    EXPECT_EQ(alloc->buffer->shape[0].as_or_throw<IntImm>()->value, 2);

    auto scope = body->seq[1].as_or_throw<AttrStmt>();
    auto access = scope->body.as_or_throw<SeqStmt>();
    auto store = access->seq[0].as_or_throw<BufferStore>();
    auto load = access->seq[1].as_or_throw<Evaluate>()->value.as_or_throw<TensorLoad>();
    EXPECT_TRUE(store->buffer.same_as(alloc->buffer));
    EXPECT_TRUE(load->source.same_as(alloc->buffer));
    EXPECT_EQ(store->indices[0].as_or_throw<IntImm>()->value, 0);
    EXPECT_EQ(load->indices[0].as_or_throw<IntImm>()->value, 0);

    if (shared) {
      auto original = retained->Lookup("main").as_or_throw<PrimFunc>();
      auto original_alloc =
          original->body.as_or_throw<SeqStmt>()->seq[0].as_or_throw<AllocBuffer>();
      EXPECT_EQ(original_alloc->buffer.scope(), "warp");
      EXPECT_EQ(original_alloc->buffer->shape[0].as_or_throw<IntImm>()->value, 64);
    }
  }
}
