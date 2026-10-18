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
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/reflection/creator.h>
#include <tvm/ir/op.h>
#include <tvm/target/target_kind.h>
#include <tvm/tirx/tile/dispatch.h>
#include <tvm/tirx/transform.h>

#include <memory>

namespace tvm::tirx {
namespace {
namespace refl = ffi::reflection;

void RegisterTestScopes() {
  static const bool registered = [] {
    // Register the inner scope first; enum indices must not define ordering.
    refl::EnumDef<ThreadScopeObj>("thread_scope_test.inner")
        .set_attr(thread_scope_attr::kBackend, ffi::String("thread_scope_test"))
        .set_attr(thread_scope_attr::kRank, 10);
    refl::EnumDef<ThreadScopeObj>("thread_scope_test.outer")
        .set_attr(thread_scope_attr::kBackend, ffi::String("thread_scope_test"))
        .set_attr(thread_scope_attr::kRank, -10);
    return true;
  }();
  (void)registered;
}

TEST(ThreadScope, NativeRegistrationIdentityAndRank) {
  RegisterTestScopes();
  ThreadScope cuda("cuda.thread");
  ThreadScope trn("trn.thread");
  EXPECT_EQ(cuda->backend(), "cuda");
  EXPECT_EQ(cuda->rank(), 4);
  EXPECT_EQ(trn->backend(), "trn");
  EXPECT_EQ(trn->rank(), 0);
  EXPECT_FALSE(cuda.same_as(trn));
  EXPECT_THROW(ScopeHigher(cuda, trn), ffi::Error);
  ThreadScope inner("thread_scope_test.inner");
  ThreadScope outer("thread_scope_test.outer");
  EXPECT_TRUE(inner.same_as(ThreadScope::Get(inner.name())));
  EXPECT_EQ(inner->backend(), "thread_scope_test");
  EXPECT_EQ(inner->local_name(), "inner");
  EXPECT_EQ(inner->rank(), 10);
  EXPECT_TRUE(ScopeHigher(outer, inner));
  EXPECT_FALSE(ScopeHigher(inner, outer));
  EXPECT_FALSE(ScopeHigher(inner, inner));
  EXPECT_GT(outer->_int_index, inner->_int_index);
  EXPECT_TRUE(ffi::Any(inner.name()).cast<ThreadScope>().same_as(inner));
  EXPECT_THROW(ThreadScope("thread_scope_test.missing"), ffi::Error);
  EXPECT_THROW(refl::EnumDef<ThreadScopeObj>(inner.name().c_str()), ffi::Error);
  EXPECT_TRUE(ffi::FromJSONGraph(ffi::ToJSONGraph(inner)).cast<ThreadScope>().same_as(inner));
}

TEST(ThreadScope, TypedCallAndDispatchWithoutPython) {
  RegisterTestScopes();
  TargetKindDef("thread_scope_test").set_default_device_type(kDLCPU);
  Target target("thread_scope_test");
  ThreadScope scope("thread_scope_test.inner");
  auto attrs = refl::ObjectCreator("tirx.tensor.ScopeAttrs")({{"scope", scope}}).cast<Attrs>();
  auto invoked = std::make_shared<int>(0);
  OpDef("thread_scope_test.instruction")
      .signature()
      .set_attr<TFixedReturnType>(op_attr::kFixedReturnType, PrimType::Void())
      .set_attr<tile::FTileOpDispatch>(
          tile::op_attr::kTileOpDispatch,
          [scope, invoked](const Call& call, const tile::DispatchContext& context) {
            EXPECT_TRUE(context->thread_scope.same_as(scope));
            EXPECT_EQ(context->scope_kind, "inner");
            EXPECT_TRUE(refl::FieldGetter(call->attrs->GetTypeKey(), "scope")(call->attrs.value())
                            .cast<ThreadScope>()
                            .same_as(scope));
            ++*invoked;
            return Evaluate(37);
          });
  Call call(PrimType::Void(), Op::Get("thread_scope_test.instruction"), {}, attrs);
  Function func({}, SeqStmt(Evaluate(call)), VoidType(), DictAttrs({{"target", target}}));
  GlobalVar main("main");
  auto result = transform::TileDispatch()(IRModule({{main, func}}));
  auto lowered = result->Lookup(main).as_or_throw<Function>();
  EXPECT_EQ(*invoked, 1);
  EXPECT_EQ(
      lowered->body.value()->seq[0].as_or_throw<Evaluate>()->value.as_or_throw<IntImm>()->value,
      37);
  EXPECT_THROW(refl::ObjectCreator("tirx.tensor.ScopeAttrs")({}), ffi::Error);
  EXPECT_THROW(refl::ObjectCreator("tirx.tensor.ScopeAttrs")({{"scope", ffi::String("inner")}}),
               ffi::Error);
}
}  // namespace
}  // namespace tvm::tirx
