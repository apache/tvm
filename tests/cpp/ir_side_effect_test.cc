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
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace {

TVM_REGISTER_OP("test.side_effect.pure")
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
TVM_REGISTER_OP("test.side_effect.read")
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kReadState));
TVM_REGISTER_OP("test.side_effect.update")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kUpdateState));
TVM_REGISTER_OP("test.side_effect.annotation")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kExprAnnotation));
TVM_REGISTER_OP("test.side_effect.control")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kControlJump));
TVM_REGISTER_OP("test.side_effect.missing");

PrimExpr EffectCall(const char* kind, ffi::Array<Expr> args = {}) {
  return Call(PrimType::Int(32), Op::Get(std::string("test.side_effect.") + kind), args)
      .as_or_throw<PrimExpr>();
}

std::string EffectError(const Expr& expr) {
  try {
    SideEffect(expr);
  } catch (const ffi::Error& error) {
    return std::string(error.kind()) + ":" + std::string(error.message());
  }
  ADD_FAILURE() << "Expected an effect attribute error";
  return "";
}

TEST(SideEffect, SharedExpressionProperties) {
  auto buf = tirx::decl_buffer({16}, PrimType::Float(32));
  auto i = PrimVar("i", PrimType::Int(32));
  EXPECT_EQ(SideEffect(tirx::BufferLoad(buf, {i})), CallEffectKind::kReadState);
  EXPECT_EQ(SideEffect(exp(prim::Cast(PrimType::Float(32), i + 1))), CallEffectKind::kPure);
  EXPECT_EQ(SideEffect(Call(PrimType::Void(), tirx::builtin::tvm_storage_sync(), {})),
            CallEffectKind::kUpdateState);
  EXPECT_EQ(SideEffect(EffectCall("annotation", {i})), CallEffectKind::kPure);
  EXPECT_EQ(SideEffect(EffectCall("control")), CallEffectKind::kUpdateState);
  EXPECT_EQ(SideEffect(EffectCall("pure", {EffectCall("read"), i})), CallEffectKind::kReadState);
  EXPECT_EQ(SideEffect(EffectCall("pure", {EffectCall("read"), EffectCall("update")})),
            CallEffectKind::kUpdateState);
  // Shared expression roots do not need a primitive result type.
  EXPECT_EQ(SideEffect(Tuple({i, EffectCall("read")})), CallEffectKind::kReadState);
  EXPECT_EQ(SideEffect(Call(OpaqueType(), Op::Get("test.side_effect.read"), {})),
            CallEffectKind::kReadState);
}

TEST(SideEffect, OpaqueAndMissingAttributes) {
  auto missing = EffectCall("missing");
  auto update = EffectCall("update");
  auto unknown = Var("callee", OpaqueType());
  EXPECT_EQ(SideEffect(Call(OpaqueType(), unknown, {missing})), CallEffectKind::kUpdateState);
  EXPECT_EQ(SideEffect(EffectCall("pure", {update, missing})), CallEffectKind::kUpdateState);
  EXPECT_EQ(EffectError(EffectCall("pure", {missing, update})), EffectError(missing));
}

TEST(SideEffect, ExcludesTypesAndConstantMetadata) {
  auto missing = EffectCall("missing");
  auto buf = tirx::decl_buffer({missing}, PrimType::Int(32));
  EXPECT_EQ(SideEffect(buf), CallEffectKind::kPure);
  EXPECT_EQ(SideEffect(tirx::BufferLoad(buf, {IntImm::Int32(0)})), CallEffectKind::kReadState);
  EXPECT_EQ(SideEffect(Call(buf.type(), Op::Get("test.side_effect.pure"), {})),
            CallEffectKind::kPure);
  EXPECT_EQ(SideEffect(Call(PrimType::Int(32), Op::Get("test.side_effect.pure"), {},
                            DictAttrs({{"metadata", missing}}), {buf.type()})),
            CallEffectKind::kPure);
  auto opaque = ffi::make_object<OpaqueExprNode>();
  opaque->ty = buf.type();
  EXPECT_EQ(SideEffect(OpaqueExpr(opaque)), CallEffectKind::kPure);
  auto tensor = te::placeholder({missing}, PrimType::Int(32), "A");
  EXPECT_EQ(SideEffect(tensor), CallEffectKind::kPure);
  EXPECT_EQ(SideEffect(tensor(IntImm::Int32(0))), CallEffectKind::kReadState);
  // An index remains an evaluated child, even when source metadata is excluded.
  EXPECT_EQ(EffectError(tensor(missing)), EffectError(missing));
  EXPECT_EQ(EffectError(tirx::BufferLoad(buf, {missing})), EffectError(missing));
}

TEST(SideEffect, ShufflePreservesIndicesFirstOrder) {
  auto missing = EffectCall("missing");
  auto update = EffectCall("update");
  EXPECT_EQ(EffectError(prim::Shuffle({update}, {missing})), EffectError(missing));
  EXPECT_EQ(SideEffect(prim::Shuffle({missing}, {update})), CallEffectKind::kUpdateState);
}

TEST(SideEffect, ExcludesVectorLaneMetadata) {
  auto missing = EffectCall("missing");
  // Object fields can be reconstructed by reflection; lane metadata stays excluded.
  auto ramp = ffi::make_object<prim::RampNode>();
  ramp->ty = PrimType::Int(32, 4);
  ramp->base = IntImm::Int32(0);
  ramp->stride = IntImm::Int32(1);
  ramp->lanes = missing;
  EXPECT_EQ(SideEffect(prim::Ramp(ramp)), CallEffectKind::kPure);
  auto broadcast = ffi::make_object<prim::BroadcastNode>();
  broadcast->ty = PrimType::Int(32, 4);
  broadcast->value = EffectCall("read");
  broadcast->lanes = missing;
  EXPECT_EQ(SideEffect(prim::Broadcast(broadcast)), CallEffectKind::kReadState);
  EXPECT_EQ(SideEffect(prim::Ramp(IntImm::Int32(0), IntImm::Int32(1), 4)), CallEffectKind::kPure);
  auto lanes =
      prim::Mul(Call(PrimType::Int(32), prim::builtin::vscale(), {}).as_or_throw<PrimExpr>(),
                IntImm::Int32(4));
  EXPECT_EQ(SideEffect(prim::Ramp(IntImm::Int32(0), IntImm::Int32(1), lanes)),
            CallEffectKind::kPure);
  EXPECT_EQ(SideEffect(prim::Broadcast(EffectCall("read"), lanes)), CallEffectKind::kReadState);
}

TEST(TETensorLoad, RegisteredRepresentationAndValidation) {
  auto tensor = te::placeholder({8}, PrimType::Float(32), "A");
  auto index = PrimVar("i", PrimType::Int(32));
  auto load = tensor(index).as_or_throw<Call>();
  EXPECT_TRUE(load->op.same_as(Op::Get("te.tensor_load")));
  EXPECT_TRUE(load->args[0].same_as(tensor));
  EXPECT_TRUE(te::IsTensorLoad(load));
  EXPECT_TRUE(te::GetTensorFromLoad(load).same_as(tensor));
  EXPECT_TRUE(te::GetTensorLoadIndices(load)[0].same_as(index));
  EXPECT_EQ(SideEffect(load), CallEffectKind::kReadState);
  auto scalar = te::placeholder({}, PrimType::Float(32), "scalar");
  auto scalar_load = scalar(ffi::Array<PrimExpr>{}).as_or_throw<Call>();
  EXPECT_EQ(scalar_load->args.size(), 1U);
  EXPECT_TRUE(te::GetTensorLoadIndices(scalar_load).empty());
  auto vector_tensor = te::placeholder({8}, PrimType::Float(32, 4), "vector");
  auto vector_load = vector_tensor(index).as_or_throw<Call>();
  EXPECT_EQ(vector_load->ty, vector_tensor->dtype);
  EXPECT_TRUE(te::GetTensorFromLoad(vector_load).same_as(vector_tensor));
  auto vector_index = prim::Ramp(IntImm::Int32(0), IntImm::Int32(1), 4);
  auto indexed = tensor(vector_index).as_or_throw<Call>();
  EXPECT_TRUE(te::GetTensorLoadIndices(indexed)[0].same_as(vector_index));
  EXPECT_EQ(indexed->ty, tensor->dtype);

  auto op = Op::Get("te.tensor_load");
  auto malformed = [&](Type result, ffi::Array<Expr> args) {
    auto call = Call(result, op, args);
    EXPECT_TRUE(te::IsTensorLoad(call));
    EXPECT_THROW(te::GetTensorFromLoad(call), ffi::Error);
    EXPECT_THROW(te::GetTensorLoadIndices(call), ffi::Error);
  };
  malformed(tensor->dtype, {});
  malformed(tensor->dtype, {index, index});
  malformed(tensor->dtype, {tensor});
  malformed(tensor->dtype, {tensor, index, index});
  malformed(tensor->dtype, {tensor, tensor});
  malformed(PrimType::Int(32), {tensor, index});
  malformed(OpaqueType(), {tensor, index});
  EXPECT_FALSE(te::IsTensorLoad(Call(tensor->dtype, tensor, {index})));
  EXPECT_FALSE(te::IsTensorLoad(tirx::BufferLoad(tirx::decl_buffer({8}), {index})));
  EXPECT_THROW(te::GetTensorFromLoad(Call(tensor->dtype, tensor, {index})), ffi::Error);

  auto negative =
      tensor.IndexWithNegativeIndices(ffi::Array<PrimExpr>{IntImm::Int32(-1)}).as_or_throw<Call>();
  auto normalized = te::GetTensorLoadIndices(negative)[0].as<prim::SelectNode>();
  ASSERT_NE(normalized, nullptr);
  ASSERT_NE(normalized->true_value.as<IntImmNode>(), nullptr);
  EXPECT_EQ(normalized->true_value.as<IntImmNode>()->value, 7);
  EXPECT_EQ(normalized->false_value.as<IntImmNode>()->value, -1);
}

}  // namespace
}  // namespace tvm
