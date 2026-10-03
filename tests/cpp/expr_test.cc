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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/source_map.h>
#include <tvm/relax/expr.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/function.h>

#include <type_traits>

namespace {
using namespace tvm;
template <typename Ref>
void CheckRequiredIRReference() {
  static_assert(!Ref::_type_is_nullable);
  static_assert(!std::is_default_constructible_v<Ref>);
  ffi::Any none;
  EXPECT_FALSE(none.as<Ref>().has_value());
  EXPECT_FALSE(none.try_cast<Ref>().has_value());
  EXPECT_THROW(none.as_or_throw<Ref>(), ffi::Error);
  EXPECT_THROW(none.cast<Ref>(), ffi::Error);
  EXPECT_FALSE(none.cast<ffi::Optional<Ref>>().has_value());
  ffi::Any missing_element = ffi::Array<ffi::Any>({none});
  EXPECT_THROW(missing_element.cast<ffi::Array<Ref>>(), ffi::Error);
  ffi::TypedFunction<Ref()> missing_result =
      ffi::Function::FromTyped([]() -> ffi::Any { return {}; });
  EXPECT_THROW(missing_result(), ffi::Error);
  ffi::TypedFunction<Ref(Ref)> required = [](Ref value) { return value; };
  EXPECT_THROW(required.packed()(none), ffi::Error);
  ffi::TypedFunction<ffi::Optional<Ref>(ffi::Optional<Ref>)> optional =
      [](ffi::Optional<Ref> value) { return value; };
  EXPECT_EQ(optional.packed()(none).type_index(), ffi::TypeIndex::kTVMFFINone);
}
}  // namespace

TEST(Expr, RequiredIRReferences) {
  using namespace tvm;
  CheckRequiredIRReference<Expr>();
  CheckRequiredIRReference<PrimExpr>();
  CheckRequiredIRReference<TypedExpr<PrimType>>();
  CheckRequiredIRReference<IntExpr>();
  CheckRequiredIRReference<PrimVar>();
  CheckRequiredIRReference<tirx::BufferVar>();
  CheckRequiredIRReference<tirx::Stmt>();
  CheckRequiredIRReference<tirx::Evaluate>();
  CheckRequiredIRReference<s_tir::SBlock>();
  CheckRequiredIRReference<tirx::PrimFunc>();
  CheckRequiredIRReference<relax::Function>();
  CheckRequiredIRReference<Type>();
  CheckRequiredIRReference<PrimType>();
  EXPECT_TRUE(Type::Missing().IsMissing());
  EXPECT_TRUE(ffi::Any(Type::Missing()).cast<Type>().IsMissing());
  EXPECT_THROW(ffi::Array<ffi::Any>({ffi::Any()}).as_or_throw<ffi::Array<Expr>>(), ffi::Error);
}

TEST(Expr, NonNullablePrimitiveFallbacks) {
  using namespace tvm;
  EXPECT_EQ(ffi::Any(false).cast<PrimExpr>().as<IntImmNode>()->value, 0);
  EXPECT_EQ(ffi::Any(0).cast<PrimExpr>().as<IntImmNode>()->value, 0);
  EXPECT_EQ(ffi::Any(1.25).cast<PrimExpr>().as<FloatImmNode>()->value, 1.25);
  EXPECT_TRUE(ffi::Any(0).cast<Expr>().defined());
  Var primitive("i", PrimType::Int(32));
  EXPECT_TRUE(ffi::Any(primitive).cast<PrimExpr>().same_as(primitive));
  EXPECT_TRUE(ffi::Any(primitive).cast<TypedExpr<PrimType>>().same_as(primitive));
  Var wrong_type("x", AnyType());
  EXPECT_FALSE(ffi::Any(wrong_type).try_cast<PrimExpr>().has_value());
  EXPECT_FALSE(ffi::Any(wrong_type).try_cast<TypedExpr<PrimType>>().has_value());
  auto tensor = te::placeholder({1}, PrimType::Float(32), "input");
  EXPECT_TRUE(ffi::Any(tensor(0)).cast<PrimExpr>().defined());
}

TEST(Expr, Basic) {
  using namespace tvm;
  using namespace tvm::tirx;
  PrimVar x("x");
  auto z = max(x + 1 + 2, 100);
  ffi::ObjectRef tmp = z;
  PrimExpr zz = tmp.as_or_throw<PrimExpr>();
  std::ostringstream os;
  os << z;
  TVM_FFI_ICHECK(zz.same_as(z));
  TVM_FFI_ICHECK(os.str() == "T.max(x + 1 + 2, 100)");
}

TEST(Expr, VarTypeAnnotation) {
  using namespace tvm;
  using namespace tvm::tirx;
  PrimVar x("x", PrimType::Float(32));
  PrimVar y("y", PrimType::Float(32));
  tvm::ffi::StructuralEqual checker;
  TVM_FFI_ICHECK(checker(x.ty(), y.ty()));
  TVM_FFI_ICHECK(checker(x->ty, y->ty));
}

TEST(Expr, VarCopyHelpers) {
  using namespace tvm;
  using namespace tvm::tirx;

  Span span(SourceName::Get("test.cc"), 1, 1, 1, 10);
  Type pointer_type = PointerType(PrimType::Float(32), "global");
  Var var("x", pointer_type, span);

  Var renamed = var.CopyWithName("y");
  EXPECT_FALSE(renamed.same_as(var));
  EXPECT_EQ(renamed->name, "y");
  EXPECT_TRUE(renamed->ty.same_as(pointer_type));
  EXPECT_TRUE(renamed->span.same_as(span));

  PrimType dtype = PrimType::Int(64);
  Var retyped = var.CopyWithDType(dtype);
  EXPECT_FALSE(retyped.same_as(var));
  EXPECT_EQ(retyped->name, "x");
  EXPECT_TRUE(retyped->ty.same_as(dtype));
  EXPECT_TRUE(retyped->span.same_as(span));

  PrimVar prim_var("i", PrimType::Int(32), span);
  static_assert(std::is_same_v<decltype(prim_var.CopyWithDType(PrimType::Float(32))), PrimVar>);
  PrimType prim_dtype = PrimType::Float(32);
  PrimVar retyped_prim_var = prim_var.CopyWithDType(prim_dtype);
  EXPECT_FALSE(retyped_prim_var.same_as(prim_var));
  EXPECT_EQ(retyped_prim_var->name, "i");
  EXPECT_TRUE(retyped_prim_var.ty().same_as(prim_dtype));
  EXPECT_TRUE(retyped_prim_var->span.same_as(span));
}

TEST(Expr, PrimTypeBoolLanes) {
  using namespace tvm;
  PrimType boolx4 = PrimType::Bool(4);
  TVM_FFI_ICHECK(boolx4.IsFixedLengthVector());
  TVM_FFI_ICHECK(boolx4.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_ICHECK_EQ(boolx4.lanes(), 4);
  TVM_FFI_ICHECK(boolx4.MatchesElementType(DLDataTypeCode::kDLBool, 8));
}

TEST(Expr, IntegerView) {
  using namespace tvm;
  for (PrimType ty : {PrimType::Int(8), PrimType::UInt(64), PrimType::Int(128)}) {
    ffi::Any value = Var("i", ty);
    EXPECT_TRUE(value.cast<IntExpr>().same_as(value.cast<PrimExpr>()));
    EXPECT_TRUE(value.cast<TypedExpr<PrimType>>().ty() == ty);
  }
  for (PrimType ty : {PrimType::Float(32), PrimType::Bool(), PrimType::Int(32, 4),
                      PrimType::ScalableVector(kDLInt, 32, 4)}) {
    ffi::Any value = Var("x", ty);
    EXPECT_FALSE(value.try_cast<IntExpr>());
    EXPECT_TRUE(value.try_cast<PrimExpr>());
  }
  ffi::Any nonprimitive = Var("x", AnyType());
  EXPECT_FALSE(nonprimitive.try_cast<IntExpr>());
  EXPECT_FALSE(nonprimitive.try_cast<PrimExpr>());
  EXPECT_FALSE(nonprimitive.try_cast<TypedExpr<PrimType>>());
  EXPECT_EQ(ffi::Any(1).cast<IntExpr>().ty().bits(), 32);
  EXPECT_EQ(ffi::Any(int64_t{1} << 40).cast<IntExpr>().ty().bits(), 64);
}

TEST(ExprNodeRef, Basic) {
  using namespace tvm;
  using namespace tvm::tirx;
  PrimVar x("x");
  PrimExpr z = max(x + 1 + 2, 100);
  const prim::MaxNode* op = z.as<prim::MaxNode>();
  TVM_FFI_ICHECK(ffi::GetRef<ffi::ObjectRef>(op).same_as(z));
}

TEST(Expr, DeepEqualTensorLoadSourceIdentity) {
  using namespace tvm;
  Var source("source", PointerType(PrimType::Float(32)));
  Var other_source("source", PointerType(PrimType::Float(32)));
  auto load = [](Expr source, PrimExpr index) {
    auto node = ffi::make_object<TensorLoadNode>(source);
    node->ty = PrimType::Float(32);
    node->indices = {index};
    return TensorLoad(node);
  };
  prim::ExprDeepEqual equal;
  EXPECT_TRUE(equal(load(source, 0), load(source, 0)));
  EXPECT_FALSE(equal(load(source, 0), load(other_source, 0)));
  EXPECT_FALSE(equal(load(source, 0), load(source, 1)));
}
