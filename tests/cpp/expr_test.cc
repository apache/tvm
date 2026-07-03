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
#include <tvm/runtime/logging.h>
#include <tvm/te/operation.h>

namespace {

template <typename LHS, typename RHS>
concept HasEqual = requires(const LHS& lhs, const RHS& rhs) { lhs == rhs; };

template <typename LHS, typename RHS>
concept HasNotEqual = requires(const LHS& lhs, const RHS& rhs) { lhs != rhs; };

template <typename LHS, typename RHS>
concept HasLess = requires(const LHS& lhs, const RHS& rhs) { lhs < rhs; };

static_assert(!HasEqual<tvm::Expr, tvm::Expr>);
static_assert(!HasNotEqual<tvm::Expr, tvm::Expr>);
static_assert(!HasLess<tvm::Expr, tvm::Expr>);
static_assert(!HasEqual<tvm::tirx::Var, tvm::tirx::Var>);
static_assert(!HasNotEqual<tvm::tirx::Var, tvm::tirx::Var>);
static_assert(!HasLess<tvm::tirx::Var, tvm::tirx::Var>);
static_assert(!HasEqual<tvm::tirx::Var, tvm::PrimExpr>);
static_assert(!HasEqual<tvm::PrimExpr, tvm::tirx::Var>);
static_assert(!HasNotEqual<tvm::tirx::Var, tvm::PrimExpr>);
static_assert(!HasNotEqual<tvm::PrimExpr, tvm::tirx::Var>);
static_assert(!HasLess<tvm::tirx::Var, tvm::PrimExpr>);
static_assert(!HasLess<tvm::PrimExpr, tvm::tirx::Var>);
static_assert(HasEqual<tvm::PrimExpr, tvm::PrimExpr>);
static_assert(HasNotEqual<tvm::PrimExpr, tvm::PrimExpr>);
static_assert(HasLess<tvm::PrimExpr, tvm::PrimExpr>);
static_assert(HasEqual<tvm::tirx::PrimVar, tvm::PrimExpr>);
static_assert(HasEqual<tvm::PrimExpr, tvm::tirx::PrimVar>);
static_assert(HasNotEqual<tvm::tirx::PrimVar, tvm::PrimExpr>);
static_assert(HasNotEqual<tvm::PrimExpr, tvm::tirx::PrimVar>);
static_assert(HasLess<tvm::tirx::PrimVar, tvm::PrimExpr>);
static_assert(HasLess<tvm::PrimExpr, tvm::tirx::PrimVar>);
static_assert(HasEqual<tvm::tirx::PrimVar, tvm::tirx::PrimVar>);
static_assert(HasNotEqual<tvm::tirx::PrimVar, tvm::tirx::PrimVar>);
static_assert(HasLess<tvm::tirx::PrimVar, tvm::tirx::PrimVar>);

}  // namespace

TEST(Expr, Basic) {
  using namespace tvm;
  using namespace tvm::tirx;
  Var x("x");
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
  Var x("x", PrimType::Float(32));
  Var y("y", PrimType::Float(32));
  tvm::ffi::StructuralEqual checker;
  TVM_FFI_ICHECK(checker(x.ty(), y.ty()));
  TVM_FFI_ICHECK(checker(x->ty, y->ty));
}

TEST(Expr, PrimVarCheckedView) {
  using namespace tvm;
  using namespace tvm::tirx;

  Var scalar("x", PrimType::Int(32));
  PrimVar prim_var = scalar.as_or_throw<PrimVar>();
  Var widened = prim_var;
  EXPECT_TRUE(prim_var.same_as(scalar));
  EXPECT_TRUE(widened.same_as(scalar));

  Var pointer("p", PointerType(PrimType::Float(32)));
  EXPECT_THROW(pointer.as_or_throw<PrimVar>(), ffi::Error);
}

TEST(Expr, PrimTypeBoolLanes) {
  using namespace tvm;
  PrimType boolx4 = PrimType::Bool(4);
  TVM_FFI_ICHECK(boolx4.IsFixedLengthVector());
  TVM_FFI_ICHECK(boolx4.MatchesCode(DLDataTypeCode::kDLBool));
  TVM_FFI_ICHECK_EQ(boolx4.lanes(), 4);
  TVM_FFI_ICHECK(boolx4.MatchesElementType(DLDataTypeCode::kDLBool, 8));
}

TEST(ExprNodeRef, Basic) {
  using namespace tvm;
  using namespace tvm::tirx;
  Var x("x");
  PrimExpr z = max(x + 1 + 2, 100);
  const tirx::MaxNode* op = z.as<tirx::MaxNode>();
  TVM_FFI_ICHECK(ffi::GetRef<ffi::ObjectRef>(op).same_as(z));
}
