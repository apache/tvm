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

#ifndef TVM_IR_PRIM_OP_UTILS_H_
#define TVM_IR_PRIM_OP_UTILS_H_
#include <tvm/ir/prim/op.h>
namespace tvm::prim::detail {
TVM_FFI_INLINE const PrimTypeNode* GetPrimTypeNode(const PrimExpr& expr) {
  // Avoid PrimExpr::ty() ObjectRef materialization on binary operator hot paths.
  const auto* node = expr.get();
  TVM_FFI_DCHECK(node != nullptr);
  TVM_FFI_DCHECK(!node->ExprNode::ty.IsMissing());
  const auto* prim_ty = node->ExprNode::ty.as<PrimTypeNode>();
  TVM_FFI_DCHECK(prim_ty != nullptr);
  return prim_ty;
}

TVM_FFI_INLINE bool IsFloatType(const PrimType& ty) {
  return ty.MatchesCode(DLDataTypeCode::kDLFloat);
}

TVM_FFI_INLINE bool IsBFloat16Type(const PrimType& ty) {
  return ty.MatchesCode(DLDataTypeCode::kDLBfloat);
}

TVM_FFI_INLINE bool IsFloat8Type(const PrimType& ty) {
  return ty.MatchesCode(DLDataTypeCode::kDLFloat8_e3m4, DLDataTypeCode::kDLFloat8_e4m3,
                        DLDataTypeCode::kDLFloat8_e4m3b11fnuz, DLDataTypeCode::kDLFloat8_e4m3fn,
                        DLDataTypeCode::kDLFloat8_e4m3fnuz, DLDataTypeCode::kDLFloat8_e5m2,
                        DLDataTypeCode::kDLFloat8_e5m2fnuz, DLDataTypeCode::kDLFloat8_e8m0fnu);
}

TVM_FFI_INLINE bool IsFloat6Type(const PrimType& ty) {
  return ty.MatchesCode(DLDataTypeCode::kDLFloat6_e2m3fn, DLDataTypeCode::kDLFloat6_e3m2fn);
}

TVM_FFI_INLINE bool IsFloat4Type(const PrimType& ty) {
  return ty.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn);
}
TVM_DLL void BinaryOpMatchTypes(PrimExpr& lhs, PrimExpr& rhs, Span span);
inline void type_check_boolean_args(const PrimExpr& arg, const char* op) {
  TVM_FFI_ICHECK(arg.ty().MatchesCode(DLDataTypeCode::kDLBool))
      << "Expected boolean argument for " << op << ", but received " << arg << " of type "
      << arg.ty();
}
inline void type_check_boolean_args(const PrimExpr& lhs, const PrimExpr& rhs, const char* op) {
  TVM_FFI_ICHECK(lhs.ty().MatchesCode(DLDataTypeCode::kDLBool))
      << "Expected boolean argument as LHS of " << op << ", but received " << lhs << " of type "
      << lhs.ty();
  TVM_FFI_ICHECK(rhs.ty().MatchesCode(DLDataTypeCode::kDLBool))
      << "Expected boolean argument as RHS of " << op << ", but received " << rhs << " of type "
      << rhs.ty();
}

inline void type_check_int_or_bool_args(const PrimExpr& arg, const char* op) {
  TVM_FFI_ICHECK(arg.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                                      DLDataTypeCode::kDLBool))
      << "Expected integer or boolean argument for " << op << ", but received " << arg
      << " of type " << arg.ty();
}

inline void type_check_integer_args(const PrimExpr& lhs, const PrimExpr& rhs, const char* op) {
  TVM_FFI_ICHECK(lhs.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt))
      << "Expected integer argument as LHS of " << op << ", but received " << lhs << " of type "
      << lhs.ty();
  TVM_FFI_ICHECK(rhs.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt))
      << "Expected integer argument as RHS of " << op << ", but received " << rhs << " of type "
      << rhs.ty();
}

inline void type_check_int_or_bool_args(const PrimExpr& lhs, const PrimExpr& rhs, const char* op) {
  TVM_FFI_ICHECK(lhs.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                                      DLDataTypeCode::kDLBool))
      << "Expected integer argument as LHS of " << op << ", but received " << lhs << " of type "
      << lhs.ty();
  TVM_FFI_ICHECK(rhs.ty().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt,
                                      DLDataTypeCode::kDLBool))
      << "Expected integer argument as RHS of " << op << ", but received " << rhs << " of type "
      << rhs.ty();
}

}  // namespace tvm::prim::detail
#endif  // TVM_IR_PRIM_OP_UTILS_H_
