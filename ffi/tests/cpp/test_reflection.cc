
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
#include <tvm/ffi/object.h>
#include <tvm/ffi/reflection/reflection.h>
#include <tvm/ffi/string.h>

#include "./testing_object.h"

namespace {
using namespace tvm::ffi;
using namespace tvm::ffi::testing;

TVM_FFI_REFLECTION_DEF(TFloatObj)
    .def_rw("value", &TFloatObj::value, "float value field", refl::DefaultValue(10.0))
    .def("sub", [](const TFloatObj* self, double other) -> double { return self->value - other; })
    .def("add", &TFloatObj::Add, "add method");

TVM_FFI_REFLECTION_DEF(TIntObj)
    .def_ro("value", &TIntObj::value)
    .def_static("static_add", &TInt::StaticAdd, "static add method");

TVM_FFI_REFLECTION_DEF(TPrimExprObj)
    .def_ro("dtype", &TPrimExprObj::dtype, "dtype field", refl::DefaultValue("float"))
    .def_ro("value", &TPrimExprObj::value, "value field", refl::DefaultValue(0))
    .def("sub", [](TPrimExprObj* self, double other) -> double {
      // this is ok because TPrimExprObj is declared asmutable
      return self->value - other;
    });

struct A : public Object {
  int64_t x;
  int64_t y;
};

TVM_FFI_REFLECTION_DEF(A).def_ro("x", &A::x).def_rw("y", &A::y);

TEST(Reflection, GetFieldByteOffset) {
  EXPECT_EQ(reflection::GetFieldByteOffsetToObject(&A::x), sizeof(TVMFFIObject));
  EXPECT_EQ(reflection::GetFieldByteOffsetToObject(&A::y), 8 + sizeof(TVMFFIObject));
  EXPECT_EQ(reflection::GetFieldByteOffsetToObject(&TIntObj::value), sizeof(TVMFFIObject));
}

TEST(Reflection, FieldGetter) {
  ObjectRef a = TInt(10);
  reflection::FieldGetter getter("test.Int", "value");
  EXPECT_EQ(getter(a).cast<int>(), 10);

  ObjectRef b = TFloat(10.0);
  reflection::FieldGetter getter_float("test.Float", "value");
  EXPECT_EQ(getter_float(b).cast<double>(), 10.0);
}

TEST(Reflection, FieldSetter) {
  ObjectRef a = TFloat(10.0);
  reflection::FieldSetter setter("test.Float", "value");
  setter(a, 20.0);
  EXPECT_EQ(a.as<TFloatObj>()->value, 20.0);
}

TEST(Reflection, FieldInfo) {
  const TVMFFIFieldInfo* info_int = reflection::GetFieldInfo("test.Int", "value");
  EXPECT_FALSE(info_int->flags & TVMFFIFieldFlagBitMaskHasDefault);
  EXPECT_FALSE(info_int->flags & TVMFFIFieldFlagBitMaskWritable);
  EXPECT_EQ(Bytes(info_int->doc).operator std::string(), "");

  const TVMFFIFieldInfo* info_float = reflection::GetFieldInfo("test.Float", "value");
  EXPECT_EQ(info_float->default_value.v_float64, 10.0);
  EXPECT_TRUE(info_float->flags & TVMFFIFieldFlagBitMaskHasDefault);
  EXPECT_TRUE(info_float->flags & TVMFFIFieldFlagBitMaskWritable);
  EXPECT_EQ(Bytes(info_float->doc).operator std::string(), "float value field");

  const TVMFFIFieldInfo* info_prim_expr_dtype = reflection::GetFieldInfo("test.PrimExpr", "dtype");
  AnyView default_value = AnyView::CopyFromTVMFFIAny(info_prim_expr_dtype->default_value);
  EXPECT_EQ(default_value.cast<String>(), "float");
  EXPECT_EQ(default_value.as<String>().value().use_count(), 2);
  EXPECT_TRUE(info_prim_expr_dtype->flags & TVMFFIFieldFlagBitMaskHasDefault);
  EXPECT_FALSE(info_prim_expr_dtype->flags & TVMFFIFieldFlagBitMaskWritable);
  EXPECT_EQ(Bytes(info_prim_expr_dtype->doc).operator std::string(), "dtype field");
}

TEST(Reflection, MethodInfo) {
  const TVMFFIMethodInfo* info_int_static_add = reflection::GetMethodInfo("test.Int", "static_add");
  EXPECT_TRUE(info_int_static_add->flags & TVMFFIFieldFlagBitMaskIsStaticMethod);
  EXPECT_EQ(Bytes(info_int_static_add->doc).operator std::string(), "static add method");

  const TVMFFIMethodInfo* info_float_add = reflection::GetMethodInfo("test.Float", "add");
  EXPECT_FALSE(info_float_add->flags & TVMFFIFieldFlagBitMaskIsStaticMethod);
  EXPECT_EQ(Bytes(info_float_add->doc).operator std::string(), "add method");

  const TVMFFIMethodInfo* info_float_sub = reflection::GetMethodInfo("test.Float", "sub");
  EXPECT_FALSE(info_float_sub->flags & TVMFFIFieldFlagBitMaskIsStaticMethod);
  EXPECT_EQ(Bytes(info_float_sub->doc).operator std::string(), "");
}

TEST(Reflection, CallMethod) {
  Function static_int_add = reflection::GetMethod("test.Int", "static_add");
  EXPECT_EQ(static_int_add(TInt(1), TInt(2)).cast<TInt>()->value, 3);

  Function float_add = reflection::GetMethod("test.Float", "add");
  EXPECT_EQ(float_add(TFloat(1), 2.0).cast<double>(), 3.0);

  Function float_sub = reflection::GetMethod("test.Float", "sub");
  EXPECT_EQ(float_sub(TFloat(1), 2.0).cast<double>(), -1.0);

  Function prim_expr_sub = reflection::GetMethod("test.PrimExpr", "sub");
  EXPECT_EQ(prim_expr_sub(TPrimExpr("float", 1), 2.0).cast<double>(), -1.0);
}

}  // namespace
