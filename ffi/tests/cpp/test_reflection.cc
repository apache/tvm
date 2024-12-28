
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
#include <tvm/ffi/reflection.h>

#include "./testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

struct A {
  ObjectRef obj;
  int32_t x;
  int32_t y;
};

TEST(Reflection, GetFieldByteOffset) {
  EXPECT_EQ(details::GetFieldByteOffset(&A::x), 8);
  EXPECT_EQ(details::GetFieldByteOffset(&A::y), 12);
}

TEST(Reflection, FieldGetter) {
  ObjectRef a = TInt(10);
  details::ReflectionFieldGetter getter(details::GetReflectionFieldInfo("test.Int", "value"));
  EXPECT_EQ(getter(a).operator int(), 10);
}
}  // namespace
