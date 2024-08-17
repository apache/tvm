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
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>

#include "./testing_object.h"

namespace {

using namespace tvm::ffi;
using namespace tvm::ffi::testing;

TEST(Object, RefCounter) {
  ObjectPtr<TIntObj> a = make_object<TIntObj>(11);
  ObjectPtr<TIntObj> b = a;

  EXPECT_EQ(a->value, 11);

  EXPECT_EQ(a.use_count(), 2);
  b.reset();
  EXPECT_EQ(a.use_count(), 1);
  EXPECT_TRUE(b == nullptr);
  EXPECT_EQ(b.use_count(), 0);

  ObjectPtr<TIntObj> c = std::move(a);
  EXPECT_EQ(c.use_count(), 1);
  EXPECT_TRUE(a == nullptr);

  EXPECT_EQ(c->value, 11);
}

TEST(Object, TypeInfo) {
  const TypeInfo* info = tvm::ffi::details::ObjectGetTypeInfo(TIntObj::RuntimeTypeIndex());
  EXPECT_TRUE(info != nullptr);
  EXPECT_EQ(info->type_index, TIntObj::RuntimeTypeIndex());
  EXPECT_EQ(info->type_depth, 2);
  EXPECT_EQ(info->type_acenstors[0], Object::_type_index);
  EXPECT_EQ(info->type_acenstors[1], TNumberObj::_type_index);
  EXPECT_GE(info->type_index, TypeIndex::kTVMFFIDynObjectBegin);
}

TEST(Object, InstanceCheck) {
  ObjectPtr<Object> a = make_object<TIntObj>(11);
  ObjectPtr<Object> b = make_object<TFloatObj>(11);

  EXPECT_TRUE(a->IsInstance<Object>());
  EXPECT_TRUE(a->IsInstance<TNumberObj>());
  EXPECT_TRUE(a->IsInstance<TIntObj>());
  EXPECT_TRUE(!a->IsInstance<TFloatObj>());

  EXPECT_TRUE(a->IsInstance<Object>());
  EXPECT_TRUE(b->IsInstance<TNumberObj>());
  EXPECT_TRUE(!b->IsInstance<TIntObj>());
  EXPECT_TRUE(b->IsInstance<TFloatObj>());
}

TEST(ObjectRef, as) {
  ObjectRef a = TInt(10);
  ObjectRef b = TFloat(20);
  // nullable object
  ObjectRef c(nullptr);

  EXPECT_TRUE(a.as<TIntObj>() != nullptr);
  EXPECT_TRUE(a.as<TFloatObj>() == nullptr);
  EXPECT_TRUE(a.as<TNumberObj>() != nullptr);

  EXPECT_TRUE(b.as<TIntObj>() == nullptr);
  EXPECT_TRUE(b.as<TFloatObj>() != nullptr);
  EXPECT_TRUE(b.as<TNumberObj>() != nullptr);

  EXPECT_TRUE(c.as<TIntObj>() == nullptr);
  EXPECT_TRUE(c.as<TFloatObj>() == nullptr);
  EXPECT_TRUE(c.as<TNumberObj>() == nullptr);

  EXPECT_EQ(a.as<TIntObj>()->value, 10);
  EXPECT_EQ(b.as<TFloatObj>()->value, 20);
}

}  // namespace
