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
  ObjectPtr<TIntObj> aa = make_object<TIntObj>(*a);
  EXPECT_EQ(aa.use_count(), 1);
  EXPECT_EQ(aa->value, 11);

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
  const TypeInfo* info = TVMFFIGetTypeInfo(TIntObj::RuntimeTypeIndex());
  EXPECT_TRUE(info != nullptr);
  EXPECT_EQ(info->type_index, TIntObj::RuntimeTypeIndex());
  EXPECT_EQ(info->type_depth, 2);
  EXPECT_EQ(info->type_acenstors[0]->type_index, Object::_type_index);
  EXPECT_EQ(info->type_acenstors[1]->type_index, TNumberObj::_type_index);
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

TEST(Object, CAPIAccessor) {
  ObjectRef a = TInt(10);
  TVMFFIObjectHandle obj = details::ObjectUnsafe::RawObjectPtrFromObjectRef(a);
  int32_t type_index = TVMFFIObjectGetTypeIndex(obj);
  EXPECT_EQ(type_index, TIntObj::RuntimeTypeIndex());
}

TEST(Object, WeakObjectPtr) {
  // Test basic construction from ObjectPtr
  ObjectPtr<TIntObj> strong_ptr = make_object<TIntObj>(42);
  WeakObjectPtr<TIntObj> weak_ptr(strong_ptr);

  EXPECT_EQ(strong_ptr.use_count(), 1);
  EXPECT_FALSE(weak_ptr.expired());
  EXPECT_EQ(weak_ptr.use_count(), 1);

  // Test lock() when object is still alive
  ObjectPtr<TIntObj> locked_ptr = weak_ptr.lock();
  EXPECT_TRUE(locked_ptr != nullptr);
  EXPECT_EQ(locked_ptr->value, 42);
  EXPECT_EQ(strong_ptr.use_count(), 2);
  EXPECT_EQ(weak_ptr.use_count(), 2);

  // Test lock() when object is expired
  strong_ptr.reset();
  locked_ptr.reset();
  EXPECT_TRUE(weak_ptr.expired());
  EXPECT_EQ(weak_ptr.use_count(), 0);

  ObjectPtr<TIntObj> expired_lock = weak_ptr.lock();
  EXPECT_TRUE(expired_lock == nullptr);
}

TEST(Object, WeakObjectPtrAssignment) {
  // Test copy construction
  ObjectPtr<TIntObj> new_strong = make_object<TIntObj>(100);
  WeakObjectPtr<TIntObj> weak1(new_strong);
  WeakObjectPtr<TIntObj> weak2(weak1);

  EXPECT_EQ(new_strong.use_count(), 1);
  EXPECT_FALSE(weak1.expired());
  EXPECT_FALSE(weak2.expired());
  EXPECT_EQ(weak1.use_count(), 1);
  EXPECT_EQ(weak2.use_count(), 1);

  // Test move construction
  WeakObjectPtr<TIntObj> weak3(std::move(weak1));
  EXPECT_TRUE(weak1.expired());  // weak1 should be moved from
  EXPECT_FALSE(weak3.expired());
  EXPECT_EQ(weak3.use_count(), 1);

  // Test assignment
  WeakObjectPtr<TIntObj> weak4;
  weak4 = weak2;
  EXPECT_FALSE(weak2.expired());
  EXPECT_FALSE(weak4.expired());
  EXPECT_EQ(weak2.use_count(), 1);
  EXPECT_EQ(weak4.use_count(), 1);

  // Test move assignment
  WeakObjectPtr<TIntObj> weak5;
  weak5 = std::move(weak2);
  EXPECT_TRUE(weak2.expired());  // weak2 should be moved from
  EXPECT_FALSE(weak5.expired());
  EXPECT_EQ(weak5.use_count(), 1);

  // Test reset()
  weak3.reset();
  EXPECT_TRUE(weak3.expired());
  EXPECT_EQ(weak3.use_count(), 0);

  // Test swap()
  ObjectPtr<TIntObj> strong_a = make_object<TIntObj>(200);
  ObjectPtr<TIntObj> strong_b = make_object<TIntObj>(300);
  WeakObjectPtr<TIntObj> weak_a(strong_a);
  WeakObjectPtr<TIntObj> weak_b(strong_b);

  weak_a.swap(weak_b);
  EXPECT_EQ(weak_a.lock()->value, 300);
  EXPECT_EQ(weak_b.lock()->value, 200);

  // Test construction from nullptr
  WeakObjectPtr<TIntObj> null_weak(nullptr);
  EXPECT_TRUE(null_weak.expired());
  EXPECT_EQ(null_weak.use_count(), 0);
  EXPECT_TRUE(null_weak.lock() == nullptr);

  // Test inheritance compatibility
  ObjectPtr<TNumberObj> number_ptr = make_object<TIntObj>(500);
  WeakObjectPtr<TNumberObj> number_weak(number_ptr);

  EXPECT_FALSE(number_weak.expired());
  EXPECT_EQ(number_weak.use_count(), 1);

  // Test that weak references don't prevent object deletion
  ObjectPtr<TIntObj> temp_strong = make_object<TIntObj>(999);
  WeakObjectPtr<TIntObj> temp_weak(temp_strong);

  EXPECT_FALSE(temp_weak.expired());
  temp_strong.reset();
  EXPECT_TRUE(temp_weak.expired());
  EXPECT_TRUE(temp_weak.lock() == nullptr);

  // Test multiple weak references
  ObjectPtr<TIntObj> multi_strong = make_object<TIntObj>(777);
  WeakObjectPtr<TIntObj> multi_weak1(multi_strong);
  WeakObjectPtr<TIntObj> multi_weak2(multi_strong);
  WeakObjectPtr<TIntObj> multi_weak3(multi_strong);

  EXPECT_EQ(multi_strong.use_count(), 1);
  EXPECT_FALSE(multi_weak1.expired());
  EXPECT_FALSE(multi_weak2.expired());
  EXPECT_FALSE(multi_weak3.expired());

  // All weak references should be able to lock
  ObjectPtr<TIntObj> lock1 = multi_weak1.lock();
  ObjectPtr<TIntObj> lock2 = multi_weak2.lock();
  ObjectPtr<TIntObj> lock3 = multi_weak3.lock();

  EXPECT_EQ(multi_strong.use_count(), 4);
  EXPECT_EQ(lock1->value, 777);
  EXPECT_EQ(lock2->value, 777);
  EXPECT_EQ(lock3->value, 777);
}

TEST(Object, OpaqueObject) {
  thread_local int deleter_trigger_counter = 0;
  struct DummyOpaqueObject {
    int value;
    DummyOpaqueObject(int value) : value(value) {}

    static void Deleter(void* handle) {
      deleter_trigger_counter++;
      delete static_cast<DummyOpaqueObject*>(handle);
    }
  };
  TVMFFIObjectHandle handle = nullptr;
  TVM_FFI_CHECK_SAFE_CALL(TVMFFIObjectCreateOpaque(new DummyOpaqueObject(10), kTVMFFIOpaquePyObject,
                                                   DummyOpaqueObject::Deleter, &handle));
  ObjectPtr<Object> a =
      details::ObjectUnsafe::ObjectPtrFromOwned<Object>(static_cast<Object*>(handle));
  EXPECT_EQ(a->type_index(), kTVMFFIOpaquePyObject);
  EXPECT_EQ(static_cast<DummyOpaqueObject*>(TVMFFIOpaqueObjectGetCellPtr(a.get())->handle)->value,
            10);
  EXPECT_EQ(a.use_count(), 1);
  EXPECT_EQ(deleter_trigger_counter, 0);
  a.reset();
  EXPECT_EQ(deleter_trigger_counter, 1);
}

}  // namespace
