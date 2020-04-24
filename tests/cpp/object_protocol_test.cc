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

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/memory.h>

namespace tvm {
namespace test {

using namespace tvm::runtime;

class ObjBase : public Object {
 public:
  // dynamically allocate slow
  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const uint32_t _type_child_slots = 1;
  static constexpr const char* _type_key = "test.ObjBase";
  TVM_DECLARE_BASE_OBJECT_INFO(ObjBase, Object);
};

class ObjA : public ObjBase {
 public:
  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const uint32_t _type_child_slots = 0;
  static constexpr const char* _type_key = "test.ObjA";
  TVM_DECLARE_BASE_OBJECT_INFO(ObjA, ObjBase);
};

class ObjB : public ObjBase {
 public:
  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const uint32_t _type_child_slots = 0;
  static constexpr const char* _type_key = "test.ObjB";
  TVM_DECLARE_BASE_OBJECT_INFO(ObjB, ObjBase);
};

class ObjAA : public ObjA {
 public:
  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "test.ObjAA";
  TVM_DECLARE_FINAL_OBJECT_INFO(ObjAA, ObjA);
};


TVM_REGISTER_OBJECT_TYPE(ObjBase);
TVM_REGISTER_OBJECT_TYPE(ObjA);
TVM_REGISTER_OBJECT_TYPE(ObjB);
TVM_REGISTER_OBJECT_TYPE(ObjAA);

}  // namespace test
}  // namespace tvm

TEST(ObjectHierachy, Basic) {
  using namespace tvm::runtime;
  using namespace tvm::test;

  ObjectRef refA(make_object<ObjA>());
  CHECK_EQ(refA->type_index(), ObjA::RuntimeTypeIndex());
  CHECK(refA.as<Object>() != nullptr);
  CHECK(refA.as<ObjA>() != nullptr);
  CHECK(refA.as<ObjBase>() != nullptr);
  CHECK(refA.as<ObjB>() == nullptr);
  CHECK(refA.as<ObjAA>() == nullptr);

  ObjectRef refAA(make_object<ObjAA>());
  CHECK_EQ(refAA->type_index(), ObjAA::RuntimeTypeIndex());
  CHECK(refAA.as<Object>() != nullptr);
  CHECK(refAA.as<ObjBase>() != nullptr);
  CHECK(refAA.as<ObjA>() != nullptr);
  CHECK(refAA.as<ObjAA>() != nullptr);
  CHECK(refAA.as<ObjB>() == nullptr);

  ObjectRef refB(make_object<ObjB>());
  CHECK_EQ(refB->type_index(), ObjB::RuntimeTypeIndex());
  CHECK(refB.as<Object>() != nullptr);
  CHECK(refB.as<ObjBase>() != nullptr);
  CHECK(refB.as<ObjA>() == nullptr);
  CHECK(refB.as<ObjAA>() == nullptr);
  CHECK(refB.as<ObjB>() != nullptr);
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
