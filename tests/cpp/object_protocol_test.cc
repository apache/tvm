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
#include <tvm/runtime/logging.h>
#include <tvm/runtime/object.h>

namespace tvm {
namespace test {

using namespace tvm::runtime;

class ObjBase : public Object {
 public:
  // dynamically allocate slow
  static constexpr const uint32_t _type_child_slots = 1;
  TVM_FFI_DECLARE_OBJECT_INFO("test.ObjBase", ObjBase, Object);
};

class ObjA : public ObjBase {
 public:
  static constexpr const uint32_t _type_child_slots = 0;
  TVM_FFI_DECLARE_OBJECT_INFO("test.ObjA", ObjA, ObjBase);
};

class ObjB : public ObjBase {
 public:
  static constexpr const uint32_t _type_child_slots = 0;
  TVM_FFI_DECLARE_OBJECT_INFO("test.ObjB", ObjB, ObjBase);
};

class ObjAA : public ObjA {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.ObjAA", ObjAA, ObjA);
};

}  // namespace test
}  // namespace tvm

TEST(ObjectHierachy, Basic) {
  using namespace tvm;
  using namespace tvm::runtime;
  using namespace tvm::test;
  using namespace tvm::ffi;

  ObjectRef refA(ffi::make_object<ObjA>());
  ICHECK_EQ(refA->type_index(), ObjA::RuntimeTypeIndex());
  ICHECK(refA.as<Object>() != nullptr);
  ICHECK(refA.as<ObjA>() != nullptr);
  ICHECK(refA.as<ObjBase>() != nullptr);
  ICHECK(refA.as<ObjB>() == nullptr);
  ICHECK(refA.as<ObjAA>() == nullptr);

  ObjectRef refAA(ffi::make_object<ObjAA>());
  ICHECK_EQ(refAA->type_index(), ObjAA::RuntimeTypeIndex());
  ICHECK(refAA.as<Object>() != nullptr);
  ICHECK(refAA.as<ObjBase>() != nullptr);
  ICHECK(refAA.as<ObjA>() != nullptr);
  ICHECK(refAA.as<ObjAA>() != nullptr);
  ICHECK(refAA.as<ObjB>() == nullptr);

  ObjectRef refB(ffi::make_object<ObjB>());
  ICHECK_EQ(refB->type_index(), ObjB::RuntimeTypeIndex());
  ICHECK(refB.as<Object>() != nullptr);
  ICHECK(refB.as<ObjBase>() != nullptr);
  ICHECK(refB.as<ObjA>() == nullptr);
  ICHECK(refB.as<ObjAA>() == nullptr);
  ICHECK(refB.as<ObjB>() != nullptr);
}
