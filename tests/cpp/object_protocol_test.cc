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

namespace tvm {
namespace test {

using namespace tvm::runtime;

class BaseObj : public ffi::Object {
 public:
  // dynamically allocate slow
  static constexpr const uint32_t _type_child_slots = 1;
  TVM_FFI_DECLARE_OBJECT_INFO("test.ObjBase", BaseObj, ffi::Object);
};

class AObj : public BaseObj {
 public:
  static constexpr const uint32_t _type_child_slots = 0;
  TVM_FFI_DECLARE_OBJECT_INFO("test.ObjA", AObj, BaseObj);
};

class BObj : public BaseObj {
 public:
  static constexpr const uint32_t _type_child_slots = 0;
  TVM_FFI_DECLARE_OBJECT_INFO("test.ObjB", BObj, BaseObj);
};

class AAObj : public AObj {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.ObjAA", AAObj, AObj);
};

}  // namespace test
}  // namespace tvm

TEST(ObjectHierachy, Basic) {
  using namespace tvm;
  using namespace tvm::runtime;
  using namespace tvm::test;
  using namespace tvm::ffi;

  ffi::ObjectRef refA(ffi::make_object<AObj>());
  TVM_FFI_ICHECK_EQ(refA->type_index(), AObj::RuntimeTypeIndex());
  TVM_FFI_ICHECK(refA.as<ffi::Object>() != nullptr);
  TVM_FFI_ICHECK(refA.as<AObj>() != nullptr);
  TVM_FFI_ICHECK(refA.as<BaseObj>() != nullptr);
  TVM_FFI_ICHECK(refA.as<BObj>() == nullptr);
  TVM_FFI_ICHECK(refA.as<AAObj>() == nullptr);

  ffi::ObjectRef refAA(ffi::make_object<AAObj>());
  TVM_FFI_ICHECK_EQ(refAA->type_index(), AAObj::RuntimeTypeIndex());
  TVM_FFI_ICHECK(refAA.as<ffi::Object>() != nullptr);
  TVM_FFI_ICHECK(refAA.as<BaseObj>() != nullptr);
  TVM_FFI_ICHECK(refAA.as<AObj>() != nullptr);
  TVM_FFI_ICHECK(refAA.as<AAObj>() != nullptr);
  TVM_FFI_ICHECK(refAA.as<BObj>() == nullptr);

  ffi::ObjectRef refB(ffi::make_object<BObj>());
  TVM_FFI_ICHECK_EQ(refB->type_index(), BObj::RuntimeTypeIndex());
  TVM_FFI_ICHECK(refB.as<ffi::Object>() != nullptr);
  TVM_FFI_ICHECK(refB.as<BaseObj>() != nullptr);
  TVM_FFI_ICHECK(refB.as<AObj>() == nullptr);
  TVM_FFI_ICHECK(refB.as<AAObj>() == nullptr);
  TVM_FFI_ICHECK(refB.as<BObj>() != nullptr);
}
