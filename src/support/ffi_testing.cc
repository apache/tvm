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

 /*!
 *  FFI registration code used for frontend testing purposes.
 * \file ffi_testing.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/te/tensor.h>
#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>

namespace tvm {
// Attrs used to python API
struct TestAttrs : public AttrsNode<TestAttrs> {
  int axis;
  std::string name;
  Array<PrimExpr> padding;
  TypedEnvFunc<int(int)> func;

  TVM_DECLARE_ATTRS(TestAttrs, "attrs.TestAttrs") {
    TVM_ATTR_FIELD(axis)
        .set_default(10)
        .set_lower_bound(1)
        .set_upper_bound(10)
        .describe("axis field");
    TVM_ATTR_FIELD(name)
        .describe("name");
    TVM_ATTR_FIELD(padding)
        .describe("padding of input")
        .set_default(Array<PrimExpr>({0, 0}));
    TVM_ATTR_FIELD(func)
        .describe("some random env function")
        .set_default(TypedEnvFunc<int(int)>(nullptr));
  }
};

TVM_REGISTER_NODE_TYPE(TestAttrs);

TVM_REGISTER_GLOBAL("testing.nop")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
  });

TVM_REGISTER_GLOBAL("testing.echo")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
  *ret = args[0];
  });

TVM_REGISTER_GLOBAL("testing.test_wrap_callback")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    PackedFunc pf = args[0];
    *ret = runtime::TypedPackedFunc<void()>([pf](){
        pf();
      });
  });

TVM_REGISTER_GLOBAL("testing.test_raise_error_callback")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    std::string msg = args[0];
    *ret = runtime::TypedPackedFunc<void()>([msg](){
        LOG(FATAL) << msg;
      });
  });

TVM_REGISTER_GLOBAL("testing.test_check_eq_callback")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    std::string msg = args[0];
    *ret = runtime::TypedPackedFunc<void(int x, int y)>([msg](int x, int y){
        CHECK_EQ(x, y) << msg;
      });
  });

TVM_REGISTER_GLOBAL("testing.context_test")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    DLContext ctx = args[0];
    int dtype = args[1];
    int did = args[2];
    CHECK_EQ(static_cast<int>(ctx.device_type), dtype);
    CHECK_EQ(static_cast<int>(ctx.device_id), did);
    *ret = ctx;
  });


// in src/api_test.cc
void ErrorTest(int x, int y) {
  // raise ValueError
  CHECK_EQ(x, y) << "ValueError: expect x and y to be equal.";
  if (x == 1) {
    // raise InternalError.
    LOG(FATAL) << "InternalError: cannot reach here";
  }
}

TVM_REGISTER_GLOBAL("testing.ErrorTest")
.set_body_typed(ErrorTest);

// internal function used for debug and testing purposes
TVM_REGISTER_GLOBAL("testing.object_use_count")
.set_body([](TVMArgs args,  TVMRetValue *ret) {
    runtime::ObjectRef obj = args[0];
    // substract the current one because we always copy
    // and get another value.
    *ret = (obj.use_count() - 1);
  });
}  // namespace tvm
