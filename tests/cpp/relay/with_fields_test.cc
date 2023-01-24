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
 * specific language governing permissions and lixmitations
 * under the License.
 */

/*!
 * \brief Proof-of-concept unit tests for the family of WithFields helpers.
 * Only Call, GlobalVar and Constant are currently tested.
 */

#include <gtest/gtest.h>
#include <tvm/relay/adt.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/relay/parser.h>

namespace tvm {
namespace relay {
namespace {

IRModule TestIRModule() {
  return ParseModule("string",
                     R"(
    #[version = "0.0.5"]
    def @main(%data : Tensor[(1, 304, 128, 128), float32],
             %weight1 : Tensor[(304, 1, 3, 3), float32],
             %bias1 : Tensor[(304), float32],
             %weight2 : Tensor[(256, 304, 1, 1), float32],
             %bias2 : Tensor[(256), float32]) -> Tensor[(1, 256, 128, 128), float32] {
      %0 = nn.conv2d(%data, %weight1, padding=[1, 1, 1, 1], groups=304, channels=304, kernel_size=[3, 3]);
      %1 = nn.bias_add(%0, %bias1);
      %2 = nn.relu(%1);
      %3 = nn.conv2d(%2, %weight2, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
      %4 = nn.bias_add(%3, %bias2);
      nn.relu(%4)
    }
  )");
}

Function TestFunction() { return Downcast<Function>(TestIRModule()->Lookup("main")); }
Call TestCall() { return Downcast<Call>(TestFunction()->body); }
GlobalVar TestGlobalVar() { return TestIRModule()->GetGlobalVar("main"); }
VirtualDevice TestVirtualDevice() { return VirtualDevice::ForDevice({kDLCUDA, 3}); }
Span TestSpan() { return Span(SourceName::Get("foo"), 3, 4, 6, 42); }
Constant TestConstant() {
  return Constant(runtime::NDArray::Empty({}, DataType::Int(32), {kDLCPU, 0}));
}

//
// Call
//

TEST(WithFields, Call_Noop) {
  Call call = TestCall();
  Call result = WithFields(call);
  ASSERT_TRUE(result.same_as(call));
}

TEST(WithFields, Call_Op) {
  Call call = TestCall();
  Op new_op = Op::Get("tanh");
  Call result = WithFields(call, new_op);
  ASSERT_FALSE(result.same_as(call));
  ASSERT_FALSE(call->op.same_as(new_op));
  ASSERT_TRUE(result->op.same_as(new_op));
}

TEST(WithFields, Call_Args) {
  Call call = TestCall();
  Array<Expr> new_args = {Tuple(Array<Expr>())};
  Call result = WithFields(call, /*opt_op=*/{}, new_args);
  ASSERT_FALSE(result.same_as(call));
  ASSERT_FALSE(call->args.same_as(new_args));
  ASSERT_TRUE(result->args.same_as(new_args));
}

TEST(WithFields, Call_Attrs) {
  Call call = TestCall();
  Attrs new_attrs = DictAttrs(Map<String, ObjectRef>());
  Call result = WithFields(call, /*opt_op=*/{}, /*opt_args=*/{}, new_attrs);
  ASSERT_FALSE(result.same_as(call));
  ASSERT_FALSE(call->attrs.same_as(new_attrs));
  ASSERT_TRUE(result->attrs.same_as(new_attrs));
}

TEST(WithFields, Call_TypeArgs) {
  Call call = TestCall();
  Array<Type> new_type_args;
  Call result = WithFields(call, /*opt_op=*/{}, /*opt_args=*/{}, /*opt_attrs=*/{}, new_type_args);
  ASSERT_FALSE(result.same_as(call));
  ASSERT_FALSE(call->type_args.same_as(new_type_args));
  ASSERT_TRUE(result->type_args.same_as(new_type_args));
}

TEST(WithFields, Call_VirtualDevice) {
  Call call = TestCall();
  VirtualDevice new_virtual_device = TestVirtualDevice();
  Call result = WithFields(call, /*opt_op=*/{}, /*opt_args=*/{}, /*opt_attrs=*/{},
                           /*opt_type_args=*/{}, new_virtual_device);
  ASSERT_FALSE(result.same_as(call));
  ASSERT_FALSE(call->virtual_device().same_as(new_virtual_device));
  ASSERT_TRUE(result->virtual_device().same_as(new_virtual_device));
}

TEST(WithFields, Call_Span) {
  Call call = TestCall();
  Span new_span = TestSpan();
  Call result = WithFields(call, /*opt_op=*/{}, /*opt_args=*/{}, /*opt_attrs=*/{},
                           /*opt_type_args=*/{}, /*opt_virtual_device=*/{}, new_span);
  ASSERT_FALSE(result.same_as(call));
  ASSERT_FALSE(call->span.same_as(new_span));
  ASSERT_TRUE(result->span.same_as(new_span));
}

//
// GlobalVar
//

TEST(WithFields, GlobalVar_Noop) {
  GlobalVar gv = TestGlobalVar();
  GlobalVar result = WithFields(gv);
  ASSERT_TRUE(result.same_as(gv));
}

TEST(WithFields, GlobalVar_Name) {
  GlobalVar gv = TestGlobalVar();
  String new_name("foo");
  GlobalVar result = WithFields(gv, new_name);
  ASSERT_FALSE(result.same_as(gv));
  ASSERT_FALSE(gv->name_hint.same_as(new_name));
  ASSERT_TRUE(result->name_hint.same_as(new_name));
}

TEST(WithFields, GlobalVar_Type) {
  GlobalVar gv = TestGlobalVar();
  Type new_type = TupleType(Array<Type>());
  GlobalVar result = WithFields(gv, /*opt_name_hint=*/{}, new_type);
  ASSERT_FALSE(result.same_as(gv));
  ASSERT_FALSE(gv->checked_type().same_as(new_type));
  ASSERT_TRUE(result->checked_type().same_as(new_type));
}

TEST(WithFields, GlobalVar_VirtualDevice) {
  GlobalVar gv = TestGlobalVar();
  VirtualDevice new_virtual_device = TestVirtualDevice();
  GlobalVar result = WithFields(gv, /*opt_name_hint=*/{}, /*opt_type=*/{}, new_virtual_device);
  ASSERT_FALSE(result.same_as(gv));
  ASSERT_FALSE(gv->virtual_device().same_as(new_virtual_device));
  ASSERT_TRUE(result->virtual_device().same_as(new_virtual_device));
}

TEST(WithFields, GlobalVar_Span) {
  GlobalVar gv = TestGlobalVar();
  Span new_span = TestSpan();
  GlobalVar result =
      WithFields(gv, /*opt_name_hint=*/{}, /*opt_type=*/{}, /*opt_virtual_device=*/{}, new_span);
  ASSERT_FALSE(result.same_as(gv));
  ASSERT_FALSE(gv->span.same_as(new_span));
  ASSERT_TRUE(result->span.same_as(new_span));
}

//
// Constant
//

TEST(WithFields, Constant_Noop) {
  Constant constant = TestConstant();
  Constant result = WithFields(constant);
  ASSERT_TRUE(result.same_as(constant));
}

TEST(WithFields, Constant_Data) {
  Constant constant = TestConstant();
  runtime::NDArray new_data = runtime::NDArray::Empty({}, DataType::Float(32), {kDLCPU, 0});
  Constant result = WithFields(constant, new_data);
  ASSERT_FALSE(result.same_as(constant));
  ASSERT_FALSE(constant->data.same_as(new_data));
  ASSERT_TRUE(result->data.same_as(new_data));
}

TEST(WithFields, Constant_VirtualDevice) {
  Constant constant = TestConstant();
  VirtualDevice new_virtual_device = TestVirtualDevice();
  Constant result = WithFields(constant, /*opt_data=*/{}, new_virtual_device);
  ASSERT_FALSE(result.same_as(constant));
  ASSERT_FALSE(constant->virtual_device().same_as(new_virtual_device));
  ASSERT_TRUE(result->virtual_device().same_as(new_virtual_device));
}

TEST(WithFields, Constant_Span) {
  Constant constant = TestConstant();
  Span new_span = TestSpan();
  Constant result = WithFields(constant, /*opt_data=*/{}, /*opt_virtual_device=*/{}, new_span);
  ASSERT_FALSE(result.same_as(constant));
  ASSERT_FALSE(constant->span.same_as(new_span));
  ASSERT_TRUE(result->span.same_as(new_span));
}

}  // namespace
}  // namespace relay
}  // namespace tvm
