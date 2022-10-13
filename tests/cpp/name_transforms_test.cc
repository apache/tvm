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

#include "../src/relay/backend/name_transforms.h"

#include <gtest/gtest.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/name_transforms.h>

using namespace tvm::relay::backend;
using namespace tvm::runtime;

TEST(NameTransforms, ToCFunctionStyle) {
  ASSERT_EQ(ToCFunctionStyle("TVM_Woof"), "TVMWoof");
  ASSERT_EQ(ToCFunctionStyle("TVM_woof"), "TVMWoof");
  ASSERT_EQ(ToCFunctionStyle("TVM_woof_woof"), "TVMWoofWoof");
  ASSERT_EQ(ToCFunctionStyle("TVMGen_woof_woof"), "TVMGenWoofWoof");
  EXPECT_THROW(ToCVariableStyle("Cake_Bakery"), InternalError);  // Incorrect prefix
  EXPECT_THROW(ToCFunctionStyle(""), InternalError);
}

TEST(NameTransforms, ToCVariableStyle) {
  ASSERT_EQ(ToCVariableStyle("TVM_Woof"), "tvm_woof");
  ASSERT_EQ(ToCVariableStyle("TVM_woof"), "tvm_woof");
  ASSERT_EQ(ToCVariableStyle("TVM_woof_Woof"), "tvm_woof_woof");
  EXPECT_THROW(ToCVariableStyle("Cake_Bakery"), InternalError);  // Incorrect prefix
  EXPECT_THROW(ToCVariableStyle(""), InternalError);
}

TEST(NameTransforms, ToCConstantStyle) {
  ASSERT_EQ(ToCConstantStyle("TVM_Woof"), "TVM_WOOF");
  ASSERT_EQ(ToCConstantStyle("TVM_woof"), "TVM_WOOF");
  ASSERT_EQ(ToCConstantStyle("TVM_woof_Woof"), "TVM_WOOF_WOOF");
  EXPECT_THROW(ToCConstantStyle("Cake_Bakery"), InternalError);  // Incorrect prefix
  EXPECT_THROW(ToCConstantStyle(""), InternalError);
}

TEST(NameTransforms, PrefixName) {
  ASSERT_EQ(PrefixName({"Woof"}), "TVM_Woof");
  ASSERT_EQ(PrefixName({"woof"}), "TVM_woof");
  ASSERT_EQ(PrefixName({"woof", "moo"}), "TVM_woof_moo");
  EXPECT_THROW(PrefixName({}), InternalError);
  EXPECT_THROW(PrefixName({""}), InternalError);
}

TEST(NameTransforms, PrefixGeneratedName) {
  ASSERT_EQ(PrefixGeneratedName({"Woof"}), "TVMGen_Woof");
  ASSERT_EQ(PrefixGeneratedName({"woof"}), "TVMGen_woof");
  ASSERT_EQ(PrefixGeneratedName({"woof", "moo"}), "TVMGen_woof_moo");
  EXPECT_THROW(PrefixGeneratedName({}), InternalError);
  EXPECT_THROW(PrefixGeneratedName({""}), InternalError);
}

TEST(NameTransforms, CombineNames) {
  ASSERT_EQ(CombineNames({"woof"}), "woof");
  ASSERT_EQ(CombineNames({"Woof", "woof"}), "Woof_woof");
  ASSERT_EQ(CombineNames({"Woof", "woof", "woof"}), "Woof_woof_woof");
  ASSERT_EQ(CombineNames({"Woof", "moo", "t"}), "Woof_moo_t");

  EXPECT_THROW(CombineNames({}), InternalError);
  EXPECT_THROW(CombineNames({""}), InternalError);
  EXPECT_THROW(CombineNames({"Woof", ""}), InternalError);
  EXPECT_THROW(CombineNames({"", "Woof"}), InternalError);
}

TEST(NameTransforms, SanitizeName) {
  ASSERT_EQ(SanitizeName("+_+ "), "____");
  ASSERT_EQ(SanitizeName("input+"), "input_");
  ASSERT_EQ(SanitizeName("input-"), "input_");
  ASSERT_EQ(SanitizeName("input++"), "input__");
  ASSERT_EQ(SanitizeName("woof:1"), "woof_1");
  EXPECT_THROW(SanitizeName(""), InternalError);
}

TEST(NameTransforms, CombinedLogic) {
  ASSERT_EQ(ToCFunctionStyle(PrefixName({"Device", "target", "Invoke"})), "TVMDeviceTargetInvoke");
  ASSERT_EQ(ToCFunctionStyle(PrefixGeneratedName({"model", "Run"})), "TVMGenModelRun");
  ASSERT_EQ(ToCVariableStyle(PrefixName({"Device", "target", "t"})), "tvm_device_target_t");
  ASSERT_EQ(ToCVariableStyle(PrefixGeneratedName({"model", "Devices"})), "tvmgen_model_devices");
}
