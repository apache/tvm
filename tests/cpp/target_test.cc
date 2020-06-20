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
#include <tvm/target/target_id.h>

#include <cmath>
#include <string>

using namespace tvm;

TVM_REGISTER_TARGET_ID("TestTargetId")
    .set_attr<std::string>("Attr1", "Value1")
    .add_attr_option<Bool>("my_bool")
    .add_attr_option<Array<String>>("your_names")
    .add_attr_option<Map<String, Integer>>("her_maps");

TEST(TargetId, GetAttrMap) {
  auto map = tvm::TargetId::GetAttrMap<std::string>("Attr1");
  auto target_id = tvm::TargetId::Get("TestTargetId");
  std::string result = map[target_id];
  CHECK_EQ(result, "Value1");
}

TEST(TargetId, SchemaValidation) {
  tvm::Map<String, ObjectRef> target;
  {
    tvm::Array<String> your_names{"junru", "jian"};
    tvm::Map<String, Integer> her_maps{
        {"a", 1},
        {"b", 2},
    };
    target.Set("my_bool", Bool(true));
    target.Set("your_names", your_names);
    target.Set("her_maps", her_maps);
    target.Set("id", String("TestTargetId"));
  }
  TargetValidateSchema(target);
  tvm::Map<String, ObjectRef> target_host(target.begin(), target.end());
  target.Set("target_host", target_host);
  TargetValidateSchema(target);
}

TEST(TargetId, SchemaValidationFail) {
  tvm::Map<String, ObjectRef> target;
  {
    tvm::Array<String> your_names{"junru", "jian"};
    tvm::Map<String, Integer> her_maps{
        {"a", 1},
        {"b", 2},
    };
    target.Set("my_bool", Bool(true));
    target.Set("your_names", your_names);
    target.Set("her_maps", her_maps);
    target.Set("ok", ObjectRef(nullptr));
    target.Set("id", String("TestTargetId"));
  }
  bool failed = false;
  try {
    TargetValidateSchema(target);
  } catch (...) {
    failed = true;
  }
  ASSERT_EQ(failed, true);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
