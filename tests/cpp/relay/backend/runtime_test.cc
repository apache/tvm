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
#include <tvm/relay/runtime.h>

#include <cmath>
#include <string>
namespace tvm {
namespace relay {

TVM_REGISTER_RUNTIME("TestRuntime")
    .add_attr_option<Bool>("my_bool")
    .add_attr_option<Array<String>>("your_names")
    .add_attr_option<String>("another_option")
    .add_attr_option<Bool>("defaulty_the_default_option", Bool(false));

TEST(Runtime, Create) {
  Map<String, ObjectRef> attrs = {{"my_bool", Bool(true)}};
  Runtime my_runtime = Runtime::Create("TestRuntime", attrs);
  ASSERT_EQ(my_runtime->GetAttr<Bool>("my_bool"), true);
  ASSERT_EQ(my_runtime->GetAttr<Array<String>>("your_names").defined(), false);
  ASSERT_EQ(my_runtime->GetAttr<Bool>("defaulty_the_default_option"), false);
}

TEST(Runtime, UnknownAttr) {
  Map<String, ObjectRef> attrs = {{"woofles", Bool(true)}};
  ASSERT_THROW(Runtime::Create("TestRuntime", attrs), Error);
}

TEST(Runtime, IncorrectAttrType) {
  Map<String, ObjectRef> attrs = {{"my_bool", String("snuck_in")}};
  ASSERT_THROW(Runtime::Create("TestRuntime", attrs), Error);
}

TEST(Runtime, UnregisteredName) {
  Map<String, ObjectRef> attrs = {};
  ASSERT_THROW(Runtime::Create("NeverNameAnRuntimeThis", attrs), Error);
}

TEST(RuntimeRegistry, ListRuntimes) {
  Array<String> names = Runtime::ListRuntimes();
  ICHECK_EQ(names.empty(), false);
  ICHECK_EQ(std::count(std::begin(names), std::end(names), "TestRuntime"), 1);
}

TEST(RuntimeRegistry, ListRuntimeOptions) {
  Map<String, String> attrs = Runtime::ListRuntimeOptions("TestRuntime");

  ICHECK_EQ(attrs.empty(), false);
  ICHECK_EQ(attrs["my_bool"], "IntImm");
  ICHECK_EQ(attrs["your_names"], "Array");
  ICHECK_EQ(attrs["another_option"], "runtime.String");
}

TEST(RuntimeRegistry, ListRuntimeOptionsNoRuntime) {
  ASSERT_THROW(Runtime::ListRuntimeOptions("NeverNameAnRuntimeThis"), Error);
}

}  // namespace relay
}  // namespace tvm
