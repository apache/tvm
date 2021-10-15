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
#include <tvm/ir/expr.h>
#include <tvm/target/target.h>

#include <cmath>
#include <string>

using namespace tvm;

TVM_REGISTER_TARGET_KIND("TestTargetKind", kDLCPU)
    .set_attr<std::string>("Attr1", "Value1")
    .add_attr_option<Bool>("my_bool")
    .add_attr_option<Array<String>>("your_names")
    .add_attr_option<Map<String, Integer>>("her_maps");

TEST(TargetKind, GetAttrMap) {
  auto map = tvm::TargetKind::GetAttrMap<std::string>("Attr1");
  auto target_kind = tvm::TargetKind::Get("TestTargetKind").value();
  std::string result = map[target_kind];
  ICHECK_EQ(result, "Value1");
}

TEST(TargetCreation, NestedConfig) {
  Map<String, ObjectRef> config = {
      {"my_bool", Bool(true)},
      {"your_names", Array<String>{"junru", "jian"}},
      {"kind", String("TestTargetKind")},
      {
          "her_maps",
          Map<String, Integer>{
              {"a", 1},
              {"b", 2},
          },
      },
  };
  Target target = Target(config);
  ICHECK_EQ(target->kind, TargetKind::Get("TestTargetKind").value());
  ICHECK_EQ(target->tag, "");
  ICHECK(target->keys.empty());
  Bool my_bool = target->GetAttr<Bool>("my_bool").value();
  ICHECK_EQ(my_bool.operator bool(), true);
  Array<String> your_names = target->GetAttr<Array<String>>("your_names").value();
  ICHECK_EQ(your_names.size(), 2U);
  ICHECK_EQ(your_names[0], "junru");
  ICHECK_EQ(your_names[1], "jian");
  Map<String, Integer> her_maps = target->GetAttr<Map<String, Integer>>("her_maps").value();
  ICHECK_EQ(her_maps.size(), 2U);
  ICHECK_EQ(her_maps["a"], 1);
  ICHECK_EQ(her_maps["b"], 2);
}

TEST(TargetCreationFail, UnrecognizedConfigOption) {
  Map<String, ObjectRef> config = {
      {"my_bool", Bool(true)},
      {"your_names", Array<String>{"junru", "jian"}},
      {"kind", String("TestTargetKind")},
      {"bad", ObjectRef(nullptr)},
      {
          "her_maps",
          Map<String, Integer>{
              {"a", 1},
              {"b", 2},
          },
      },
  };
  bool failed = false;
  try {
    Target tgt(config);
  } catch (...) {
    failed = true;
  }
  ASSERT_EQ(failed, true);
}

TEST(TargetCreationFail, TypeMismatch) {
  Map<String, ObjectRef> config = {
      {"my_bool", String("true")},
      {"your_names", Array<String>{"junru", "jian"}},
      {"kind", String("TestTargetKind")},
      {
          "her_maps",
          Map<String, Integer>{
              {"a", 1},
              {"b", 2},
          },
      },
  };
  bool failed = false;
  try {
    Target tgt(config);
  } catch (...) {
    failed = true;
  }
  ASSERT_EQ(failed, true);
}

TEST(TargetCreationFail, TargetKindNotFound) {
  Map<String, ObjectRef> config = {
      {"my_bool", Bool("true")},
      {"your_names", Array<String>{"junru", "jian"}},
      {
          "her_maps",
          Map<String, Integer>{
              {"a", 1},
              {"b", 2},
          },
      },
  };
  bool failed = false;
  try {
    Target tgt(config);
  } catch (...) {
    failed = true;
  }
  ASSERT_EQ(failed, true);
}

TEST(TargetCreation, DeduplicateKeys) {
  Map<String, ObjectRef> config = {
      {"kind", String("llvm")},
      {"keys", Array<String>{"cpu", "arm_cpu"}},
      {"device", String("arm_cpu")},
  };
  Target target = Target(config);
  ICHECK_EQ(target->kind, TargetKind::Get("llvm").value());
  ICHECK_EQ(target->tag, "");
  ICHECK_EQ(target->keys.size(), 2U);
  ICHECK_EQ(target->keys[0], "cpu");
  ICHECK_EQ(target->keys[1], "arm_cpu");
  ICHECK_EQ(target->attrs.size(), 2U);
  ICHECK_EQ(target->GetAttr<String>("device"), "arm_cpu");
  ICHECK_EQ(target->GetAttr<Bool>("link-params"), false);
}

TEST(TargetKindRegistry, ListTargetKinds) {
  Array<String> names = TargetKindRegEntry::ListTargetKinds();
  ICHECK_EQ(names.empty(), false);
  ICHECK_EQ(std::count(std::begin(names), std::end(names), "llvm"), 1);
}

TEST(TargetKindRegistry, ListTargetOptions) {
  TargetKind llvm = TargetKind::Get("llvm").value();
  Map<String, String> attrs = TargetKindRegEntry::ListTargetKindOptions(llvm);
  ICHECK_EQ(attrs.empty(), false);

  ICHECK_EQ(attrs["mattr"], "Array");
  ICHECK_EQ(attrs["mcpu"], "runtime.String");
  ICHECK_EQ(attrs["system-lib"], "IntImm");
}
