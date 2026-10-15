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
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>

using namespace tvm;

TVM_FFI_STATIC_INIT_BLOCK() {
  TargetKindDef("TestTargetKind").set_default_device_type(kDLCPU).def_option<bool>("my_bool");
}

ffi::Map<ffi::String, ffi::Any> TestTargetParser(ffi::Map<ffi::String, ffi::Any> target) {
  ffi::String mcpu = target.at("mcpu").as_or_throw<ffi::String>();
  target.Set("mcpu", ffi::String("super_") + mcpu);
  target.Set("keys", ffi::Array<ffi::String>({"super"}));
  target.Set("feature.test", true);
  return target;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  TargetKindDef("TestTargetParser")
      .set_default_device_type(kDLCPU)
      .def_option<ffi::String>("mattr")
      .def_option<ffi::String>("mcpu")
      .set_default_keys({"cpu"})
      .set_target_canonicalizer(TestTargetParser);
}

TEST(TargetCreationFail, TargetKindNotFound) {
  ffi::Map<ffi::String, ffi::Any> config = {};
  ASSERT_THROW({ Target{config}; }, ffi::Error);
}

TEST(TargetCreation, TargetParser) {
  Target test_target(ffi::Map<ffi::String, ffi::Any>{
      {"kind", ffi::String("TestTargetParser")},
      {"mcpu", ffi::String("woof")},
      {"mattr", ffi::String("cake")},
  });
  ASSERT_EQ(test_target->GetAttr<ffi::String>("mcpu").value(), "super_woof");
  ASSERT_EQ(test_target->keys.size(), 1);
  ASSERT_EQ(test_target->keys[0], "super");
  ASSERT_EQ(test_target->GetAttr<ffi::String>("mattr").value(), "cake");
  ASSERT_EQ(test_target->GetAttr<bool>("feature.test").value(), true);
  ASSERT_EQ(test_target->GetAttr<bool>("feature.nonexistent"), std::nullopt);

  Target test_target_no_parser("TestTargetKind");
  ASSERT_EQ(test_target_no_parser->GetAttr<bool>("feature.test"), std::nullopt);
  ASSERT_EQ(test_target_no_parser->GetAttr<bool>("feature.test", true).value(), true);
}

TEST(TargetCreation, RoundTripCanonicalizerFeatures) {
  // Construct a target whose canonicalizer sets feature.test and transforms mcpu
  Target original(ffi::Map<ffi::String, ffi::Any>{
      {"kind", ffi::String("TestTargetParser")},
      {"mcpu", ffi::String("woof")},
  });
  ASSERT_EQ(original->GetAttr<ffi::String>("mcpu").value(), "super_woof");
  ASSERT_EQ(original->GetAttr<bool>("feature.test").value(), true);

  // Export to config and reconstruct
  ffi::Map<ffi::String, ffi::Any> exported = original->ToConfig();
  Target reconstructed(exported);

  // Canonicalized attrs must survive the round-trip
  // Note: mcpu gets canonicalized again (super_super_woof) because the canonicalizer runs
  ASSERT_TRUE(reconstructed->GetAttr<ffi::String>("mcpu").has_value());
  ASSERT_EQ(reconstructed->GetAttr<bool>("feature.test").value(), true);
  ASSERT_EQ(reconstructed->keys.size(), 1);
  ASSERT_EQ(reconstructed->keys[0], "super");
}

TEST(TargetCreation, RoundTripCanonicalizerFeaturesNestedHost) {
  // Construct a host target whose canonicalizer sets feature.test
  Target host(ffi::Map<ffi::String, ffi::Any>{
      {"kind", ffi::String("TestTargetParser")},
      {"mcpu", ffi::String("woof")},
  });
  ASSERT_EQ(host->GetAttr<bool>("feature.test").value(), true);

  // Attach it as host to another target
  Target outer(ffi::Map<ffi::String, ffi::Any>{
      {"kind", ffi::String("TestTargetKind")},
      {"my_bool", true},
  });
  Target combined(outer, host);

  // Export the outer target (includes nested host) and reconstruct
  ffi::Map<ffi::String, ffi::Any> exported = combined->ToConfig();
  Target reconstructed(exported);

  // The nested host must reconstruct successfully with feature.* preserved
  ffi::Optional<Target> reconstructed_host = reconstructed->GetHost();
  ASSERT_TRUE(reconstructed_host.has_value());
  ASSERT_EQ(reconstructed_host.value()->GetAttr<bool>("feature.test").value(), true);
  ASSERT_TRUE(reconstructed_host.value()->GetAttr<ffi::String>("mcpu").has_value());
}

TEST(TargetCreationFail, UnknownNonFeatureKeyStillFails) {
  // Verify that unknown non-feature.* keys still fail schema validation
  ffi::Map<ffi::String, ffi::Any> config = {
      {"kind", ffi::String("TestTargetParser")},
      {"mcpu", ffi::String("woof")},
      {"unknown_key", ffi::String("bad")},
  };
  ASSERT_THROW({ Target{config}; }, tvm::ffi::Error);
}

TEST(TargetCreation, DeduplicateKeys) {
  ffi::Map<ffi::String, ffi::Any> config = {
      {"kind", ffi::String("llvm")},
      {"keys", ffi::Array<ffi::String>{"cpu", "arm_cpu"}},
      {"device", ffi::String("arm_cpu")},
  };
  Target target = Target(config);
  TVM_FFI_ICHECK_EQ(target->kind, TargetKind::Get("llvm").value());
  TVM_FFI_ICHECK_EQ(target->tag, "");
  TVM_FFI_ICHECK_EQ(target->keys.size(), 2U);
  TVM_FFI_ICHECK_EQ(target->keys[0], "cpu");
  TVM_FFI_ICHECK_EQ(target->keys[1], "arm_cpu");
  TVM_FFI_ICHECK_EQ(target->attrs.count("keys"), 0U);
  TVM_FFI_ICHECK_EQ(target->GetAttr<ffi::String>("device"), "arm_cpu");
}
