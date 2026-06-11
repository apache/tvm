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
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/config_schema.h>

using namespace tvm;
using namespace tvm::ir;
namespace refl = tvm::ffi::reflection;

// Basic int/bool/string option resolution
TEST(ConfigSchema, BasicTypes) {
  ConfigSchema schema;
  schema.def_option<int64_t>("num_threads");
  schema.def_option<bool>("verbose");
  schema.def_option<ffi::String>("name");

  ffi::Map<ffi::String, ffi::Any> config = {
      {"num_threads", int64_t(8)},
      {"verbose", true},
      {"name", ffi::String("test")},
  };

  auto resolved = schema.Resolve(config);
  ASSERT_EQ(resolved.at("num_threads").cast<int64_t>(), 8);
  ASSERT_EQ(resolved.at("verbose").cast<bool>(), true);
  ASSERT_EQ(resolved.at("name").cast<ffi::String>(), "test");
}

// Type coercion: correct type passes
TEST(ConfigSchema, TypeCoercionPasses) {
  ConfigSchema schema;
  schema.def_option<int64_t>("count");

  ffi::Map<ffi::String, ffi::Any> config = {{"count", int64_t(42)}};
  auto resolved = schema.Resolve(config);
  ASSERT_EQ(resolved.at("count").cast<int64_t>(), 42);
}

// Type mismatch rejection (TypeError)
TEST(ConfigSchema, TypeMismatchRejection) {
  ConfigSchema schema;
  schema.def_option<int64_t>("count");

  // Pass a string where int64_t is expected
  ffi::Map<ffi::String, ffi::Any> config = {{"count", ffi::String("not_a_number")}};
  try {
    schema.Resolve(config);
    FAIL() << "Expected TypeError";
  } catch (const ffi::Error& e) {
    EXPECT_EQ(e.kind(), "TypeError");
    EXPECT_NE(std::string(e.message()).find("count"), std::string::npos);
  }
}

TEST(ConfigSchema, TypeMismatchBoolForInt) {
  ConfigSchema schema;
  schema.def_option<bool>("flag");

  // Pass a string where bool is expected
  ffi::Map<ffi::String, ffi::Any> config = {{"flag", ffi::String("true")}};
  try {
    schema.Resolve(config);
    FAIL() << "Expected TypeError";
  } catch (const ffi::Error& e) {
    EXPECT_EQ(e.kind(), "TypeError");
    EXPECT_NE(std::string(e.message()).find("flag"), std::string::npos);
  }
}

// Default value materialization via refl::DefaultValue
TEST(ConfigSchema, DefaultValueMaterialization) {
  ConfigSchema schema;
  schema.def_option<int64_t>("max_threads", refl::DefaultValue(int64_t(256)));
  schema.def_option<bool>("debug", refl::DefaultValue(false));

  // Empty config -> defaults are applied
  ffi::Map<ffi::String, ffi::Any> config = {};
  auto resolved = schema.Resolve(config);
  ASSERT_EQ(resolved.at("max_threads").cast<int64_t>(), 256);
  ASSERT_EQ(resolved.at("debug").cast<bool>(), false);
}

// Default value materialization via refl::DefaultValue
TEST(ConfigSchema, DefaultValueViaTrait) {
  ConfigSchema schema;
  schema.def_option<int64_t>("max_threads", refl::DefaultValue(int64_t(128)));

  ffi::Map<ffi::String, ffi::Any> config = {};
  auto resolved = schema.Resolve(config);
  ASSERT_EQ(resolved.at("max_threads").cast<int64_t>(), 128);
}

// Explicit value overrides default
TEST(ConfigSchema, ExplicitOverridesDefault) {
  ConfigSchema schema;
  schema.def_option<int64_t>("max_threads", refl::DefaultValue(int64_t(256)));

  ffi::Map<ffi::String, ffi::Any> config = {{"max_threads", int64_t(512)}};
  auto resolved = schema.Resolve(config);
  ASSERT_EQ(resolved.at("max_threads").cast<int64_t>(), 512);
}

// Missing non-required option stays absent
TEST(ConfigSchema, MissingNonRequired) {
  ConfigSchema schema;
  schema.def_option<ffi::String>("optional_name");  // no default

  ffi::Map<ffi::String, ffi::Any> config = {};
  auto resolved = schema.Resolve(config);
  ASSERT_EQ(resolved.find("optional_name"), resolved.end());
}

// Unknown key rejection (ValueError)
TEST(ConfigSchema, UnknownKeyRejection) {
  ConfigSchema schema;
  schema.def_option<int64_t>("known_key");

  ffi::Map<ffi::String, ffi::Any> config = {
      {"known_key", int64_t(1)},
      {"unknown_key", ffi::String("surprise")},
  };
  try {
    schema.Resolve(config);
    FAIL() << "Expected ValueError";
  } catch (const ffi::Error& e) {
    EXPECT_EQ(e.kind(), "ValueError");
    EXPECT_NE(std::string(e.message()).find("unknown_key"), std::string::npos);
  }
}

// Unknown key allowed when error_on_unknown is false
TEST(ConfigSchema, UnknownKeyAllowed) {
  ConfigSchema schema;
  schema.def_option<int64_t>("known_key");
  schema.set_error_on_unknown(false);

  ffi::Map<ffi::String, ffi::Any> config = {
      {"known_key", int64_t(1)},
      {"unknown_key", ffi::String("allowed")},
  };
  auto resolved = schema.Resolve(config);
  // Unknown keys are silently dropped (not validated, not forwarded)
  ASSERT_EQ(resolved.at("known_key").cast<int64_t>(), 1);
}

// Canonicalizer runs as final step
TEST(ConfigSchema, CanonicalizerRunsLast) {
  ConfigSchema schema;
  schema.def_option<ffi::String>("mcpu");

  auto canonicalizer = [](ffi::Map<ffi::String, ffi::Any> config) {
    ffi::String mcpu = config.at("mcpu").cast<ffi::String>();
    config.Set("mcpu", ffi::String("canonical_") + mcpu);
    // Canonicalizer can add new keys
    config.Set("feature.is_fast", true);
    return config;
  };
  schema.set_canonicalizer(canonicalizer);

  ffi::Map<ffi::String, ffi::Any> config = {{"mcpu", ffi::String("arm")}};
  auto resolved = schema.Resolve(config);
  ASSERT_EQ(resolved.at("mcpu").cast<ffi::String>(), "canonical_arm");
  ASSERT_EQ(resolved.at("feature.is_fast").cast<bool>(), true);
}

// Option declaration order is preserved in output
TEST(ConfigSchema, DeclarationOrderPreserved) {
  ConfigSchema schema;
  schema.def_option<ffi::String>("zebra");
  schema.def_option<ffi::String>("alpha");
  schema.def_option<ffi::String>("middle");

  auto& options = schema.ListOptions();
  ASSERT_EQ(options.size(), 3);
  ASSERT_EQ(std::string(options[0].key), "zebra");
  ASSERT_EQ(std::string(options[1].key), "alpha");
  ASSERT_EQ(std::string(options[2].key), "middle");
}

// Doc string trait accepted without affecting behavior
TEST(ConfigSchema, DocStringTrait) {
  ConfigSchema schema;
  schema.def_option<int64_t>("count", "Number of items to process");

  ffi::Map<ffi::String, ffi::Any> config = {{"count", int64_t(5)}};
  auto resolved = schema.Resolve(config);
  ASSERT_EQ(resolved.at("count").cast<int64_t>(), 5);
}

// Array option type
TEST(ConfigSchema, ArrayOptionType) {
  ConfigSchema schema;
  schema.def_option<ffi::Array<ffi::String>>("keys");

  ffi::Map<ffi::String, ffi::Any> config = {
      {"keys", ffi::Array<ffi::String>{"cpu", "gpu"}},
  };
  auto resolved = schema.Resolve(config);
  auto keys = resolved.at("keys").cast<ffi::Array<ffi::String>>();
  ASSERT_EQ(keys.size(), 2);
  ASSERT_EQ(keys[0], "cpu");
  ASSERT_EQ(keys[1], "gpu");
}

// Map option type
TEST(ConfigSchema, MapOptionType) {
  ConfigSchema schema;
  schema.def_option<ffi::Map<ffi::String, int64_t>>("params");

  ffi::Map<ffi::String, int64_t> params = {{"a", 1}, {"b", 2}};
  ffi::Map<ffi::String, ffi::Any> config = {{"params", params}};
  auto resolved = schema.Resolve(config);
  auto result = resolved.at("params").cast<ffi::Map<ffi::String, int64_t>>();
  ASSERT_EQ(result.size(), 2);
  ASSERT_EQ(result["a"], 1);
  ASSERT_EQ(result["b"], 2);
}

// Duplicate option key rejected
TEST(ConfigSchema, DuplicateKeyRejected) {
  ConfigSchema schema;
  schema.def_option<int64_t>("key1");
  try {
    schema.def_option<int64_t>("key1");
    FAIL() << "Expected ValueError";
  } catch (const ffi::Error& e) {
    EXPECT_EQ(e.kind(), "ValueError");
    EXPECT_NE(std::string(e.message()).find("key1"), std::string::npos);
  }
}

// ListOptions returns option entries in declaration order
TEST(ConfigSchema, ListOptionsNames) {
  ConfigSchema schema;
  schema.def_option<int64_t>("no_default");
  schema.def_option<bool>("with_default", refl::DefaultValue(true));

  auto& options = schema.ListOptions();
  ASSERT_EQ(options.size(), 2);
  ASSERT_EQ(std::string(options[0].key), "no_default");
  ASSERT_EQ(std::string(options[1].key), "with_default");
  ASSERT_EQ(std::string(options[0].type_str), "int");
  ASSERT_EQ(std::string(options[1].type_str), "bool");
  // Check default_value
  ASSERT_FALSE(options[0].has_default);
  ASSERT_TRUE(options[1].has_default);
  ASSERT_EQ(options[1].default_value.cast<bool>(), true);
}

// OptionEntry stores correct type strings
TEST(ConfigSchema, OptionEntryTypeStr) {
  ConfigSchema schema;
  schema.def_option<int64_t>("count");
  schema.def_option<bool>("flag");

  auto& options = schema.ListOptions();
  ASSERT_EQ(std::string(options[0].type_str), "int");
  ASSERT_EQ(std::string(options[1].type_str), "bool");
}

// HasOption query
TEST(ConfigSchema, HasOption) {
  ConfigSchema schema;
  schema.def_option<int64_t>("exists");
  ASSERT_TRUE(schema.HasOption("exists"));
  ASSERT_FALSE(schema.HasOption("does_not_exist"));
}

// Empty config with no options
TEST(ConfigSchema, EmptySchemaEmptyConfig) {
  ConfigSchema schema;
  ffi::Map<ffi::String, ffi::Any> config = {};
  auto resolved = schema.Resolve(config);
  ASSERT_EQ(resolved.size(), 0);
}

// Canonicalizer with defaults
TEST(ConfigSchema, CanonicalizerWithDefaults) {
  ConfigSchema schema;
  schema.def_option<int64_t>("threads", refl::DefaultValue(int64_t(4)));

  auto canonicalizer = [](ffi::Map<ffi::String, ffi::Any> config) {
    int64_t threads = config.at("threads").cast<int64_t>();
    config.Set("threads", threads * 2);
    return config;
  };
  schema.set_canonicalizer(canonicalizer);

  // Empty config: default 4 applied, then canonicalizer doubles it
  ffi::Map<ffi::String, ffi::Any> config = {};
  auto resolved = schema.Resolve(config);
  ASSERT_EQ(resolved.at("threads").cast<int64_t>(), 8);
}

// Combined doc + default traits
TEST(ConfigSchema, DocAndDefaultTraits) {
  ConfigSchema schema;
  schema.def_option<int64_t>("num_cores", "CPU core count", refl::DefaultValue(int64_t(4)));

  ffi::Map<ffi::String, ffi::Any> config = {};
  auto resolved = schema.Resolve(config);
  ASSERT_EQ(resolved.at("num_cores").cast<int64_t>(), 4);
}
