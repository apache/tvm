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
 * \file tvm/target/parsers/aprofile.cc
 * \brief Target Parser for Arm(R) Cortex(R) A-Profile CPUs
 */

#include "aprofile.h"

#include <string>

#include "../../support/utils.h"

namespace tvm {
namespace target {
namespace parsers {
namespace aprofile {

double GetArchVersion(Array<String> mattr) {
  for (const String& attr : mattr) {
    std::string attr_string = attr;
    size_t attr_len = attr_string.size();
    if (attr_len >= 4 && attr_string.substr(0, 2) == "+v" && attr_string.back() == 'a') {
      std::string version_string = attr_string.substr(2, attr_string.size() - 2);
      return atof(version_string.data());
    }
  }
  return 0.0;
}

double GetArchVersion(Optional<Array<String>> attr) {
  if (!attr) {
    return false;
  }
  return GetArchVersion(attr.value());
}

static inline bool HasFlag(String attr, std::string flag) {
  std::string attr_str = attr;
  return attr_str.find(flag) != std::string::npos;
}

static inline bool HasFlag(Optional<String> attr, std::string flag) {
  if (!attr) {
    return false;
  }
  return HasFlag(attr.value(), flag);
}

static inline bool HasFlag(Optional<Array<String>> attr, std::string flag) {
  if (!attr) {
    return false;
  }
  Array<String> attr_array = attr.value();

  auto matching_attr = std::find_if(attr_array.begin(), attr_array.end(),
                                    [flag](String attr_str) { return HasFlag(attr_str, flag); });
  return matching_attr != attr_array.end();
}

static bool HasFlag(Optional<String> mcpu, Optional<Array<String>> mattr, std::string flag) {
  return HasFlag(mcpu, flag) || HasFlag(mattr, flag);
}

bool IsAArch32(Optional<String> mtriple, Optional<String> mcpu) {
  if (mtriple) {
    bool is_mprofile = mcpu && support::StartsWith(mcpu.value(), "cortex-m");
    return support::StartsWith(mtriple.value(), "arm") && !is_mprofile;
  }
  return false;
}

bool IsAArch64(Optional<String> mtriple) {
  if (mtriple) {
    return support::StartsWith(mtriple.value(), "aarch64");
  }
  return false;
}

bool IsArch(TargetJSON attrs) {
  Optional<String> mtriple = Downcast<Optional<String>>(attrs.Get("mtriple"));
  Optional<String> mcpu = Downcast<Optional<String>>(attrs.Get("mcpu"));

  return IsAArch32(mtriple, mcpu) || IsAArch64(mtriple);
}

static TargetFeatures GetFeatures(TargetJSON target) {
  Optional<String> mcpu = Downcast<Optional<String>>(target.Get("mcpu"));
  Optional<String> mtriple = Downcast<Optional<String>>(target.Get("mtriple"));
  Optional<Array<String>> mattr = Downcast<Optional<Array<String>>>(target.Get("mattr"));

  double arch_version = GetArchVersion(mattr);

  bool is_aarch64 = IsAArch64(mtriple);

  bool simd_flag = HasFlag(mcpu, mattr, "+neon") || HasFlag(mcpu, mattr, "+simd");
  bool has_asimd = is_aarch64 || simd_flag;

  bool i8mm_flag = HasFlag(mcpu, mattr, "+i8mm");
  bool i8mm_disable = HasFlag(mcpu, mattr, "+noi8mm");
  bool i8mm_default = arch_version >= 8.6;
  bool i8mm_support = arch_version >= 8.2 && arch_version <= 8.5;
  bool has_i8mm = (i8mm_default && !i8mm_disable) || (i8mm_support && i8mm_flag);

  bool dotprod_flag = HasFlag(mcpu, mattr, "+dotprod");
  bool dotprod_disable = HasFlag(mcpu, mattr, "+nodotprod");
  bool dotprod_default = arch_version >= 8.4;
  bool dotprod_support = arch_version >= 8.2 && arch_version <= 8.3;
  bool has_dotprod = (dotprod_default && !dotprod_disable) || (dotprod_support && dotprod_flag);

  return {
      {"is_aarch64", Bool(is_aarch64)},
      {"has_asimd", Bool(has_asimd)},
      {"has_dotprod", Bool(has_dotprod)},
      {"has_matmul_i8", Bool(has_i8mm)},
  };
}

static Array<String> MergeKeys(Optional<Array<String>> existing_keys) {
  const String kExtraKey = "arm_cpu";

  if (!existing_keys) {
    return {kExtraKey};
  }

  Array<String> keys = existing_keys.value();
  if (std::find(keys.begin(), keys.end(), kExtraKey) == keys.end()) {
    keys.push_back(kExtraKey);
  }
  return keys;
}

TargetJSON ParseTarget(TargetJSON target) {
  target.Set("features", GetFeatures(target));
  target.Set("keys", MergeKeys(Downcast<Optional<Array<String>>>(target.Get("keys"))));

  return target;
}

}  // namespace aprofile
}  // namespace parsers
}  // namespace target
}  // namespace tvm
