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
 * \file tvm/target/parsers/mprofile.cc
 * \brief Target Parser for Arm(R) Cortex(R) M-Profile CPUs
 */

#include "mprofile.h"

#include <string>

namespace tvm {
namespace target {
namespace parsers {
namespace mprofile {

const TargetFeatures kNoExt = {{"has_dsp", Bool(false)}, {"has_mve", Bool(false)}};
const TargetFeatures kHasDSP = {{"has_dsp", Bool(true)}, {"has_mve", Bool(false)}};
const TargetFeatures kHasMVE = {{"has_dsp", Bool(true)}, {"has_mve", Bool(true)}};

static const char* baseCPUs[] = {"cortex-m0", "cortex-m3"};
static const char* dspCPUs[] = {"cortex-m55", "cortex-m4",   "cortex-m7",
                                "cortex-m33", "cortex-m35p", "cortex-m85"};
static const char* mveCPUs[] = {"cortex-m55", "cortex-m85"};

template <typename Container>
static inline bool MatchesCpu(Optional<String> mcpu, const Container& cpus) {
  if (!mcpu) {
    return false;
  }
  std::string mcpu_string = mcpu.value();
  auto matches_cpu = [mcpu_string](const char* cpu) { return mcpu_string.find(cpu) == 0; };
  return std::find_if(std::begin(cpus), std::end(cpus), matches_cpu) != std::end(cpus);
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

bool IsArch(TargetJSON attrs) {
  Optional<String> mcpu = Downcast<Optional<String>>(attrs.Get("mcpu"));
  if (mcpu) {
    bool matches_base = MatchesCpu(mcpu, baseCPUs);
    bool matches_dsp = MatchesCpu(mcpu, dspCPUs);
    bool matches_mve = MatchesCpu(mcpu, mveCPUs);
    return matches_base || matches_mve || matches_dsp;
  }
  return false;
}

static TargetFeatures GetFeatures(TargetJSON target) {
  Optional<String> mcpu = Downcast<Optional<String>>(target.Get("mcpu"));
  Optional<Array<String>> mattr = Downcast<Optional<Array<String>>>(target.Get("mattr"));

  bool nomve = HasFlag(mcpu, "+nomve") || HasFlag(mattr, "+nomve");
  bool nodsp = HasFlag(mcpu, "+nodsp") || HasFlag(mattr, "+nodsp");

  bool has_mve = MatchesCpu(mcpu, mveCPUs);
  if (has_mve && !nomve && !nodsp) {
    return kHasMVE;
  }

  bool has_dsp = MatchesCpu(mcpu, dspCPUs);
  if (has_dsp && !nodsp) {
    return kHasDSP;
  }

  return kNoExt;
}

static Array<String> MergeKeys(Optional<Array<String>> existing_keys) {
  const Array<String> kExtraKeys = {"arm_cpu", "cpu"};

  if (!existing_keys) {
    return kExtraKeys;
  }

  Array<String> keys = existing_keys.value();
  for (String key : kExtraKeys) {
    if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
      keys.push_back(key);
    }
  }
  return keys;
}

TargetJSON ParseTarget(TargetJSON target) {
  target.Set("features", GetFeatures(target));
  target.Set("keys", MergeKeys(Downcast<Optional<Array<String>>>(target.Get("keys"))));

  return target;
}

}  // namespace mprofile
}  // namespace parsers
}  // namespace target
}  // namespace tvm
