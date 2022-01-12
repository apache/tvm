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

#include "vulkan_common.h"

#include <set>

namespace tvm {
namespace runtime {
namespace vulkan {

std::vector<const char*> FindEnabledExtensions(
    const std::vector<VkExtensionProperties>& ext_prop,
    const std::vector<const char*>& required_extensions,
    const std::vector<const char*>& optional_extensions) {
  std::set<std::string> available_extensions;
  for (const auto& prop : ext_prop) {
    if (prop.specVersion > 0) {
      available_extensions.insert(prop.extensionName);
    }
  }

  std::vector<const char*> enabled_extensions;
  for (const auto& ext : required_extensions) {
    ICHECK(available_extensions.count(ext))
        << "Required vulkan extension \"" << ext << "\" not supported by driver";
    enabled_extensions.push_back(ext);
  }

  for (const auto& ext : optional_extensions) {
    if (available_extensions.count(ext)) {
      enabled_extensions.push_back(ext);
    }
  }

  return enabled_extensions;
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
