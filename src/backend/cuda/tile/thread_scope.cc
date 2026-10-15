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
#include "thread_scope.h"

#include <tvm/runtime/logging.h>
namespace tvm::backend::cuda {
std::string ScopeKindToString(ScopeKind kind) {
  switch (kind) {
    case ScopeKind::kCluster:
      return "cluster";
    case ScopeKind::kCta:
      return "cta";
    case ScopeKind::kWarpgroup:
      return "warpgroup";
    case ScopeKind::kWarp:
      return "warp";
    case ScopeKind::kThread:
      return "thread";
  }
  LOG(FATAL) << "Internal Error: unknown ScopeKind " << static_cast<int>(kind);
}

ScopeKind StringToScopeKind(const ffi::String& name) {
  if (name == "cluster") return ScopeKind::kCluster;
  if (name == "cta") return ScopeKind::kCta;
  if (name == "warpgroup") return ScopeKind::kWarpgroup;
  if (name == "warp") return ScopeKind::kWarp;
  if (name == "thread") return ScopeKind::kThread;
  LOG(FATAL) << "Unknown scope kind name: " << name;
}

std::pair<ffi::String, ffi::String> ScopeBindingToStringPair(ScopeBinding binding) {
  switch (binding) {
    case ScopeBinding::kKernelCluster:
      return {"kernel", "cluster"};
    case ScopeBinding::kKernelCta:
      return {"kernel", "cta"};
    case ScopeBinding::kClusterCta:
      return {"cluster", "cta"};
    case ScopeBinding::kCtaWarpgroup:
      return {"cta", "warpgroup"};
    case ScopeBinding::kCtaWarp:
      return {"cta", "warp"};
    case ScopeBinding::kWarpgroupWarp:
      return {"warpgroup", "warp"};
    case ScopeBinding::kWarpThread:
      return {"warp", "thread"};
    case ScopeBinding::kCtaThread:
      return {"cta", "thread"};
    case ScopeBinding::kWarpgroupThread:
      return {"warpgroup", "thread"};
    case ScopeBinding::kClusterCtaPair:
      return {"cluster", "cta_pair"};
  }
  LOG(FATAL) << "Internal Error: unknown ScopeBinding " << static_cast<int>(binding);
}

ScopeBinding StringPairToScopeBinding(const ffi::String& parent, const ffi::String& cur) {
  if (parent == "kernel" && cur == "cluster") return ScopeBinding::kKernelCluster;
  if (parent == "kernel" && cur == "cta") return ScopeBinding::kKernelCta;
  if (parent == "cluster" && cur == "cta") return ScopeBinding::kClusterCta;
  if (parent == "cta" && cur == "warpgroup") return ScopeBinding::kCtaWarpgroup;
  if (parent == "cta" && cur == "warp") return ScopeBinding::kCtaWarp;
  if (parent == "warpgroup" && cur == "warp") return ScopeBinding::kWarpgroupWarp;
  if (parent == "warp" && cur == "thread") return ScopeBinding::kWarpThread;
  if (parent == "cta" && cur == "thread") return ScopeBinding::kCtaThread;
  if (parent == "warpgroup" && cur == "thread") return ScopeBinding::kWarpgroupThread;
  if (parent == "cluster" && cur == "cta_pair") return ScopeBinding::kClusterCtaPair;
  LOG(FATAL) << "Unknown scope binding: parent=" << parent << " cur=" << cur;
}

}  // namespace tvm::backend::cuda
