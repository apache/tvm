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
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/exec_scope.h>
#include <tvm/tirx/op/gpu.h>

#include <queue>

namespace tvm {
namespace tirx {
using namespace tvm::prim;

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

TVM_FFI_STATIC_INIT_BLOCK() { ExecScopeNode::RegisterReflection(); }

/******** Definition of Execution Scope ********/
//
// "kernel" is retained as a structural label for the ``kKernelCluster`` /
// ``kKernelCta`` ScopeBinding parent string, even though ``ScopeKind::kKernel``
// no longer exists. Treat it as the virtual root: wider than every real
// ScopeKind. Real ScopeKinds compare via ``ScopeKindHigher``.
static constexpr int kRootScopeRank = -1;  // wider than any real ScopeKind
static int ScopeNameRank(const ffi::String& name) {
  if (name == "kernel") return kRootScopeRank;
  return static_cast<int>(StringToScopeKind(name));
}

bool ScopeNameHigher(const ffi::String& a, const ffi::String& b) {
  return ScopeNameRank(a) < ScopeNameRank(b);
}

ExecScope::ExecScope(ScopeKind kind) {
  auto n = ffi::make_object<ExecScopeNode>();
  n->kind = kind;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.ExecScope", [](ffi::String name) { return ExecScope(name); });
}

}  // namespace tirx
}  // namespace tvm
