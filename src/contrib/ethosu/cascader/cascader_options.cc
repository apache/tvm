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
#include "cascader_options.h"

#include <utility>

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

void CascaderOptionsNode::VisitAttrs(AttrVisitor* v) {
  v->Visit("cascade_region", &cascade_region);
  v->Visit("max_proposals", &max_proposals);
  v->Visit("stripe_factors", &stripe_factors);
  v->Visit("max_plan_size", &max_plan_size);
  v->Visit("always_copy_size", &always_copy_size);
  v->Visit("enable_striping", &enable_striping);
}

CascaderOptions::CascaderOptions(const MemoryRegion& cascade_region, int max_proposals,
                                 int stripe_factors, int max_plan_size, int always_copy_size,
                                 bool enable_striping) {
  auto n = make_object<CascaderOptionsNode>();
  n->cascade_region = std::move(cascade_region);
  n->max_proposals = max_proposals;
  n->stripe_factors = stripe_factors;
  n->max_plan_size = max_plan_size;
  n->always_copy_size = always_copy_size;
  n->enable_striping = enable_striping;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.CascaderOptions")
    .set_body_typed([](MemoryRegion cascade_region, int max_proposals, int stripe_factors,
                       int max_plan_size, int always_copy_size, bool enable_striping) {
      return CascaderOptions(cascade_region, max_proposals, stripe_factors, max_plan_size,
                             always_copy_size, enable_striping);
    });

TVM_REGISTER_NODE_TYPE(CascaderOptionsNode);

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm
