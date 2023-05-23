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
#include "pareto.h"

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <array>
#include <utility>
#include <vector>

#include "common.h"
#include "plan.h"
#include "proposal.h"
#include "tensor_config.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

template <int N>
std::vector<bool> GetParetoFrontier(const std::vector<std::array<float, N>>& costs) {
  std::vector<bool> is_optimal(costs.size(), true);
  for (size_t i = 0; i < costs.size(); i++) {
    if (is_optimal[i]) {
      for (size_t j = 0; j < costs.size(); j++) {
        if (is_optimal[j]) {
          bool optimal = false;
          for (size_t k = 0; k < N; k++) {
            if (costs[i][k] > costs[j][k]) {
              optimal = true;
              break;
            }
          }
          is_optimal[j] = optimal;
        }
      }
      is_optimal[i] = true;
    }
  }
  return is_optimal;
}

template <class T>
std::vector<T> ThinVector(const std::vector<T>& vec, size_t max_size) {
  if (max_size < 1) {
    return std::vector<T>();
  }
  if (vec.size() <= max_size || vec.size() == 0) {
    return vec;
  }
  if (max_size == 1) {
    return std::vector<T>{vec[0]};
  }
  std::vector<T> thin_vec;
  float step = static_cast<float>(vec.size()) / static_cast<float>(max_size - 1);
  for (float i = 0; i < vec.size() - 1; i += step) {
    thin_vec.push_back(vec[static_cast<int>(i)]);
  }
  thin_vec.push_back(vec.back());
  return thin_vec;
}

std::vector<Plan> ParetoCullPlans(std::vector<Plan> plans, size_t max_plans,
                                  bool disable_pareto_metric) {
  if (plans.size() <= max_plans) {
    return plans;
  }
  if (disable_pareto_metric) {
    // Sample from all plans
    return ThinVector(plans, max_plans);
  }

  std::sort(plans.begin(), plans.end(), [](const Plan& a, const Plan& b) -> bool {
    if (a->GetMemoryUsage() == b->GetMemoryUsage()) {
      return a->GetCycles() < b->GetCycles();
    }
    return a->GetMemoryUsage() < b->GetMemoryUsage();
  });
  std::vector<std::array<float, 2>> costs;
  for (const auto& plan : plans) {
    std::array<float, 2> cost = {static_cast<float>(plan->GetMemoryUsage()),
                                 static_cast<float>(plan->GetCycles())};
    costs.emplace_back(cost);
  }
  std::vector<bool> is_optimal = GetParetoFrontier<2>(costs);
  std::vector<Plan> optimal_plans;
  size_t i = 0;
  for (bool optimal : is_optimal) {
    if (optimal) {
      optimal_plans.push_back(plans[i]);
    }
    i++;
  }
  if (optimal_plans.size() <= max_plans) {
    return optimal_plans;
  }
  return ThinVector(optimal_plans, max_plans);
}

std::vector<Proposal> ParetoCullProposals(std::vector<Proposal> proposals, size_t max_proposals,
                                          bool disable_pareto_metric) {
  if (disable_pareto_metric) {
    // Sample from all Proposals
    return ThinVector(proposals, max_proposals);
  }

  std::sort(proposals.begin(), proposals.end(), [](const Proposal& a, const Proposal& b) -> bool {
    if (a->GetMemoryUsage() == b->GetMemoryUsage()) {
      return a->GetCycles() < b->GetCycles();
    }
    return a->GetMemoryUsage() < b->GetMemoryUsage();
  });
  std::vector<std::array<float, 2>> costs;
  for (const auto& proposal : proposals) {
    std::array<float, 2> cost = {static_cast<float>(proposal->GetMemoryUsage()),
                                 static_cast<float>(proposal->GetCycles())};
    costs.emplace_back(cost);
  }
  std::vector<bool> is_optimal = GetParetoFrontier<2>(costs);
  std::vector<Proposal> optimal_proposals;
  size_t i = 0;
  for (bool optimal : is_optimal) {
    if (optimal) {
      optimal_proposals.push_back(proposals[i]);
    }
    i++;
  }
  if (optimal_proposals.size() <= max_proposals) {
    return optimal_proposals;
  }
  return ThinVector(optimal_proposals, max_proposals);
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.GetParetoFrontier")
    .set_body_typed([](Array<Array<FloatImm>> tcosts) {
      std::vector<std::array<float, 2>> costs;
      for (const auto& tcost : tcosts) {
        ICHECK_EQ(tcost.size(), 2);
        std::array<float, 2> point = {static_cast<float>(tcost[0]->value),
                                      static_cast<float>(tcost[1]->value)};
        costs.push_back(point);
      }
      Array<Bool> is_optimal;
      for (bool opt : GetParetoFrontier<2>(costs)) {
        is_optimal.push_back(Bool(opt));
      }
      return is_optimal;
    });

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.ThinVector")
    .set_body_typed([](Array<ObjectRef> vec, int max_size) {
      std::vector<ObjectRef> vvec(vec.begin(), vec.end());
      return Array<ObjectRef>(ThinVector<ObjectRef>(vvec, max_size));
    });

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.ParetoCullPlans")
    .set_body_typed([](Array<Plan> plans, int max_size, bool disable_pareto_metric) {
      std::vector<Plan> vplans(plans.begin(), plans.end());
      return Array<Plan>(ParetoCullPlans(vplans, max_size, disable_pareto_metric));
    });

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm
