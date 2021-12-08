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

/*
 * Just a smoke test for the device planner's unification domain, mostly to tease out how we'd
 * like to organize our cpp unit tests for functionality that's not obviously a Pass or should
 * be exposed via FFI.
 */

// TODO(mbs): Revisit cpp unit test layout or setup include dir at root of src/
#include "../../../src/relay/transforms/device_domains.h"

#include <gtest/gtest.h>
#include <tvm/parser/parser.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {
namespace transform {
namespace {

IRModule TestModule() {
  return InferType()(tvm::parser::ParseModule("test", R"(
    #[version = "0.0.5"]
    def @f(%x : Tensor[(3, 7), float32], %y : Tensor[(3, 7), float32]) {
      add(%x, %y)
    }
  )"));
}

TEST(DeviceDomains, SmokeTest) {
  SEScope cpu = SEScope::ForDeviceType(kDLCPU);
  SEScope cuda = SEScope::ForDeviceType(kDLCUDA);
  TargetMap target_map;
  target_map.Set(Integer(static_cast<int>(kDLCPU)), Target("llvm"));
  target_map.Set(Integer(static_cast<int>(kDLCUDA)), Target("cuda"));
  transform::PassContext ctxt = transform::PassContext::Create();
  CompilationConfig config(ctxt, target_map, /*optional_host_target=*/{});
  DeviceDomains domains(config);
  IRModule mod = TestModule();
  Function f = Downcast<Function>(mod->Lookup("f"));

  DeviceDomainPtr actual_add_domain = domains.DomainForCallee(Downcast<Call>(f->body));
  DeviceDomainPtr x_domain = domains.DomainFor(f->params[0]);
  DeviceDomainPtr y_domain = domains.DomainFor(f->params[1]);
  DeviceDomainPtr result_domain = domains.Free(f->ret_type);
  std::vector<DeviceDomainPtr> arg_and_results;
  arg_and_results.push_back(x_domain);
  arg_and_results.push_back(y_domain);
  arg_and_results.push_back(result_domain);
  DeviceDomainPtr implied_add_domain = domains.MakeHigherOrderDomain(std::move(arg_and_results));
  EXPECT_FALSE(domains.UnifyOrNull(actual_add_domain, implied_add_domain) == nullptr);
  EXPECT_FALSE(domains.UnifyOrNull(
                   x_domain, domains.ForSEScope(f->params[0]->checked_type(), cuda)) == nullptr);

  EXPECT_EQ(domains.ResultSEScope(y_domain), config->CanonicalSEScope(cuda));
  EXPECT_EQ(domains.ResultSEScope(result_domain), config->CanonicalSEScope(cuda));
}

}  // namespace
}  // namespace transform
}  // namespace relay
}  // namespace tvm
