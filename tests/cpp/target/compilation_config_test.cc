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
#include <tvm/target/compilation_config.h>
#include <tvm/target/target.h>

namespace tvm {
namespace {

Target TestCpuTarget() { return Target("llvm -mcpu arm64"); }

Target TestCudaTarget() { return Target("nvidia/tesla-p40"); }

Target TestDefaultCpuTarget() { return Target("llvm"); }

Target TestExtDevTarget() { return Target("ext_dev"); }

TVM_REGISTER_TARGET_KIND("test_ext_codegen_1", kDLCUDA)
    .set_attr<Bool>(tvm::attr::kIsExternalCodegen, Bool(true));

TVM_REGISTER_TARGET_KIND("test_ext_codegen_2", kDLCUDA)
    .set_attr<Bool>(tvm::attr::kIsExternalCodegen, Bool(true));

Target TestExtCodegenTarget1() { return Target("test_ext_codegen_1"); }
Target TestExtCodegenTarget2() { return Target("test_ext_codegen_2"); }

CompilationConfig TestCompilationConfig() {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  Target host_target = TestDefaultCpuTarget();
  Target cuda_target = Target::WithHost(TestCudaTarget(), host_target);
  Target cpu_target = Target::WithHost(TestCpuTarget(), host_target);
  return CompilationConfig(pass_ctx, {cuda_target, cpu_target});
}

TEST(CompilationConfig, Constructor_Heterogeneous_RuleA_RuleF_ReplaceHost) {
  transform::PassContext pass_ctx = transform::PassContext::Create();

  Target host_target = TestDefaultCpuTarget();
  Target cuda_target = Target::WithHost(TestCudaTarget(), host_target);
  Target ignored_target = TestExtDevTarget();
  Target raw_cpu_target = Target::WithHost(TestCpuTarget(), ignored_target);
  CompilationConfig config(pass_ctx, {cuda_target, raw_cpu_target});

  Target cpu_target = Target::WithHost(TestCpuTarget(), host_target);
  VirtualDevice expected_default_primitive_virtual_device(kDLCPU, 0, cpu_target);
  VirtualDevice expected_host_virtual_device(kDLCPU, 0, host_target);

  // Host is chosen as per Rule A.
  EXPECT_TRUE(config->host_target.defined());
  EXPECT_TRUE(StructuralEqual()(config->host_target, host_target));
  EXPECT_TRUE(StructuralEqual()(config->host_virtual_device, expected_host_virtual_device));

  ASSERT_EQ(config->primitive_targets.size(), 2);
  EXPECT_TRUE(StructuralEqual()(config->primitive_targets[0], cuda_target));
  // The host is taken from first raw target and overwritten in second.
  EXPECT_TRUE(StructuralEqual()(config->primitive_targets[1], cpu_target));

  // Default primitive virtual device chosen as per Rule F
  EXPECT_TRUE(StructuralEqual()(config->default_primitive_virtual_device,
                                expected_default_primitive_virtual_device));

  // Heterogeneous case.
  ASSERT_FALSE(config->optional_homogeneous_target.defined());
}

TEST(CompilationConfig, Constructor_Homogeneous_RuleA_RuleE) {
  transform::PassContext pass_ctx = transform::PassContext::Create();

  Target host_target = TestDefaultCpuTarget();
  Target cuda_target = Target::WithHost(TestCudaTarget(), host_target);
  CompilationConfig config(pass_ctx, {cuda_target});

  VirtualDevice expected_default_primitive_virtual_device(kDLCUDA, 0, cuda_target);
  VirtualDevice expected_host_virtual_device(kDLCPU, 0, host_target);

  // Host is chose as per Rule A.
  EXPECT_TRUE(config->host_target.defined());
  EXPECT_TRUE(StructuralEqual()(config->host_target, host_target));
  EXPECT_TRUE(StructuralEqual()(config->host_virtual_device, expected_host_virtual_device));

  ASSERT_EQ(config->primitive_targets.size(), 1);
  EXPECT_TRUE(StructuralEqual()(config->primitive_targets[0], cuda_target));

  // Default primitive virtual device chose as per rule E.
  EXPECT_TRUE(StructuralEqual()(config->default_primitive_virtual_device,
                                expected_default_primitive_virtual_device));

  // Homogeneous case.
  ASSERT_TRUE(config->optional_homogeneous_target.defined());
  EXPECT_TRUE(StructuralEqual()(config->optional_homogeneous_target, cuda_target));
}

TEST(CompilationConfig, Constructor_Heterogeneous_RuleB_RuleD) {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("relay.fallback_device_type", Integer(static_cast<int>(kDLCUDA)));

  Target raw_cuda_target = TestCudaTarget();
  Target raw_cpu_target = TestCpuTarget();
  CompilationConfig config(pass_ctx, {raw_cuda_target, raw_cpu_target});

  Target host_target = TestCpuTarget();
  Target cuda_target = Target::WithHost(TestCudaTarget(), host_target);
  Target cpu_target = Target::WithHost(TestCpuTarget(), host_target);

  VirtualDevice expected_default_primitive_virtual_device(kDLCUDA, 0, cuda_target);
  VirtualDevice expected_host_virtual_device(kDLCPU, 0, host_target);

  // Host is chosen as per Rule B.
  EXPECT_TRUE(config->host_target.defined());
  EXPECT_TRUE(StructuralEqual()(config->host_target, host_target));
  EXPECT_TRUE(StructuralEqual()(config->host_virtual_device, expected_host_virtual_device));

  ASSERT_EQ(config->primitive_targets.size(), 2);
  EXPECT_TRUE(StructuralEqual()(config->primitive_targets[0], cuda_target));
  EXPECT_TRUE(StructuralEqual()(config->primitive_targets[1], cpu_target));

  // Default primitive virtual device chosen as per Rule D
  EXPECT_TRUE(StructuralEqual()(config->default_primitive_virtual_device,
                                expected_default_primitive_virtual_device));

  // Heterogeneous case.
  ASSERT_FALSE(config->optional_homogeneous_target.defined());
}

TEST(CompilationConfig, Constructor_Homogeneous_RuleC_RuleE) {
  transform::PassContext pass_ctx = transform::PassContext::Create();

  Target raw_cuda_target = TestCudaTarget();
  CompilationConfig config(pass_ctx, {raw_cuda_target});

  Target host_target = TestDefaultCpuTarget();
  Target cuda_target = Target::WithHost(TestCudaTarget(), host_target);
  Target cpu_target = Target::WithHost(TestDefaultCpuTarget(), host_target);

  VirtualDevice expected_default_primitive_virtual_device(kDLCUDA, 0, cuda_target);
  VirtualDevice expected_host_virtual_device(kDLCPU, 0, host_target);

  // Host is chosen as per Rule C.
  EXPECT_TRUE(config->host_target.defined());
  EXPECT_TRUE(StructuralEqual()(config->host_target, host_target));
  EXPECT_TRUE(StructuralEqual()(config->host_virtual_device, expected_host_virtual_device));

  ASSERT_EQ(config->primitive_targets.size(), 1);
  EXPECT_TRUE(StructuralEqual()(config->primitive_targets[0], cuda_target));

  // Default primitive virtual device chosen as per Rule E
  EXPECT_TRUE(StructuralEqual()(config->default_primitive_virtual_device,
                                expected_default_primitive_virtual_device));

  // Homogeneous case.
  ASSERT_TRUE(config->optional_homogeneous_target.defined());
  EXPECT_TRUE(StructuralEqual()(config->optional_homogeneous_target, cuda_target));
}

TEST(CompilationConfig, Constructor_Heterogeneous_CorrectOrdering) {
  transform::PassContext pass_ctx = transform::PassContext::Create();

  Target host_target = TestDefaultCpuTarget();
  Target cuda_target = Target::WithHost(TestCudaTarget(), host_target);
  Target ext_codegen1_target = Target::WithHost(TestExtCodegenTarget1(), host_target);
  Target ext_codegen2_target = Target::WithHost(TestExtCodegenTarget2(), host_target);
  CompilationConfig config(pass_ctx, {cuda_target, ext_codegen1_target, ext_codegen2_target});

  ASSERT_EQ(config->primitive_targets.size(), 3);
  EXPECT_TRUE(StructuralEqual()(config->primitive_targets[0], cuda_target));
  EXPECT_TRUE(StructuralEqual()(config->primitive_targets[1], ext_codegen1_target));
  EXPECT_TRUE(StructuralEqual()(config->primitive_targets[2], ext_codegen2_target));
}

TEST(CompilationConfig, Constructor_Heterogeneous_InvalidOrdering) {
  transform::PassContext pass_ctx = transform::PassContext::Create();

  Target host_target = TestDefaultCpuTarget();
  Target ext_codegen1_target = Target::WithHost(TestExtCodegenTarget1(), host_target);
  Target cuda_target = Target::WithHost(TestCudaTarget(), host_target);
  Target ext_codegen2_target = Target::WithHost(TestExtCodegenTarget2(), host_target);

  EXPECT_ANY_THROW(
      CompilationConfig(pass_ctx, {ext_codegen1_target, cuda_target, ext_codegen2_target}));
}

TEST(CompilationConfig, Constructor_Homogenous_JustExternalCodegen) {
  transform::PassContext pass_ctx = transform::PassContext::Create();

  Target host_target = TestDefaultCpuTarget();
  Target ext_codegen1_target = Target::WithHost(TestExtCodegenTarget1(), host_target);

  CompilationConfig config(pass_ctx, {ext_codegen1_target});
  ASSERT_EQ(config->primitive_targets.size(), 1);
  EXPECT_TRUE(StructuralEqual()(config->primitive_targets[0], ext_codegen1_target));
}

TEST(CompliationConfig, Constructor_DuplicateKinds) {
  transform::PassContext pass_ctx = transform::PassContext::Create();

  Target host_target = TestDefaultCpuTarget();
  Target cuda_target_1 = Target::WithHost(TestCudaTarget(), host_target);
  Target cuda_target_2 = Target::WithHost(TestCudaTarget(), host_target);

  EXPECT_ANY_THROW(CompilationConfig(pass_ctx, {cuda_target_1, cuda_target_2}));
}

TEST(CompilationConfig, Constructor_NoTargets) {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  EXPECT_ANY_THROW(CompilationConfig(pass_ctx, {}));
}

TEST(CompilationConfig, Constructor_InvalidAttribute) {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("relay.fallback_device_type", Integer(static_cast<int>(kInvalidDeviceType)));

  Target cuda_target = Target::WithHost(TestCudaTarget(), TestDefaultCpuTarget());
  EXPECT_ANY_THROW(CompilationConfig(pass_ctx, {cuda_target}));
}

TEST(CompilationConfig, Constructor_NoMatchingPrimitiveTarget) {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("relay.fallback_device_type", Integer(static_cast<int>(kDLMetal)));
  Target host_target = TestDefaultCpuTarget();
  Target cuda_target = Target::WithHost(TestCudaTarget(), host_target);
  EXPECT_ANY_THROW(CompilationConfig(pass_ctx, {cuda_target}));
}

TEST(CompilationConfig, Constructor_DefaultNoMatchingPrimitiveTarget) {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  Target host_target = TestDefaultCpuTarget();
  Target cuda_target = Target::WithHost(TestCudaTarget(), host_target);
  Target ext_target = Target::WithHost(TestExtDevTarget(), host_target);
  EXPECT_ANY_THROW(CompilationConfig config(pass_ctx, {cuda_target, ext_target}));
}

TEST(CompilationConfig, Constructor_Idempotent) {
  transform::PassContext pass_ctx = transform::PassContext::Create();

  Target host_target = TestDefaultCpuTarget();
  Target cuda_target = Target::WithHost(TestCudaTarget(), host_target);
  Target ignored_target = TestExtDevTarget();
  Target raw_cpu_target = Target::WithHost(TestCpuTarget(), ignored_target);
  CompilationConfig orig_config(pass_ctx, {cuda_target, raw_cpu_target});

  CompilationConfig reconstructed_config(pass_ctx, orig_config->primitive_targets);

  ASSERT_EQ(orig_config->primitive_targets.size(), reconstructed_config->primitive_targets.size());
  ASSERT_TRUE(StructuralEqual()(orig_config->primitive_targets[0],
                                reconstructed_config->primitive_targets[0]));
  ASSERT_TRUE(StructuralEqual()(orig_config->primitive_targets[1],
                                reconstructed_config->primitive_targets[1]));
}

TEST(CompilationConfig, FindPrimitiveTargetForDeviceOrFail_Valid) {
  CompilationConfig config = TestCompilationConfig();
  Target cpu_target = Target::WithHost(TestCpuTarget(), TestDefaultCpuTarget());
  ASSERT_TRUE(StructuralEqual()(config->FindPrimitiveTargetForDeviceOrFail(kDLCPU), cpu_target));
}

TEST(CompilationConfig, FindPrimitiveTargetForDeviceOrFail_Invalid) {
  CompilationConfig config = TestCompilationConfig();
  EXPECT_ANY_THROW(config->FindPrimitiveTargetForDeviceOrFail(kDLMetal));
}

TEST(CompilationConfig, FindPrimitiveTargetForKind_Found) {
  CompilationConfig config = TestCompilationConfig();
  Target cuda_target = Target::WithHost(TestCudaTarget(), TestDefaultCpuTarget());
  ASSERT_TRUE(StructuralEqual()(config->FindPrimitiveTargetForKind("cuda").value(), cuda_target));
}

TEST(CompilationConfig, FindPrimitiveTargetForKind_NotFound) {
  CompilationConfig config = TestCompilationConfig();
  ASSERT_FALSE(config->FindPrimitiveTargetForKind("cutlass").defined());
}

TEST(CompilationConfig, CanonicalTarget) {
  Target host_target = TestDefaultCpuTarget();
  Target cuda_target = TestCudaTarget();
  Target cpu_target = TestCpuTarget();
  CompilationConfig config = TestCompilationConfig();

  {
    Target other_cuda_target = Target::WithHost(TestCudaTarget(), TestDefaultCpuTarget());
    ASSERT_NE(other_cuda_target, cuda_target);
    ASSERT_EQ(config->CanonicalTarget(other_cuda_target),
              config->FindPrimitiveTargetForKind("cuda"));
  }
  {
    Target other_host_target = TestDefaultCpuTarget();
    ASSERT_NE(other_host_target, cuda_target);
    ASSERT_EQ(config->CanonicalTarget(other_host_target), config->host_target);
  }
  {
    Target other_target("cuda -max_num_threads=7");
    ASSERT_EQ(config->CanonicalTarget(other_target), other_target);
  }
}

TEST(CompilationConfig, CanonicalVirtualDevice) {
  Target host_target = TestDefaultCpuTarget();
  Target cuda_target = TestCudaTarget();
  Target cpu_target = TestCpuTarget();
  CompilationConfig config = TestCompilationConfig();

  {
    VirtualDevice in = VirtualDevice(kDLCPU);
    VirtualDevice actual = config->CanonicalVirtualDevice(in);
    ASSERT_TRUE(actual->target.defined());
    EXPECT_TRUE(StructuralEqual()(actual->target, Target::WithHost(cpu_target, host_target)));
    EXPECT_EQ(config->CanonicalVirtualDevice(in), actual);
  }
  {
    VirtualDevice in = VirtualDevice(kDLCUDA);
    VirtualDevice actual = config->CanonicalVirtualDevice(in);
    ASSERT_TRUE(actual->target.defined());
    EXPECT_TRUE(StructuralEqual()(actual->target, Target::WithHost(cuda_target, host_target)));
    EXPECT_EQ(config->CanonicalVirtualDevice(in), actual);
  }
  {
    Target other_cuda_target = Target::WithHost(TestCudaTarget(), TestDefaultCpuTarget());
    VirtualDevice in = VirtualDevice(kDLCUDA, -1, other_cuda_target);
    VirtualDevice actual = config->CanonicalVirtualDevice(in);
    ASSERT_EQ(actual->target, config->FindPrimitiveTargetForKind("cuda"));
  }
  {
    VirtualDevice in = VirtualDevice::ForMemoryScope("scope");
    VirtualDevice actual = config->CanonicalVirtualDevice(in);
    EXPECT_EQ(config->CanonicalVirtualDevice(in), actual);
  }
  {
    VirtualDevice in = VirtualDevice::FullyUnconstrained();
    VirtualDevice actual = config->CanonicalVirtualDevice(in);
    EXPECT_EQ(config->CanonicalVirtualDevice(in), actual);
  }
  {
    VirtualDevice in = VirtualDevice();  // ie structurally equal to FullyUnconstrained.
    VirtualDevice actual = config->CanonicalVirtualDevice(in);
    EXPECT_EQ(config->CanonicalVirtualDevice(in), VirtualDevice::FullyUnconstrained());
  }
}

TEST(CompilationConfig, CanonicalVirtualDevice_NoMatchingTarget) {
  CompilationConfig config = TestCompilationConfig();
  VirtualDevice no_such_target(kDLMetal);
  EXPECT_ANY_THROW(config->CanonicalVirtualDevice(no_such_target));
}

}  // namespace
}  // namespace tvm
