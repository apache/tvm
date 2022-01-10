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

CompilationConfig TestCompilationConfig() {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  TargetMap legacy_target_map;
  legacy_target_map.Set(Integer(static_cast<int>(kDLCUDA)), TestCudaTarget());
  legacy_target_map.Set(Integer(static_cast<int>(kDLCPU)), TestCpuTarget());
  return CompilationConfig(pass_ctx, legacy_target_map, TestDefaultCpuTarget());
}

TEST(CompilationConfig, Constructor_Homogeneous_FallbackCPUHost) {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  Target host_target = TestDefaultCpuTarget();
  Target cuda_target = TestCudaTarget();
  TargetMap legacy_target_map;
  legacy_target_map.Set(Integer(static_cast<int>(kDLCUDA)), cuda_target);
  CompilationConfig config(pass_ctx, legacy_target_map, /*optional_host_target_arg=*/{});

  VirtualDevice expected_default_primitive_virtual_device(
      kDLCUDA, 0, Target::WithHost(cuda_target, host_target));
  VirtualDevice expected_host_virtual_device(kDLCPU, 0, host_target);

  ASSERT_EQ(config->legacy_target_map.size(), 1);
  EXPECT_TRUE(StructuralEqual()((*config->legacy_target_map.begin()).second,
                                Target::WithHost(cuda_target, host_target)));
  EXPECT_TRUE(config->host_target.defined());
  EXPECT_TRUE(StructuralEqual()(config->host_target, host_target));
  ASSERT_EQ(config->primitive_targets.size(), 1);
  EXPECT_TRUE(
      StructuralEqual()(config->primitive_targets[0], Target::WithHost(cuda_target, host_target)));
  EXPECT_TRUE(StructuralEqual()(config->default_primitive_virtual_device,
                                expected_default_primitive_virtual_device));
  EXPECT_TRUE(StructuralEqual()(config->host_virtual_device, expected_host_virtual_device));
  ASSERT_TRUE(config->optional_homogeneous_target.defined());
  EXPECT_TRUE(StructuralEqual()(config->optional_homogeneous_target,
                                Target::WithHost(cuda_target, host_target)));
}

TEST(CompilationConfig, Constructor_Homegenoous_InnerHost) {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  Target host_target = TestCpuTarget();
  Target cuda_target = Target::WithHost(TestCudaTarget(), host_target);
  TargetMap legacy_target_map;
  legacy_target_map.Set(Integer(static_cast<int>(kDLCUDA)), cuda_target);
  CompilationConfig config(pass_ctx, legacy_target_map, /*optional_host_target_arg=*/{});

  EXPECT_TRUE(StructuralEqual()(config->host_target, host_target));
}

TEST(CompilationConfig, Constructor_Homogenous_CPUHost) {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  Target host_target = TestCpuTarget();
  Target cpu_target = TestCpuTarget();
  TargetMap legacy_target_map;
  legacy_target_map.Set(Integer(static_cast<int>(kDLCPU)),
                        Target::WithHost(cpu_target, host_target));
  CompilationConfig config(pass_ctx, legacy_target_map, /*optional_host_target_arg=*/{});

  EXPECT_TRUE(StructuralEqual()(config->host_target, cpu_target));
  ASSERT_TRUE(config->optional_homogeneous_target.defined());
  EXPECT_TRUE(StructuralEqual()(config->optional_homogeneous_target,
                                Target::WithHost(cpu_target, cpu_target)));
}

TEST(CompilationConfig, Constructor_Hetrogeneous_FallbackCPUHost) {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("relay.fallback_device_type", Integer(static_cast<int>(kDLCUDA)));
  Target host_target = TestDefaultCpuTarget();
  Target cuda_target = TestCudaTarget();
  Target cpu_target = TestCpuTarget();
  TargetMap legacy_target_map;
  legacy_target_map.Set(Integer(static_cast<int>(kDLCPU)),
                        Target::WithHost(cpu_target, host_target));
  legacy_target_map.Set(Integer(static_cast<int>(kDLCUDA)),
                        Target::WithHost(cuda_target, host_target));
  CompilationConfig config(pass_ctx, legacy_target_map, /*optional_host_target_arg=*/{});

  VirtualDevice expected_default_primitive_virtual_device(
      kDLCUDA, 0, Target::WithHost(cuda_target, host_target));
  VirtualDevice expected_host_virtual_device(kDLCPU, 0, host_target);

  ASSERT_EQ(config->legacy_target_map.size(), 2);
  for (const auto& pair : config->legacy_target_map) {
    if (pair.first->value == kDLCPU) {
      EXPECT_TRUE(StructuralEqual()(pair.second, Target::WithHost(cpu_target, host_target)));
    } else if (pair.first->value == kDLCUDA) {
      EXPECT_TRUE(StructuralEqual()(pair.second, Target::WithHost(cuda_target, host_target)));
    }
  }
  EXPECT_TRUE(config->host_target.defined());
  EXPECT_TRUE(StructuralEqual()(config->host_target, host_target));
  EXPECT_TRUE(StructuralEqual()(config->default_primitive_virtual_device,
                                expected_default_primitive_virtual_device));
  EXPECT_TRUE(StructuralEqual()(config->host_virtual_device, expected_host_virtual_device));
  EXPECT_FALSE(config->optional_homogeneous_target.defined());
}

TEST(CompilationConfig, Constructor_Hetrogeneous_ExplicitHost) {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("relay.fallback_device_type", Integer(static_cast<int>(kDLCUDA)));
  Target host_target = TestCpuTarget();
  Target cuda_target = TestCudaTarget();
  Target cpu_target = TestCpuTarget();
  TargetMap legacy_target_map;
  legacy_target_map.Set(Integer(static_cast<int>(kDLCPU)),
                        Target::WithHost(cpu_target, host_target));
  legacy_target_map.Set(Integer(static_cast<int>(kDLCUDA)),
                        Target::WithHost(cuda_target, host_target));
  CompilationConfig config(pass_ctx, legacy_target_map, host_target);

  VirtualDevice expected_default_primitive_virtual_device(
      kDLCUDA, 0, Target::WithHost(cuda_target, host_target));
  VirtualDevice expected_host_virtual_device(kDLCPU, 0, host_target);

  ASSERT_EQ(config->legacy_target_map.size(), 2);
  for (const auto& pair : config->legacy_target_map) {
    if (pair.first->value == kDLCPU) {
      EXPECT_TRUE(StructuralEqual()(pair.second, Target::WithHost(cpu_target, host_target)));
    } else if (pair.first->value == kDLCUDA) {
      EXPECT_TRUE(StructuralEqual()(pair.second, Target::WithHost(cuda_target, host_target)));
    }
  }
  EXPECT_TRUE(config->host_target.defined());
  EXPECT_TRUE(StructuralEqual()(config->host_target, host_target));
  ASSERT_EQ(config->primitive_targets.size(), 2);
  EXPECT_TRUE(StructuralEqual()(config->default_primitive_virtual_device,
                                expected_default_primitive_virtual_device));
  EXPECT_TRUE(StructuralEqual()(config->host_virtual_device, expected_host_virtual_device));
  EXPECT_FALSE(config->optional_homogeneous_target.defined());
}

TEST(CompilationConfig, Constructor_InvalidAttribute) {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("relay.fallback_device_type", Integer(static_cast<int>(kInvalidDeviceType)));
  TargetMap legacy_target_map;
  legacy_target_map.Set(Integer(static_cast<int>(kDLCUDA)), TestCudaTarget());
  EXPECT_ANY_THROW(
      CompilationConfig config(pass_ctx, legacy_target_map, /*optional_host_target_arg=*/{}));
}

TEST(CompilationConfig, Constructor_NoMatchingPrimitiveTarget) {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  pass_ctx->config.Set("relay.fallback_device_type", Integer(static_cast<int>(kDLMetal)));
  TargetMap legacy_target_map;
  legacy_target_map.Set(Integer(static_cast<int>(kDLCUDA)), TestCudaTarget());
  EXPECT_ANY_THROW(
      CompilationConfig config(pass_ctx, legacy_target_map, /*optional_host_target_arg=*/{}));
}

TEST(CompilationConfig, Constructor_DefaultNoMatchingPrimitiveTarget) {
  transform::PassContext pass_ctx = transform::PassContext::Create();
  TargetMap legacy_target_map;
  legacy_target_map.Set(Integer(static_cast<int>(kDLCUDA)), TestCudaTarget());
  legacy_target_map.Set(Integer(static_cast<int>(kDLExtDev)), TestExtDevTarget());
  EXPECT_ANY_THROW(
      CompilationConfig config(pass_ctx, legacy_target_map, /*optional_host_target_arg=*/{}));
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
}

TEST(CompilationConfig, CanonicalVirtualDevice_NoDevice) {
  CompilationConfig config = TestCompilationConfig();
  VirtualDevice fully_unconstrained;
  EXPECT_ANY_THROW(config->CanonicalVirtualDevice(fully_unconstrained));
  VirtualDevice missing_device(kInvalidDeviceType, 3, {}, "local");
  EXPECT_ANY_THROW(config->CanonicalVirtualDevice(missing_device));
}

TEST(CompilationConfig, CanonicalVirtualDevice_NoMatchingTarget) {
  CompilationConfig config = TestCompilationConfig();
  VirtualDevice no_such_target(kDLMetal);
  EXPECT_ANY_THROW(config->CanonicalVirtualDevice(no_such_target));
}

}  // namespace
}  // namespace tvm
