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
#include <tvm/runtime/profiling.h>

#include <chrono>
#include <regex>

#include "../src/runtime/opencl/opencl_common.h"

using namespace tvm::runtime;
using namespace tvm::runtime::cl;

namespace {
// This kernel was generated by TVM for conv2d operation
const std::string kernelTemplate = R"(
// Function: kernel_name_placeholder0
__kernel void kernel_name_placeholder0(__write_only image2d_t pad_temp_texture, __read_only image2d_t placeholder0) {
  const sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  float4 _1 = read_imagef(placeholder0, image_sampler, (int2)(((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) % 9) - 1), ((((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) / 81) * 7) + ((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) % 81) / 9)) - 1)));
  (void)write_imagef(pad_temp_texture, (int2)((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) % 9), (((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) / 9)), (((((9 <= (((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) % 81)) && ((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) % 81) < 72)) && (1 <= (((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) % 9))) && ((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) % 9) < 8)) ? _1 : ((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f))));
}

// Function: kernel_name_placeholder1
__kernel void kernel_name_placeholder1(__read_only image2d_t pad_temp_texture, __read_only image2d_t placeholder1, __write_only image2d_t compute, __read_only image2d_t placeholder2, __read_only image2d_t placeholder3) {
  const sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  float4 compute1[14];
  vstore4(((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f)), 0, (float*)compute1 + 0);
  vstore4(((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f)), 0, (float*)compute1 + 28);
  vstore4(((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f)), 0, (float*)compute1 + 4);
  vstore4(((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f)), 0, (float*)compute1 + 32);
  vstore4(((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f)), 0, (float*)compute1 + 8);
  vstore4(((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f)), 0, (float*)compute1 + 36);
  vstore4(((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f)), 0, (float*)compute1 + 12);
  vstore4(((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f)), 0, (float*)compute1 + 40);
  vstore4(((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f)), 0, (float*)compute1 + 16);
  vstore4(((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f)), 0, (float*)compute1 + 44);
  vstore4(((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f)), 0, (float*)compute1 + 20);
  vstore4(((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f)), 0, (float*)compute1 + 48);
  vstore4(((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f)), 0, (float*)compute1 + 24);
  vstore4(((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f)), 0, (float*)compute1 + 52);
  for (int rc_inner = 0; rc_inner < 128; ++rc_inner) {
    for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
      for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
        for (int rc = 0; rc < 4; ++rc) {
          float4 _1 = read_imagef(pad_temp_texture, image_sampler, (int2)((((int)get_local_id(0)) + rx_inner), ((rc_inner * 9) + ry_inner)));
          float4 _2 = read_imagef(placeholder1, image_sampler, (int2)(((((rc_inner * 36) + (rc * 9)) + (ry_inner * 3)) + rx_inner), ((((int)get_group_id(2)) * 16) + ((int)get_local_id(2)))));
          vstore4((vload4(0, (float*)compute1 + 0) + (((float*)&_1)[rc] * _2)), 0, (float*)compute1 + 0);
          float4 _3 = read_imagef(placeholder1, image_sampler, (int2)(((((rc_inner * 36) + (rc * 9)) + (ry_inner * 3)) + rx_inner), (((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8)));
          vstore4((vload4(0, (float*)compute1 + 28) + (((float*)&_1)[rc] * _3)), 0, (float*)compute1 + 28);
          float4 _4 = read_imagef(pad_temp_texture, image_sampler, (int2)((((int)get_local_id(0)) + rx_inner), (((rc_inner * 9) + ry_inner) + 1)));
          vstore4((vload4(0, (float*)compute1 + 4) + (((float*)&_4)[rc] * _2)), 0, (float*)compute1 + 4);
          vstore4((vload4(0, (float*)compute1 + 32) + (((float*)&_4)[rc] * _3)), 0, (float*)compute1 + 32);
          float4 _5 = read_imagef(pad_temp_texture, image_sampler, (int2)((((int)get_local_id(0)) + rx_inner), (((rc_inner * 9) + ry_inner) + 2)));
          vstore4((vload4(0, (float*)compute1 + 8) + (((float*)&_5)[rc] * _2)), 0, (float*)compute1 + 8);
          vstore4((vload4(0, (float*)compute1 + 36) + (((float*)&_5)[rc] * _3)), 0, (float*)compute1 + 36);
          float4 _6 = read_imagef(pad_temp_texture, image_sampler, (int2)((((int)get_local_id(0)) + rx_inner), (((rc_inner * 9) + ry_inner) + 3)));
          vstore4((vload4(0, (float*)compute1 + 12) + (((float*)&_6)[rc] * _2)), 0, (float*)compute1 + 12);
          vstore4((vload4(0, (float*)compute1 + 40) + (((float*)&_6)[rc] * _3)), 0, (float*)compute1 + 40);
          float4 _7 = read_imagef(pad_temp_texture, image_sampler, (int2)((((int)get_local_id(0)) + rx_inner), (((rc_inner * 9) + ry_inner) + 4)));
          vstore4((vload4(0, (float*)compute1 + 16) + (((float*)&_7)[rc] * _2)), 0, (float*)compute1 + 16);
          vstore4((vload4(0, (float*)compute1 + 44) + (((float*)&_7)[rc] * _3)), 0, (float*)compute1 + 44);
          float4 _8 = read_imagef(pad_temp_texture, image_sampler, (int2)((((int)get_local_id(0)) + rx_inner), (((rc_inner * 9) + ry_inner) + 5)));
          vstore4((vload4(0, (float*)compute1 + 20) + (((float*)&_8)[rc] * _2)), 0, (float*)compute1 + 20);
          vstore4((vload4(0, (float*)compute1 + 48) + (((float*)&_8)[rc] * _3)), 0, (float*)compute1 + 48);
          float4 _9 = read_imagef(pad_temp_texture, image_sampler, (int2)((((int)get_local_id(0)) + rx_inner), (((rc_inner * 9) + ry_inner) + 6)));
          vstore4((vload4(0, (float*)compute1 + 24) + (((float*)&_9)[rc] * _2)), 0, (float*)compute1 + 24);
          vstore4((vload4(0, (float*)compute1 + 52) + (((float*)&_9)[rc] * _3)), 0, (float*)compute1 + 52);
        }
      }
    }
  }
  float4 _10 = read_imagef(placeholder2, image_sampler, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  float4 _11 = read_imagef(placeholder3, image_sampler, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  (void)write_imagef(compute, (int2)(((int)get_local_id(0)), ((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7))), max(((vload4(0, (float*)compute1 + 0) * _10) + _11), ((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f))));
  float4 _12 = read_imagef(placeholder2, image_sampler, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  float4 _13 = read_imagef(placeholder3, image_sampler, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  (void)write_imagef(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 56)), max(((vload4(0, (float*)compute1 + 28) * _12) + _13), ((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f))));
  float4 _14 = read_imagef(placeholder2, image_sampler, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  float4 _15 = read_imagef(placeholder3, image_sampler, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  (void)write_imagef(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 1)), max(((vload4(0, (float*)compute1 + 4) * _14) + _15), ((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f))));
  float4 _16 = read_imagef(placeholder2, image_sampler, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  float4 _17 = read_imagef(placeholder3, image_sampler, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  (void)write_imagef(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 57)), max(((vload4(0, (float*)compute1 + 32) * _16) + _17), ((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f))));
  float4 _18 = read_imagef(placeholder2, image_sampler, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  float4 _19 = read_imagef(placeholder3, image_sampler, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  (void)write_imagef(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 2)), max(((vload4(0, (float*)compute1 + 8) * _18) + _19), ((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f))));
  float4 _20 = read_imagef(placeholder2, image_sampler, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  float4 _21 = read_imagef(placeholder3, image_sampler, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  (void)write_imagef(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 58)), max(((vload4(0, (float*)compute1 + 36) * _20) + _21), ((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f))));
  float4 _22 = read_imagef(placeholder2, image_sampler, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  float4 _23 = read_imagef(placeholder3, image_sampler, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  (void)write_imagef(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 3)), max(((vload4(0, (float*)compute1 + 12) * _22) + _23), ((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f))));
  float4 _24 = read_imagef(placeholder2, image_sampler, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  float4 _25 = read_imagef(placeholder3, image_sampler, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  (void)write_imagef(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 59)), max(((vload4(0, (float*)compute1 + 40) * _24) + _25), ((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f))));
  float4 _26 = read_imagef(placeholder2, image_sampler, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  float4 _27 = read_imagef(placeholder3, image_sampler, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  (void)write_imagef(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 4)), max(((vload4(0, (float*)compute1 + 16) * _26) + _27), ((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f))));
  float4 _28 = read_imagef(placeholder2, image_sampler, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  float4 _29 = read_imagef(placeholder3, image_sampler, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  (void)write_imagef(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 60)), max(((vload4(0, (float*)compute1 + 44) * _28) + _29), ((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f))));
  float4 _30 = read_imagef(placeholder2, image_sampler, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  float4 _31 = read_imagef(placeholder3, image_sampler, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  (void)write_imagef(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 5)), max(((vload4(0, (float*)compute1 + 20) * _30) + _31), ((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f))));
  float4 _32 = read_imagef(placeholder2, image_sampler, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  float4 _33 = read_imagef(placeholder3, image_sampler, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  (void)write_imagef(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 61)), max(((vload4(0, (float*)compute1 + 48) * _32) + _33), ((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f))));
  float4 _34 = read_imagef(placeholder2, image_sampler, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  float4 _35 = read_imagef(placeholder3, image_sampler, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  (void)write_imagef(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 6)), max(((vload4(0, (float*)compute1 + 24) * _34) + _35), ((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f))));
  float4 _36 = read_imagef(placeholder2, image_sampler, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  float4 _37 = read_imagef(placeholder3, image_sampler, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  (void)write_imagef(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 62)), max(((vload4(0, (float*)compute1 + 52) * _36) + _37), ((float4)((float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f, (float)0.000000e+00f))));
}

    )";
}  // namespace

using Timestamp = std::chrono::time_point<std::chrono::high_resolution_clock>;

class OpenCLCompileBin : public ::testing::Test {
 protected:
  virtual void SetUp() override {
    m_workspace = OpenCLWorkspace::Global();
    OpenCLThreadEntry* t = m_workspace->GetThreadEntry();
    t->kernel_table.resize(m_kernelsNum * 2);
    m_kernelNames.resize(m_kernelsNum * 2);
    m_dataSrc = "";
    m_fmap.clear();
    for (size_t i = 0; i < m_kernelsNum; ++i) {
      std::string kernel_name = "generated_kernel_" + std::to_string(i) + "_";
      std::string kernelSource =
          std::regex_replace(kernelTemplate, std::regex("kernel_name_placeholder"), kernel_name);
      FunctionInfo fi1 = {kernel_name + "0"};
      FunctionInfo fi2 = {kernel_name + "1"};
      m_fmap[fi1.name] = fi1;
      m_fmap[fi2.name] = fi2;
      m_kernelNames[i * 2] = fi1.name;
      m_kernelNames[i * 2 + 1] = fi2.name;
      m_dataSrc += kernelSource;
    }
  }

 protected:
  const size_t m_kernelsNum = 100;
  const std::string m_tmpDirName = "OpenCLCompileBin_dir";
  OpenCLWorkspace* m_workspace;
  std::string m_dataSrc;
  std::unordered_map<std::string, FunctionInfo> m_fmap;
  std::vector<std::string> m_kernelNames;
};

TEST_F(OpenCLCompileBin, SourceVsBinaryCompilationPerf) {
  double compileFromSourceTimeMS, compileFromBinTimeMS;
  std::string bytes;
  {
    OpenCLModuleNode module(m_dataSrc, "cl", m_fmap, std::string());
    module.Init();
    EXPECT_TRUE(module.SupportPreCompiledPrograms());
    Timestamp comp_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < m_kernelNames.size(); ++i) {
      OpenCLModuleNode::KTRefEntry e = {i, 1};
      module.InstallKernel(m_workspace, m_workspace->GetThreadEntry(), m_kernelNames[i], e);
    }
    Timestamp comp_end = std::chrono::high_resolution_clock::now();
    bytes = module.GetPreCompiledPrograms();
    std::chrono::duration duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(comp_end - comp_start);
    compileFromSourceTimeMS = duration.count() * 1e-6;
    std::cout << "Compile time from source:  " << compileFromSourceTimeMS << " ms." << std::endl;
  }
  {
    OpenCLModuleNode module(m_dataSrc, "cl", m_fmap, std::string());
    module.Init();
    EXPECT_TRUE(module.SupportPreCompiledPrograms());
    module.SetPreCompiledPrograms(bytes);
    Timestamp comp_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < m_kernelNames.size(); ++i) {
      OpenCLModuleNode::KTRefEntry e = {i, 1};
      module.InstallKernel(m_workspace, m_workspace->GetThreadEntry(), m_kernelNames[i], e);
    }
    Timestamp comp_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(comp_end - comp_start);
    compileFromBinTimeMS = duration.count() * 1e-6;
    std::cout << "Compile time from bin:  " << compileFromBinTimeMS << " ms." << std::endl;
  }
  ASSERT_LT(compileFromBinTimeMS, compileFromSourceTimeMS);
}
