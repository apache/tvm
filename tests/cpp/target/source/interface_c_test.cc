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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>
#include <tvm/tir/usmp/utils.h>

using ::testing::HasSubstr;

namespace tvm {
namespace codegen {

runtime::Module InterfaceCCreate(std::string module_name, Array<String> inputs,
                                 Array<String> outputs, Array<tir::usmp::AllocatedPoolInfo> pools,
                                 Map<String, tir::usmp::PoolAllocation> io_pool_allocations,
                                 Array<String> devices, int workspace_size);

namespace {

TEST(InterfaceAPI, ContainsHeaderGuards) {
  std::stringstream upper_header_guard;
  std::stringstream lower_header_guard;

  upper_header_guard << "#ifndef TVMGEN_ULTIMATE_CAT_SPOTTER_H_\n"
                     << "#define TVMGEN_ULTIMATE_CAT_SPOTTER_H_\n"
                     << "#include <stdint.h>\n\n"
                     << "#ifdef __cplusplus\n"
                     << "extern \"C\" {\n"
                     << "#endif\n\n";

  lower_header_guard << "\n#ifdef __cplusplus\n"
                     << "}\n"
                     << "#endif\n\n"
                     << "#endif // TVMGEN_ULTIMATE_CAT_SPOTTER_H_\n";

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {}, {}, {}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(upper_header_guard.str()));
  ASSERT_THAT(header_source, HasSubstr(lower_header_guard.str()));
}

TEST(InterfaceAPI, ContainsRunFunction) {
  std::stringstream run_function;

  run_function << "/*!\n"
               << " * \\brief entrypoint function for TVM module \"ultimate_cat_spotter\"\n"
               << " * \\param inputs Input tensors for the module \n"
               << " * \\param outputs Output tensors for the module \n"
               << " */\n"
               << "int32_t tvmgen_ultimate_cat_spotter_run(\n"
               << "  struct tvmgen_ultimate_cat_spotter_inputs* inputs,\n"
               << "  struct tvmgen_ultimate_cat_spotter_outputs* outputs\n"
               << ");\n";

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {}, {}, {}, 0);
  std::string header_source = test_module->GetSource();
  ASSERT_THAT(header_source, HasSubstr(run_function.str()));
}

TEST(InterfaceAPI, ContainsRunFunctionWithDevices) {
  std::stringstream run_function;

  run_function << "/*!\n"
               << " * \\brief entrypoint function for TVM module \"ultimate_cat_spotter\"\n"
               << " * \\param inputs Input tensors for the module \n"
               << " * \\param outputs Output tensors for the module \n"
               << " * \\param devices Device context pointers for the module \n"
               << " */\n"
               << "int32_t tvmgen_ultimate_cat_spotter_run(\n"
               << "  struct tvmgen_ultimate_cat_spotter_inputs* inputs,\n"
               << "  struct tvmgen_ultimate_cat_spotter_outputs* outputs,\n"
               << "  struct tvmgen_ultimate_cat_spotter_devices* devices\n"
               << ");\n";

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {}, {}, {"device"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(run_function.str()));
}

TEST(InterfaceAPI, ContainsRunFunctionWithWorkspacePools) {
  std::stringstream run_function;

  run_function << "/*!\n"
               << " * \\brief entrypoint function for TVM module \"ultimate_cat_spotter\"\n"
               << " * \\param inputs Input tensors for the module \n"
               << " * \\param outputs Output tensors for the module \n"
               << " * \\param workspace_pools Workspace memory pool pointers for the module \n"
               << " */\n"
               << "int32_t tvmgen_ultimate_cat_spotter_run(\n"
               << "  struct tvmgen_ultimate_cat_spotter_inputs* inputs,\n"
               << "  struct tvmgen_ultimate_cat_spotter_outputs* outputs,\n"
               << "  struct tvmgen_ultimate_cat_spotter_workspace_pools* workspace_pools\n"
               << ");\n";

  PoolInfo pool_info = WorkspacePoolInfo("my_memory_pool", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info =
      tir::usmp::AllocatedPoolInfo(pool_info, 100000);
  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"},
                                                 {allocated_pool_info}, {}, {}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(run_function.str()));
}

TEST(InterfaceAPI, ContainsRunFunctionWithWorkspacePoolsAndDevices) {
  std::stringstream run_function;

  run_function << "/*!\n"
               << " * \\brief entrypoint function for TVM module \"ultimate_cat_spotter\"\n"
               << " * \\param inputs Input tensors for the module \n"
               << " * \\param outputs Output tensors for the module \n"
               << " * \\param workspace_pools Workspace memory pool pointers for the module \n"
               << " * \\param devices Device context pointers for the module \n"
               << " */\n"
               << "int32_t tvmgen_ultimate_cat_spotter_run(\n"
               << "  struct tvmgen_ultimate_cat_spotter_inputs* inputs,\n"
               << "  struct tvmgen_ultimate_cat_spotter_outputs* outputs,\n"
               << "  struct tvmgen_ultimate_cat_spotter_workspace_pools* workspace_pools,\n"
               << "  struct tvmgen_ultimate_cat_spotter_devices* devices\n"
               << ");\n";

  PoolInfo pool_info = WorkspacePoolInfo("my_memory_pool", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info =
      tir::usmp::AllocatedPoolInfo(pool_info, 100000);
  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"},
                                                 {allocated_pool_info}, {}, {"device"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(run_function.str()));
}

TEST(InterfaceAPI, ContainsRunFunctionWithWorkspaceIO) {
  std::stringstream run_function_with_map_functions;

  run_function_with_map_functions
      << "/*!\n"
      << " * \\brief Maps I/O inside the workspace pools for TVM module \"ultimate_cat_spotter\"\n"
      << " * \\param workspace_pools Workspace memory pool struct for the module \n"
      << " * \\return I/O tensor struct for the module \n"
      << " */\n"
      << "struct tvmgen_ultimate_cat_spotter_inputs tvmgen_ultimate_cat_spotter_map_inputs(\n"
      << "  struct tvmgen_ultimate_cat_spotter_workspace_pools* workspace_pools\n"
      << ");\n"
      << "\n"
      << "/*!\n"
      << " * \\brief Maps I/O inside the workspace pools for TVM module \"ultimate_cat_spotter\"\n"
      << " * \\param workspace_pools Workspace memory pool struct for the module \n"
      << " * \\return I/O tensor struct for the module \n"
      << " */\n"
      << "struct tvmgen_ultimate_cat_spotter_outputs tvmgen_ultimate_cat_spotter_map_outputs(\n"
      << "  struct tvmgen_ultimate_cat_spotter_workspace_pools* workspace_pools\n"
      << ");\n"
      << "\n"
      << "/*!\n"
      << " * \\brief entrypoint function for TVM module \"ultimate_cat_spotter\"\n"
      << " * \\param workspace_pools Workspace memory pool pointers for the module \n"
      << " */\n"
      << "int32_t tvmgen_ultimate_cat_spotter_run(\n"
      << "  struct tvmgen_ultimate_cat_spotter_workspace_pools* workspace_pools\n"
      << ");\n";

  PoolInfo pool_info = WorkspacePoolInfo("my_memory_pool", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info =
      tir::usmp::AllocatedPoolInfo(pool_info, 100000);
  tir::usmp::PoolAllocation pool_allocation_input{pool_info, 1000};
  tir::usmp::PoolAllocation pool_allocation_output{pool_info, 2000};
  runtime::Module test_module = InterfaceCCreate(
      "ultimate_cat_spotter", {"input"}, {"output"}, {allocated_pool_info},
      {{"input", pool_allocation_input}, {"output", pool_allocation_output}}, {}, 0);
  std::string header_source = test_module->GetSource();
  std::cout << header_source << "\n";
  ASSERT_THAT(header_source, HasSubstr(run_function_with_map_functions.str()));
}

TEST(InterfaceAPI, ContainsInputStructSingle) {
  std::stringstream input_struct;

  input_struct << "/*!\n"
               << " * \\brief Input tensor pointers for TVM module \"ultimate_cat_spotter\" \n"
               << " */\n"
               << "struct tvmgen_ultimate_cat_spotter_inputs {\n"
               << "  void* input;\n"
               << "};\n\n";

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {}, {}, {}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(input_struct.str()));
}

TEST(InterfaceAPI, ContainsInputStructMany) {
  std::stringstream input_struct;

  input_struct << "struct tvmgen_ultimate_cat_spotter_inputs {\n"
               << "  void* input1;\n"
               << "  void* input2;\n"
               << "};\n\n";

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input1", "input2"}, {"output"}, {}, {}, {}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(input_struct.str()));
}

TEST(InterfaceAPI, ContainsInputStructSanitised) {
  std::stringstream input_struct;

  input_struct << "struct tvmgen_ultimate_cat_spotter_inputs {\n"
               << "  void* input_1;\n"
               << "  void* input_2;\n"
               << "};\n\n";

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input+1", "input+2"}, {"output"}, {}, {}, {}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(input_struct.str()));
}

TEST(InterfaceAPI, ContainsInputStructClash) {
  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input+", "input-"}, {"output"}, {}, {}, {}, 0);
  ASSERT_THROW(test_module->GetSource(), InternalError);
}

TEST(InterfaceAPI, ContainsOutputStructSingle) {
  std::stringstream output_struct;

  output_struct << "/*!\n"
                << " * \\brief Output tensor pointers for TVM module \"ultimate_cat_spotter\" \n"
                << " */\n"
                << "struct tvmgen_ultimate_cat_spotter_outputs {\n"
                << "  void* output;\n"
                << "};\n\n";

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {}, {}, {}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(output_struct.str()));
}

TEST(InterfaceAPI, ContainsOutputStructMany) {
  std::stringstream output_struct;

  output_struct << "struct tvmgen_ultimate_cat_spotter_outputs {\n"
                << "  void* output1;\n"
                << "  void* output2;\n"
                << "};\n\n";

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output1", "output2"}, {}, {}, {}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(output_struct.str()));
}

TEST(InterfaceAPI, ContainsOutputStructSanitised) {
  std::stringstream output_struct;

  output_struct << "struct tvmgen_ultimate_cat_spotter_outputs {\n"
                << "  void* output_1;\n"
                << "  void* output_2;\n"
                << "};\n\n";

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output+1", "output-2"}, {}, {}, {}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(output_struct.str()));
}

TEST(InterfaceAPI, ContainsOutputStructClash) {
  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output+", "output-"}, {}, {}, {}, 0);
  ASSERT_THROW(test_module->GetSource(), InternalError);
}

TEST(InterfaceAPI, NoDeviceAPIStructIfNoDevices) {
  std::stringstream device_struct;

  device_struct << "/*!\n"
                << " * \\brief Device context pointers for TVM module \"ultimate_cat_spotter\" \n"
                << " */\n"
                << "struct tvmgen_ultimate_cat_spotter_devices {\n"
                << "};\n\n";

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {}, {}, {}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, Not(HasSubstr(device_struct.str())));
}

TEST(InterfaceAPI, ContainsDeviceStructSingle) {
  std::stringstream device_struct;

  device_struct << "/*!\n"
                << " * \\brief Device context pointers for TVM module \"ultimate_cat_spotter\" \n"
                << " */\n"
                << "struct tvmgen_ultimate_cat_spotter_devices {\n"
                << "  void* device;\n"
                << "};\n\n";

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {}, {}, {"device"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(device_struct.str()));
}

TEST(InterfaceAPI, ContainsDeviceStructMany) {
  std::stringstream device_struct;

  device_struct << "struct tvmgen_ultimate_cat_spotter_devices {\n"
                << "  void* device1;\n"
                << "  void* device2;\n"
                << "};\n\n";

  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {},
                                                 {}, {"device1", "device2"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(device_struct.str()));
}

TEST(InterfaceAPI, ContainsDeviceStructSanitised) {
  std::stringstream device_struct;

  device_struct << "struct tvmgen_ultimate_cat_spotter_devices {\n"
                << "  void* device_1;\n"
                << "  void* device_2;\n"
                << "};\n\n";

  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {},
                                                 {}, {"device+1", "device+2"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(device_struct.str()));
}

TEST(InterfaceAPI, ContainsDeviceStructClash) {
  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {},
                                                 {}, {"device+", "device-"}, 0);
  ASSERT_THROW(test_module->GetSource(), InternalError);
}

TEST(InterfaceAPI, ContainsWorkspaceSize) {
  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {}, {}, {}, 765432);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source,
              HasSubstr("* \\brief Workspace size for TVM module \"ultimate_cat_spotter\""));

  ASSERT_THAT(header_source,
              HasSubstr("#define TVMGEN_ULTIMATE_CAT_SPOTTER_WORKSPACE_SIZE 765432"));
}

TEST(InterfaceAPI, ContainsWorkspacePoolStructSingle) {
  PoolInfo pool_info = WorkspacePoolInfo("my_memory_pool", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info =
      tir::usmp::AllocatedPoolInfo(pool_info, 100000);

  std::stringstream workspace_struct;

  workspace_struct
      << "/*!\n"
      << " * \\brief Workspace pool pointers for TVM module \"ultimate_cat_spotter\" \n"
      << " */\n"
      << "struct tvmgen_ultimate_cat_spotter_workspace_pools {\n"
      << "  void* my_memory_pool;\n"
      << "};\n\n";

  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"},
                                                 {allocated_pool_info}, {}, {}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(workspace_struct.str()));

  ASSERT_THAT(header_source,
              HasSubstr("* \\brief my_memory_pool size for TVM module \"ultimate_cat_spotter\""));

  ASSERT_THAT(
      header_source,
      HasSubstr("#define TVMGEN_ULTIMATE_CAT_SPOTTER_MY_MEMORY_POOL_WORKSPACE_POOL_SIZE 100000"));
}

TEST(InterfaceAPI, ContainsWorkspacePoolStructMany) {
  PoolInfo pool_info1 = WorkspacePoolInfo("my_memory_pool_1", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info1 =
      tir::usmp::AllocatedPoolInfo(pool_info1, 100000);
  PoolInfo pool_info2 = WorkspacePoolInfo("my_memory_pool_2", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info2 =
      tir::usmp::AllocatedPoolInfo(pool_info2, 200000);

  std::stringstream workspace_struct;

  workspace_struct
      << "/*!\n"
      << " * \\brief Workspace pool pointers for TVM module \"ultimate_cat_spotter\" \n"
      << " */\n"
      << "struct tvmgen_ultimate_cat_spotter_workspace_pools {\n"
      << "  void* my_memory_pool_1;\n"
      << "  void* my_memory_pool_2;\n"
      << "};\n\n";

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"},
                       {allocated_pool_info1, allocated_pool_info2}, {}, {}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(workspace_struct.str()));

  ASSERT_THAT(header_source,
              HasSubstr("* \\brief my_memory_pool_1 size for TVM module \"ultimate_cat_spotter\""));

  ASSERT_THAT(
      header_source,
      HasSubstr("#define TVMGEN_ULTIMATE_CAT_SPOTTER_MY_MEMORY_POOL_1_WORKSPACE_POOL_SIZE 100000"));

  ASSERT_THAT(header_source,
              HasSubstr("* \\brief my_memory_pool_2 size for TVM module \"ultimate_cat_spotter\""));

  ASSERT_THAT(
      header_source,
      HasSubstr("#define TVMGEN_ULTIMATE_CAT_SPOTTER_MY_MEMORY_POOL_2_WORKSPACE_POOL_SIZE 200000"));
}

TEST(InterfaceAPI, ContainsWorkspacePoolStructSanitized) {
  PoolInfo pool_info = WorkspacePoolInfo("my_memory_pool+1", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info =
      tir::usmp::AllocatedPoolInfo(pool_info, 100000);

  std::stringstream workspace_struct;

  workspace_struct
      << "/*!\n"
      << " * \\brief Workspace pool pointers for TVM module \"ultimate_cat_spotter\" \n"
      << " */\n"
      << "struct tvmgen_ultimate_cat_spotter_workspace_pools {\n"
      << "  void* my_memory_pool_1;\n"
      << "};\n\n";

  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"},
                                                 {allocated_pool_info}, {}, {}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(workspace_struct.str()));

  ASSERT_THAT(header_source,
              HasSubstr("* \\brief my_memory_pool_1 size for TVM module \"ultimate_cat_spotter\""));

  ASSERT_THAT(
      header_source,
      HasSubstr("#define TVMGEN_ULTIMATE_CAT_SPOTTER_MY_MEMORY_POOL_1_WORKSPACE_POOL_SIZE 100000"));
}

TEST(InterfaceAPI, ContainsWorkspacePoolStructClash) {
  PoolInfo pool_info1 = WorkspacePoolInfo("my_memory_pool+", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info1 =
      tir::usmp::AllocatedPoolInfo(pool_info1, 100000);
  PoolInfo pool_info2 = WorkspacePoolInfo("my_memory_pool-", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info2 =
      tir::usmp::AllocatedPoolInfo(pool_info2, 200000);

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"},
                       {allocated_pool_info1, allocated_pool_info2}, {}, {}, 0);
  ASSERT_THROW(test_module->GetSource(), InternalError);
}

}  // namespace
}  // namespace codegen
}  // namespace tvm
