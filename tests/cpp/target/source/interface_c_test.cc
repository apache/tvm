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

using ::testing::ContainsRegex;
using ::testing::HasSubstr;

namespace tvm {
namespace codegen {

runtime::Module InterfaceCCreate(std::string module_name, Array<String> inputs,
                                 Array<String> outputs, Array<tir::usmp::AllocatedPoolInfo> pools,
                                 Map<String, tir::usmp::PoolAllocation> io_pool_allocations,
                                 Array<String> devices, int workspace_size,
                                 Map<String, IntImm> input_sizes, Map<String, IntImm> output_sizes);

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

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));

  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {},
                                                 {}, {}, 0, input_sizes, output_sizes);
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

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));

  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {},
                                                 {}, {}, 0, input_sizes, output_sizes);
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

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));

  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {},
                                                 {}, {"device"}, 0, input_sizes, output_sizes);
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

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));

  PoolInfo pool_info = WorkspacePoolInfo("my_memory_pool", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info =
      tir::usmp::AllocatedPoolInfo(pool_info, 100000);
  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {allocated_pool_info}, {}, {},
                       0, input_sizes, output_sizes);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(run_function.str()));
}

TEST(InterfaceAPI, ContainsRunFunctionWithWorkspaceAndConstantPools) {
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

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));

  PoolInfo pool_info = WorkspacePoolInfo("my_memory_pool", {});
  PoolInfo const_info = ConstantPoolInfo(
      "my_constant_pool", {},
      {{"const1", 0, runtime::NDArray::Empty({1}, DataType::Int(32), {kDLCPU, 0})},
       {"const2", 16, runtime::NDArray::Empty({1}, DataType::Float(64), {kDLCPU, 0})}});
  tir::usmp::AllocatedPoolInfo allocated_pool_info =
      tir::usmp::AllocatedPoolInfo(pool_info, 100000);
  tir::usmp::AllocatedPoolInfo allocated_const_info =
      tir::usmp::AllocatedPoolInfo(const_info, 100000);
  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"},
                                                 {allocated_pool_info, allocated_const_info}, {},
                                                 {}, 0, input_sizes, output_sizes);
  std::string header_source = test_module->GetSource();
  ASSERT_THAT(header_source, HasSubstr(run_function.str()));
  ASSERT_THAT(
      header_source,
      HasSubstr("#define TVMGEN_ULTIMATE_CAT_SPOTTER_MY_CONSTANT_POOL_CONSTANT_POOL_SIZE 24"));
  ASSERT_THAT(
      header_source,
      ContainsRegex(
          "#define TVMGEN_ULTIMATE_CAT_SPOTTER_MY_CONSTANT_POOL_CONSTANT_POOL_DATA \\\\\\\n    "
          "0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w, "
          "0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w, "
          "0x\\w\\w, 0x\\w\\w, 0x\\w\\w, \\\\\\\n    0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w, "
          "0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w\\\\\\\n"));
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

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));

  PoolInfo pool_info = WorkspacePoolInfo("my_memory_pool", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info =
      tir::usmp::AllocatedPoolInfo(pool_info, 100000);
  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {allocated_pool_info}, {},
                       {"device"}, 0, input_sizes, output_sizes);
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

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));

  PoolInfo pool_info = WorkspacePoolInfo("my_memory_pool", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info =
      tir::usmp::AllocatedPoolInfo(pool_info, 100000);
  tir::usmp::PoolAllocation pool_allocation_input{pool_info, 1000};
  tir::usmp::PoolAllocation pool_allocation_output{pool_info, 2000};
  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {allocated_pool_info},
                       {{"input", pool_allocation_input}, {"output", pool_allocation_output}}, {},
                       0, input_sizes, output_sizes);
  std::string header_source = test_module->GetSource();
  std::cout << header_source << "\n";
  ASSERT_THAT(header_source, HasSubstr(run_function_with_map_functions.str()));
}

TEST(InterfaceAPI, ContainsInputStructSingle) {
  std::stringstream input_struct;
  std::stringstream input_size_macro;

  input_size_macro
      << "/*!\n"
      << " * \\brief Input tensor input size (in bytes) for TVM module \"ultimate_cat_spotter\" \n"
      << " */\n"
      << "#define TVMGEN_ULTIMATE_CAT_SPOTTER_INPUT_SIZE 537\n";

  input_struct << "/*!\n"
               << " * \\brief Input tensor pointers for TVM module \"ultimate_cat_spotter\" \n"
               << " */\n"
               << "struct tvmgen_ultimate_cat_spotter_inputs {\n"
               << "  void* input;\n"
               << "};\n\n";

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 537));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));

  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {},
                                                 {}, {}, 0, input_sizes, output_sizes);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(input_struct.str()));

  ASSERT_THAT(header_source, HasSubstr(input_size_macro.str()));
}

TEST(InterfaceAPI, ContainsInputStructMany) {
  std::stringstream input_struct;
  std::stringstream input1_size_macro;
  std::stringstream input2_size_macro;

  input1_size_macro
      << "/*!\n"
      << " * \\brief Input tensor input1 size (in bytes) for TVM module \"ultimate_cat_spotter\" \n"
      << " */\n"
      << "#define TVMGEN_ULTIMATE_CAT_SPOTTER_INPUT1_SIZE 765\n";

  input2_size_macro
      << "/*!\n"
      << " * \\brief Input tensor input2 size (in bytes) for TVM module \"ultimate_cat_spotter\" \n"
      << " */\n"
      << "#define TVMGEN_ULTIMATE_CAT_SPOTTER_INPUT2_SIZE 127\n";

  input_struct << "struct tvmgen_ultimate_cat_spotter_inputs {\n"
               << "  void* input1;\n"
               << "  void* input2;\n"
               << "};\n\n";

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input1", IntImm(DataType::Int(32), 765));
  input_sizes.Set("input2", IntImm(DataType::Int(32), 127));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input1", "input2"}, {"output"}, {}, {}, {}, 0,
                       input_sizes, output_sizes);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(input_struct.str()));
  ASSERT_THAT(header_source, HasSubstr(input1_size_macro.str()));
  ASSERT_THAT(header_source, HasSubstr(input2_size_macro.str()));
}

TEST(InterfaceAPI, ContainsInputStructSanitised) {
  std::stringstream input_struct;
  std::stringstream input1_size_macro;
  std::stringstream input2_size_macro;

  input1_size_macro << "/*!\n"
                    << " * \\brief Input tensor input_1 size (in bytes) for TVM module "
                       "\"ultimate_cat_spotter\" \n"
                    << " */\n"
                    << "#define TVMGEN_ULTIMATE_CAT_SPOTTER_INPUT_1_SIZE 765\n";

  input2_size_macro << "/*!\n"
                    << " * \\brief Input tensor input_2 size (in bytes) for TVM module "
                       "\"ultimate_cat_spotter\" \n"
                    << " */\n"
                    << "#define TVMGEN_ULTIMATE_CAT_SPOTTER_INPUT_2_SIZE 127\n";

  input_struct << "struct tvmgen_ultimate_cat_spotter_inputs {\n"
               << "  void* input_1;\n"
               << "  void* input_2;\n"
               << "};\n\n";

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input+1", IntImm(DataType::Int(32), 765));
  input_sizes.Set("input+2", IntImm(DataType::Int(32), 127));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input+1", "input+2"}, {"output"}, {}, {}, {}, 0,
                       input_sizes, output_sizes);
  std::string header_source = test_module->GetSource();

  std::cout << header_source << std::endl;

  ASSERT_THAT(header_source, HasSubstr(input_struct.str()));
  ASSERT_THAT(header_source, HasSubstr(input1_size_macro.str()));
  ASSERT_THAT(header_source, HasSubstr(input2_size_macro.str()));
}

TEST(InterfaceAPI, ContainsInputStructClash) {
  Map<String, IntImm> input_sizes;
  input_sizes.Set("input+", IntImm(DataType::Int(32), 0));
  input_sizes.Set("input-", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input+", "input-"}, {"output"}, {}, {}, {}, 0,
                       input_sizes, output_sizes);
  ASSERT_THROW(test_module->GetSource(), InternalError);
}

TEST(InterfaceAPI, ContainsOutputStructSingle) {
  std::stringstream output_struct;
  std::stringstream output_size_macro;

  output_size_macro << "/*!\n"
                    << " * \\brief Output tensor output size (in bytes) for TVM module "
                       "\"ultimate_cat_spotter\" \n"
                    << " */\n"
                    << "#define TVMGEN_ULTIMATE_CAT_SPOTTER_OUTPUT_SIZE 543\n";

  output_struct << "/*!\n"
                << " * \\brief Output tensor pointers for TVM module \"ultimate_cat_spotter\" \n"
                << " */\n"
                << "struct tvmgen_ultimate_cat_spotter_outputs {\n"
                << "  void* output;\n"
                << "};\n\n";

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 543));

  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {},
                                                 {}, {}, 0, input_sizes, output_sizes);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(output_struct.str()));
  ASSERT_THAT(header_source, HasSubstr(output_size_macro.str()));
}

TEST(InterfaceAPI, ContainsOutputStructMany) {
  std::stringstream output_struct;
  std::stringstream output1_size_macro;
  std::stringstream output2_size_macro;

  output1_size_macro << "/*!\n"
                     << " * \\brief Output tensor output1 size (in bytes) for TVM module "
                        "\"ultimate_cat_spotter\" \n"
                     << " */\n"
                     << "#define TVMGEN_ULTIMATE_CAT_SPOTTER_OUTPUT1_SIZE 345\n";

  output2_size_macro << "/*!\n"
                     << " * \\brief Output tensor output2 size (in bytes) for TVM module "
                        "\"ultimate_cat_spotter\" \n"
                     << " */\n"
                     << "#define TVMGEN_ULTIMATE_CAT_SPOTTER_OUTPUT2_SIZE 984\n";

  output_struct << "struct tvmgen_ultimate_cat_spotter_outputs {\n"
                << "  void* output1;\n"
                << "  void* output2;\n"
                << "};\n\n";

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output1", IntImm(DataType::Int(32), 345));
  output_sizes.Set("output2", IntImm(DataType::Int(32), 984));

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output1", "output2"}, {}, {}, {}, 0,
                       input_sizes, output_sizes);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(output_struct.str()));
  ASSERT_THAT(header_source, HasSubstr(output1_size_macro.str()));
  ASSERT_THAT(header_source, HasSubstr(output2_size_macro.str()));
}

TEST(InterfaceAPI, ContainsOutputStructSanitised) {
  std::stringstream output_struct;
  std::stringstream output1_size_macro;
  std::stringstream output2_size_macro;

  output1_size_macro << "/*!\n"
                     << " * \\brief Output tensor output_1 size (in bytes) for TVM module "
                        "\"ultimate_cat_spotter\" \n"
                     << " */\n"
                     << "#define TVMGEN_ULTIMATE_CAT_SPOTTER_OUTPUT_1_SIZE 345\n";

  output2_size_macro << "/*!\n"
                     << " * \\brief Output tensor output_2 size (in bytes) for TVM module "
                        "\"ultimate_cat_spotter\" \n"
                     << " */\n"
                     << "#define TVMGEN_ULTIMATE_CAT_SPOTTER_OUTPUT_2_SIZE 984\n";

  output_struct << "struct tvmgen_ultimate_cat_spotter_outputs {\n"
                << "  void* output_1;\n"
                << "  void* output_2;\n"
                << "};\n\n";

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output+1", IntImm(DataType::Int(32), 345));
  output_sizes.Set("output-2", IntImm(DataType::Int(32), 984));

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output+1", "output-2"}, {}, {}, {}, 0,
                       input_sizes, output_sizes);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(output_struct.str()));
  ASSERT_THAT(header_source, HasSubstr(output1_size_macro.str()));
  ASSERT_THAT(header_source, HasSubstr(output2_size_macro.str()));
}

TEST(InterfaceAPI, ContainsOutputStructClash) {
  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output+", IntImm(DataType::Int(32), 0));
  output_sizes.Set("output-", IntImm(DataType::Int(32), 0));
  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output+", "output-"}, {}, {}, {}, 0,
                       input_sizes, output_sizes);
  ASSERT_THROW(test_module->GetSource(), InternalError);
}

TEST(InterfaceAPI, NoDeviceAPIStructIfNoDevices) {
  std::stringstream device_struct;

  device_struct << "/*!\n"
                << " * \\brief Device context pointers for TVM module \"ultimate_cat_spotter\" \n"
                << " */\n"
                << "struct tvmgen_ultimate_cat_spotter_devices {\n"
                << "};\n\n";

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));
  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {},
                                                 {}, {}, 0, input_sizes, output_sizes);
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

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));
  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {},
                                                 {}, {"device"}, 0, input_sizes, output_sizes);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(device_struct.str()));
}

TEST(InterfaceAPI, ContainsDeviceStructMany) {
  std::stringstream device_struct;

  device_struct << "struct tvmgen_ultimate_cat_spotter_devices {\n"
                << "  void* device1;\n"
                << "  void* device2;\n"
                << "};\n\n";

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));
  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {}, {},
                       {"device1", "device2"}, 0, input_sizes, output_sizes);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(device_struct.str()));
}

TEST(InterfaceAPI, ContainsDeviceStructSanitised) {
  std::stringstream device_struct;

  device_struct << "struct tvmgen_ultimate_cat_spotter_devices {\n"
                << "  void* device_1;\n"
                << "  void* device_2;\n"
                << "};\n\n";

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));
  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {}, {},
                       {"device+1", "device+2"}, 0, input_sizes, output_sizes);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(device_struct.str()));
}

TEST(InterfaceAPI, ContainsDeviceStructClash) {
  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));
  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {}, {},
                       {"device+", "device-"}, 0, input_sizes, output_sizes);
  ASSERT_THROW(test_module->GetSource(), InternalError);
}

TEST(InterfaceAPI, ContainsWorkspaceSize) {
  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));
  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {},
                                                 {}, {}, 765432, input_sizes, output_sizes);
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

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));
  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {allocated_pool_info}, {}, {},
                       0, input_sizes, output_sizes);
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

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));
  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"},
                                                 {allocated_pool_info1, allocated_pool_info2}, {},
                                                 {}, 0, input_sizes, output_sizes);
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

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));
  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {allocated_pool_info}, {}, {},
                       0, input_sizes, output_sizes);
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

  Map<String, IntImm> input_sizes;
  input_sizes.Set("input", IntImm(DataType::Int(32), 0));
  Map<String, IntImm> output_sizes;
  output_sizes.Set("output", IntImm(DataType::Int(32), 0));
  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"},
                                                 {allocated_pool_info1, allocated_pool_info2}, {},
                                                 {}, 0, input_sizes, output_sizes);
  ASSERT_THROW(test_module->GetSource(), InternalError);
}

}  // namespace
}  // namespace codegen
}  // namespace tvm
