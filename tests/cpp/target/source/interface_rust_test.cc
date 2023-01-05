
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

runtime::Module InterfaceRustCreate(std::string module_name,
                                    Map<String, Map<String, ObjectRef>> inputs,
                                    Map<String, Map<String, ObjectRef>> outputs,
                                    Array<tir::usmp::AllocatedPoolInfo> pools,
                                    Map<String, tir::usmp::PoolAllocation> io_pool_allocations,
                                    Array<String> devices, Array<String> input_names,
                                    Array<String> output_names, int workspace_size);

namespace {

Map<String, ObjectRef> TestIO(String dtype, Integer size) {
  return {{"dtype", dtype}, {"size", size}};
}

Map<String, ObjectRef> TestIO() { return TestIO("uint8", 100); }

TEST(RustInterfaceAPI, ContainsRunFunction) {
  std::stringstream run_function;

  run_function << "/// Entrypoint function for TVM module \"ultimate_cat_spotter\"\n"
               << "/// # Arguments\n"
               << "/// * `inputs` - Input tensors for the module\n"
               << "/// * `outputs` - Output tensors for the module\n"
               << "pub fn run(\n"
               << "    inputs: &mut Inputs,\n"
               << "    outputs: &mut Outputs,\n"
               << ") -> Result<(), ()> {\n"
               << "    unsafe {\n"
               << "        let ret = tvmgen_ultimate_cat_spotter_run(\n"
               << "            inputs,\n"
               << "            outputs,\n"
               << "        );\n"
               << "        if ret == 0 {\n"
               << "            Ok(())\n"
               << "        } else {\n"
               << "            Err(())\n"
               << "        }\n"
               << "    }\n"
               << "}\n"
               << "\n"
               << "extern \"C\" {\n"
               << "    pub fn tvmgen_ultimate_cat_spotter_run(\n"
               << "        inputs: *mut Inputs,\n"
               << "        outputs: *mut Outputs,\n"
               << "    ) -> i32;\n"
               << "}\n";

  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}}, {},
                          {}, {}, {"input"}, {"output"}, 0);
  std::string header_source = test_module->GetSource();
  ASSERT_THAT(header_source, HasSubstr(run_function.str()));
}

TEST(RustInterfaceAPI, ContainsRunFunctionWithDevices) {
  std::stringstream run_function;

  run_function << "/// Entrypoint function for TVM module \"ultimate_cat_spotter\"\n"
               << "/// # Arguments\n"
               << "/// * `inputs` - Input tensors for the module\n"
               << "/// * `outputs` - Output tensors for the module\n"
               << "/// * `devices` - Device context pointers for the module\n"
               << "pub fn run(\n"
               << "    inputs: &mut Inputs,\n"
               << "    outputs: &mut Outputs,\n"
               << "    devices: &mut Devices,\n"
               << ") -> Result<(), ()> {\n"
               << "    unsafe {\n"
               << "        let ret = tvmgen_ultimate_cat_spotter_run(\n"
               << "            inputs,\n"
               << "            outputs,\n"
               << "            devices,\n"
               << "        );\n"
               << "        if ret == 0 {\n"
               << "            Ok(())\n"
               << "        } else {\n"
               << "            Err(())\n"
               << "        }\n"
               << "    }\n"
               << "}\n"
               << "\n"
               << "extern \"C\" {\n"
               << "    pub fn tvmgen_ultimate_cat_spotter_run(\n"
               << "        inputs: *mut Inputs,\n"
               << "        outputs: *mut Outputs,\n"
               << "        devices: *mut Devices,\n"
               << "    ) -> i32;\n"
               << "}\n";

  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}}, {},
                          {}, {"device"}, {"input"}, {"output"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(run_function.str()));
}

TEST(RustInterfaceAPI, ContainsRunFunctionWithWorkspacePools) {
  std::stringstream run_function;

  run_function << "/// Entrypoint function for TVM module \"ultimate_cat_spotter\"\n"
               << "/// # Arguments\n"
               << "/// * `inputs` - Input tensors for the module\n"
               << "/// * `outputs` - Output tensors for the module\n"
               << "/// * `workspace_pools` - Workspace memory pools for the module\n"
               << "pub fn run(\n"
               << "    inputs: &mut Inputs,\n"
               << "    outputs: &mut Outputs,\n"
               << "    workspace_pools: &mut WorkspacePools,\n"
               << ") -> Result<(), ()> {\n"
               << "    unsafe {\n"
               << "        let ret = tvmgen_ultimate_cat_spotter_run(\n"
               << "            inputs,\n"
               << "            outputs,\n"
               << "            workspace_pools,\n"
               << "        );\n"
               << "        if ret == 0 {\n"
               << "            Ok(())\n"
               << "        } else {\n"
               << "            Err(())\n"
               << "        }\n"
               << "    }\n"
               << "}\n"
               << "\n"
               << "extern \"C\" {\n"
               << "    pub fn tvmgen_ultimate_cat_spotter_run(\n"
               << "        inputs: *mut Inputs,\n"
               << "        outputs: *mut Outputs,\n"
               << "        workspace_pools: *mut WorkspacePools,\n"
               << "    ) -> i32;\n"
               << "}\n";

  PoolInfo pool_info = WorkspacePoolInfo("my_memory_pool", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info =
      tir::usmp::AllocatedPoolInfo(pool_info, 100000);
  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}},
                          {allocated_pool_info}, {}, {}, {"input"}, {"output"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(run_function.str()));
}

TEST(RustInterfaceAPI, ContainsRunFunctionWithWorkspaceAndConstantPools) {
  std::stringstream run_function;

  run_function << "/// Entrypoint function for TVM module \"ultimate_cat_spotter\"\n"
               << "/// # Arguments\n"
               << "/// * `inputs` - Input tensors for the module\n"
               << "/// * `outputs` - Output tensors for the module\n"
               << "/// * `workspace_pools` - Workspace memory pools for the module\n"
               << "pub fn run(\n"
               << "    inputs: &mut Inputs,\n"
               << "    outputs: &mut Outputs,\n"
               << "    workspace_pools: &mut WorkspacePools,\n"
               << ") -> Result<(), ()> {\n";

  PoolInfo pool_info = WorkspacePoolInfo("my_memory_pool", {});
  PoolInfo const_info = ConstantPoolInfo(
      "my_constant_pool", {},
      {{"const1", 0, runtime::NDArray::Empty({1}, DataType::Int(32), {kDLCPU, 0})},
       {"const2", 16, runtime::NDArray::Empty({1}, DataType::Float(64), {kDLCPU, 0})}});
  tir::usmp::AllocatedPoolInfo allocated_pool_info =
      tir::usmp::AllocatedPoolInfo(pool_info, 100000);
  tir::usmp::AllocatedPoolInfo allocated_const_info =
      tir::usmp::AllocatedPoolInfo(const_info, 100000);
  runtime::Module test_module = InterfaceRustCreate(
      "ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}},
      {allocated_pool_info, allocated_const_info}, {}, {}, {"input"}, {"output"}, 0);
  std::string header_source = test_module->GetSource();
  ASSERT_THAT(header_source, HasSubstr(run_function.str()));
  ASSERT_THAT(header_source, HasSubstr("pub const MY_CONSTANT_POOL_CONSTANT_POOL_SIZE: u32 = 24;"));
  ASSERT_THAT(
      header_source,
      ContainsRegex(
          "#\\[macro_export\\]\\\n"
          "macro_rules! my_constant_pool_constant_pool_data \\{\\\n"
          "    \\(\\) => \\{\\[\\\n"
          "        0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w, "
          "0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w, "
          "0x\\w\\w, 0x\\w\\w, 0x\\w\\w, \\\\\\\n        0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w, "
          "0x\\w\\w, 0x\\w\\w, 0x\\w\\w, 0x\\w\\w\\\\\\\n"
          "    \\]\\};\\\n"
          "}\\\n"));
}

TEST(RustInterfaceAPI, ContainsRunFunctionWithWorkspacePoolsAndDevices) {
  std::stringstream run_function;

  run_function << "/// Entrypoint function for TVM module \"ultimate_cat_spotter\"\n"
               << "/// # Arguments\n"
               << "/// * `inputs` - Input tensors for the module\n"
               << "/// * `outputs` - Output tensors for the module\n"
               << "/// * `workspace_pools` - Workspace memory pools for the module\n"
               << "/// * `devices` - Device context pointers for the module\n"
               << "pub fn run(\n"
               << "    inputs: &mut Inputs,\n"
               << "    outputs: &mut Outputs,\n"
               << "    workspace_pools: &mut WorkspacePools,\n"
               << "    devices: &mut Devices,\n"
               << ") -> Result<(), ()> {\n"
               << "    unsafe {\n"
               << "        let ret = tvmgen_ultimate_cat_spotter_run(\n"
               << "            inputs,\n"
               << "            outputs,\n"
               << "            workspace_pools,\n"
               << "            devices,\n"
               << "        );\n"
               << "        if ret == 0 {\n"
               << "            Ok(())\n"
               << "        } else {\n"
               << "            Err(())\n"
               << "        }\n"
               << "    }\n"
               << "}\n"
               << "\n"
               << "extern \"C\" {\n"
               << "    pub fn tvmgen_ultimate_cat_spotter_run(\n"
               << "        inputs: *mut Inputs,\n"
               << "        outputs: *mut Outputs,\n"
               << "        workspace_pools: *mut WorkspacePools,\n"
               << "        devices: *mut Devices,\n"
               << "    ) -> i32;\n"
               << "}\n";

  PoolInfo pool_info = WorkspacePoolInfo("my_memory_pool", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info =
      tir::usmp::AllocatedPoolInfo(pool_info, 100000);
  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}},
                          {allocated_pool_info}, {}, {"device"}, {"input"}, {"output"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(run_function.str()));
}

TEST(RustInterfaceAPI, ContainsRunFunctionWithWorkspaceIO_Unsupported) {
  std::stringstream run_function_with_map_functions;
  PoolInfo pool_info = WorkspacePoolInfo("my_memory_pool", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info =
      tir::usmp::AllocatedPoolInfo(pool_info, 100000);
  tir::usmp::PoolAllocation pool_allocation_input{pool_info, 1000};
  tir::usmp::PoolAllocation pool_allocation_output{pool_info, 2000};

  ASSERT_THROW(
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}},
                          {allocated_pool_info},
                          {{"input", pool_allocation_input}, {"output", pool_allocation_output}},
                          {}, {"input"}, {"output"}, 0),
      InternalError);
}

TEST(RustInterfaceAPI, ContainsInputStructSingle) {
  std::stringstream input_struct;

  input_struct << "/// Input tensors for TVM module \"ultimate_cat_spotter\"\n"
               << "#[repr(C)]\n"
               << "pub struct Inputs {\n"
               << "    input: *mut ::std::os::raw::c_void,\n"
               << "}\n"
               << "\n"
               << "impl Inputs {\n"
               << "    pub fn new <'a>(\n"
               << "        input: &mut [u8; 100],\n"
               << "    ) -> Self {\n"
               << "        Self {\n"
               << "            input: input.as_ptr() as *mut ::std::os::raw::c_void,\n"
               << "        }\n"
               << "    }\n"
               << "}\n";

  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}}, {},
                          {}, {}, {"input"}, {"output"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(input_struct.str()));
}

TEST(RustInterfaceAPI, ContainsInputStructMany) {
  std::stringstream input_struct;

  input_struct << "/// Input tensors for TVM module \"ultimate_cat_spotter\"\n"
               << "#[repr(C)]\n"
               << "pub struct Inputs {\n"
               << "    input1: *mut ::std::os::raw::c_void,\n"
               << "    input2: *mut ::std::os::raw::c_void,\n"
               << "    input3: *mut ::std::os::raw::c_void,\n"
               << "}\n"
               << "\n"
               << "impl Inputs {\n"
               << "    pub fn new <'a>(\n"
               << "        input1: &mut [f32; 15],\n"
               << "        input2: &mut [u8; 120],\n"
               << "        input3: &mut [i32; 45],\n"
               << "    ) -> Self {\n"
               << "        Self {\n"
               << "            input1: input1.as_ptr() as *mut ::std::os::raw::c_void,\n"
               << "            input2: input2.as_ptr() as *mut ::std::os::raw::c_void,\n"
               << "            input3: input3.as_ptr() as *mut ::std::os::raw::c_void,\n"
               << "        }\n"
               << "    }\n"
               << "}\n";

  runtime::Module test_module = InterfaceRustCreate("ultimate_cat_spotter",
                                                    {{"input1", TestIO("float32", 60)},
                                                     {"input2", TestIO("uint8", 120)},
                                                     {"input3", TestIO("int32", 180)}},
                                                    {{"output", TestIO()}}, {}, {}, {},
                                                    {"input1", "input2", "input3"}, {"output"}, 0);
  std::string header_source = test_module->GetSource();
  ASSERT_THAT(header_source, HasSubstr(input_struct.str()));
}

TEST(RustInterfaceAPI, ContainsInputStructSanitised) {
  std::stringstream input_struct;

  input_struct << "/// Input tensors for TVM module \"ultimate_cat_spotter\"\n"
               << "#[repr(C)]\n"
               << "pub struct Inputs {\n"
               << "    input_1: *mut ::std::os::raw::c_void,\n"
               << "    input_2: *mut ::std::os::raw::c_void,\n"
               << "}\n"
               << "\n"
               << "impl Inputs {\n"
               << "    pub fn new <'a>(\n"
               << "        input_1: &mut [u8; 100],\n"
               << "        input_2: &mut [u8; 100],\n"
               << "    ) -> Self {\n"
               << "        Self {\n"
               << "            input_1: input_1.as_ptr() as *mut ::std::os::raw::c_void,\n"
               << "            input_2: input_2.as_ptr() as *mut ::std::os::raw::c_void,\n"
               << "        }\n"
               << "    }\n"
               << "}\n";

  runtime::Module test_module = InterfaceRustCreate(
      "ultimate_cat_spotter", {{"input+1", TestIO()}, {"input+2", TestIO()}},
      {{"output", TestIO()}}, {}, {}, {}, {"input+1", "input+2"}, {"output"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(input_struct.str()));
}

TEST(RustInterfaceAPI, ContainsInputStructClash) {
  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input+", TestIO()}, {"input-", TestIO()}},
                          {{"output", TestIO()}}, {}, {}, {}, {"input+", "input-"}, {"output"}, 0);
  ASSERT_THROW(test_module->GetSource(), InternalError);
}

TEST(RustInterfaceAPI, ContainsOutputStructSingle) {
  std::stringstream output_struct;

  output_struct << "/// Output tensors for TVM module \"ultimate_cat_spotter\"\n"
                << "#[repr(C)]\n"
                << "pub struct Outputs {\n"
                << "    output: *mut ::std::os::raw::c_void,\n"
                << "}\n"
                << "\n"
                << "impl Outputs {\n"
                << "    pub fn new <'a>(\n"
                << "        output: &mut [u8; 100],\n"
                << "    ) -> Self {\n"
                << "        Self {\n"
                << "            output: output.as_ptr() as *mut ::std::os::raw::c_void,\n"
                << "        }\n"
                << "    }\n"
                << "}\n";

  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}}, {},
                          {}, {}, {"input"}, {"output"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(output_struct.str()));
}

TEST(RustInterfaceAPI, ContainsOutputStructMany) {
  std::stringstream output_struct;
  output_struct << "/// Output tensors for TVM module \"ultimate_cat_spotter\"\n"
                << "#[repr(C)]\n"
                << "pub struct Outputs {\n"
                << "    output1: *mut ::std::os::raw::c_void,\n"
                << "    output2: *mut ::std::os::raw::c_void,\n"
                << "    output3: *mut ::std::os::raw::c_void,\n"
                << "}\n"
                << "\n"
                << "impl Outputs {\n"
                << "    pub fn new <'a>(\n"
                << "        output1: &mut [i32; 25],\n"
                << "        output2: &mut [f32; 19],\n"
                << "        output3: &mut [u64; 2],\n"
                << "    ) -> Self {\n"
                << "        Self {\n"
                << "            output1: output1.as_ptr() as *mut ::std::os::raw::c_void,\n"
                << "            output2: output2.as_ptr() as *mut ::std::os::raw::c_void,\n"
                << "            output3: output3.as_ptr() as *mut ::std::os::raw::c_void,\n"
                << "        }\n"
                << "    }\n"
                << "}\n";

  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}},
                          {{"output1", TestIO("int32", 100)},
                           {"output2", TestIO("float32", 76)},
                           {"output3", TestIO("uint64", 16)}},
                          {}, {}, {}, {"input"}, {"output1", "output2", "output3"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(output_struct.str()));
}

TEST(RustInterfaceAPI, ContainsOutputStructSanitised) {
  std::stringstream output_struct;
  output_struct << "/// Output tensors for TVM module \"ultimate_cat_spotter\"\n"
                << "#[repr(C)]\n"
                << "pub struct Outputs {\n"
                << "    output_1: *mut ::std::os::raw::c_void,\n"
                << "    output_2: *mut ::std::os::raw::c_void,\n"
                << "}\n"
                << "\n"
                << "impl Outputs {\n"
                << "    pub fn new <'a>(\n"
                << "        output_1: &mut [u8; 100],\n"
                << "        output_2: &mut [u8; 100],\n"
                << "    ) -> Self {\n"
                << "        Self {\n"
                << "            output_1: output_1.as_ptr() as *mut ::std::os::raw::c_void,\n"
                << "            output_2: output_2.as_ptr() as *mut ::std::os::raw::c_void,\n"
                << "        }\n"
                << "    }\n"
                << "}\n";

  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}},
                          {{"output+1", TestIO()}, {"output-2", TestIO()}}, {}, {}, {}, {"input"},
                          {"output+1", "output-2"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(output_struct.str()));
}

TEST(RustInterfaceAPI, ContainsOutputStructClash) {
  runtime::Module test_module = InterfaceRustCreate(
      "ultimate_cat_spotter", {{"input", TestIO()}}, {{"output+", TestIO()}, {"output-", TestIO()}},
      {}, {}, {}, {"input"}, {"output+", "output-"}, 0);
  ASSERT_THROW(test_module->GetSource(), InternalError);
}

TEST(RustInterfaceAPI, NoDeviceAPIStructIfNoDevices) {
  std::stringstream device_struct;
  device_struct << "/// Device context pointers for TVM module \"ultimate_cat_spotter\"\n"
                << "#[repr(C)]\n"
                << "pub struct Devices {\n"
                << "    device: *mut ::std::os::raw::c_void,\n"
                << "}\n"
                << "\n";

  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}}, {},
                          {}, {}, {"input"}, {"output"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, Not(HasSubstr(device_struct.str())));
}

TEST(RustInterfaceAPI, ContainsDeviceStructSingle) {
  std::stringstream device_struct;

  device_struct << "/// Device context pointers for TVM module \"ultimate_cat_spotter\"\n"
                << "#[repr(C)]\n"
                << "pub struct Devices {\n"
                << "    device: *mut ::std::os::raw::c_void,\n"
                << "}\n"
                << "\n"
                << "impl Devices {\n"
                << "    pub fn new <'a>(\n"
                << "        device: *mut ::std::os::raw::c_void,\n"
                << "    ) -> Self {\n"
                << "        Self {\n"
                << "            device: device,\n"
                << "        }\n"
                << "    }\n"
                << "}\n";

  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}}, {},
                          {}, {"device"}, {"input"}, {"output"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(device_struct.str()));
}

TEST(RustInterfaceAPI, ContainsDeviceStructMany) {
  std::stringstream device_struct;

  device_struct << "/// Device context pointers for TVM module \"ultimate_cat_spotter\"\n"
                << "#[repr(C)]\n"
                << "pub struct Devices {\n"
                << "    device1: *mut ::std::os::raw::c_void,\n"
                << "    device2: *mut ::std::os::raw::c_void,\n"
                << "}\n"
                << "\n"
                << "impl Devices {\n"
                << "    pub fn new <'a>(\n"
                << "        device1: *mut ::std::os::raw::c_void,\n"
                << "        device2: *mut ::std::os::raw::c_void,\n"
                << "    ) -> Self {\n"
                << "        Self {\n"
                << "            device1: device1,\n"
                << "            device2: device2,\n"
                << "        }\n"
                << "    }\n"
                << "}\n";

  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}}, {},
                          {}, {"device1", "device2"}, {"input"}, {"output"}, 0);
  std::string header_source = test_module->GetSource();
  ASSERT_THAT(header_source, HasSubstr(device_struct.str()));
}

TEST(RustInterfaceAPI, ContainsDeviceStructSanitised) {
  std::stringstream device_struct;

  device_struct << "/// Device context pointers for TVM module \"ultimate_cat_spotter\"\n"
                << "#[repr(C)]\n"
                << "pub struct Devices {\n"
                << "    device_1: *mut ::std::os::raw::c_void,\n"
                << "    device_2: *mut ::std::os::raw::c_void,\n"
                << "}\n"
                << "\n"
                << "impl Devices {\n"
                << "    pub fn new <'a>(\n"
                << "        device_1: *mut ::std::os::raw::c_void,\n"
                << "        device_2: *mut ::std::os::raw::c_void,\n"
                << "    ) -> Self {\n"
                << "        Self {\n"
                << "            device_1: device_1,\n"
                << "            device_2: device_2,\n"
                << "        }\n"
                << "    }\n"
                << "}\n";

  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}}, {},
                          {}, {"device+1", "device+2"}, {"input"}, {"output"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(device_struct.str()));
}

TEST(RustInterfaceAPI, ContainsDeviceStructClash) {
  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}}, {},
                          {}, {"device+", "device-"}, {"input"}, {"output"}, 0);
  ASSERT_THROW(test_module->GetSource(), InternalError);
}

TEST(RustInterfaceAPI, ContainsWorkspaceSize) {
  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}}, {},
                          {}, {}, {"input"}, {"output"}, 765432);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source,
              HasSubstr("/// Workspace size for TVM module \"ultimate_cat_spotter\""));

  ASSERT_THAT(header_source, HasSubstr("pub const WORKSPACE_SIZE: usize = 765432;"));
}

TEST(RustInterfaceAPI, ContainsWorkspacePoolStructSingle) {
  PoolInfo pool_info = WorkspacePoolInfo("my_memory_pool", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info =
      tir::usmp::AllocatedPoolInfo(pool_info, 100000);

  std::stringstream workspace_struct;

  workspace_struct
      << "/// Workspace pools for TVM module \"ultimate_cat_spotter\"\n"
      << "#[repr(C)]\n"
      << "pub struct WorkspacePools {\n"
      << "    my_memory_pool: *mut ::std::os::raw::c_void,\n"
      << "}\n"
      << "\n"
      << "impl WorkspacePools {\n"
      << "    pub fn new <'a>(\n"
      << "        my_memory_pool: &mut [u8; 100000],\n"
      << "    ) -> Self {\n"
      << "        Self {\n"
      << "            my_memory_pool: my_memory_pool.as_ptr() as *mut ::std::os::raw::c_void,\n"
      << "        }\n"
      << "    }\n"
      << "}\n";

  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}},
                          {allocated_pool_info}, {}, {}, {"input"}, {"output"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(workspace_struct.str()));

  ASSERT_THAT(header_source,
              HasSubstr("/// my_memory_pool size for TVM module \"ultimate_cat_spotter\""));

  ASSERT_THAT(header_source,
              HasSubstr("pub const MY_MEMORY_POOL_WORKSPACE_POOL_SIZE: usize = 100000;"));
}

TEST(RustInterfaceAPI, ContainsWorkspacePoolStructMany) {
  PoolInfo pool_info1 = WorkspacePoolInfo("my_memory_pool_1", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info1 =
      tir::usmp::AllocatedPoolInfo(pool_info1, 100000);
  PoolInfo pool_info2 = WorkspacePoolInfo("my_memory_pool_2", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info2 =
      tir::usmp::AllocatedPoolInfo(pool_info2, 200000);

  std::stringstream workspace_struct;

  workspace_struct
      << "/// Workspace pools for TVM module \"ultimate_cat_spotter\"\n"
      << "#[repr(C)]\n"
      << "pub struct WorkspacePools {\n"
      << "    my_memory_pool_1: *mut ::std::os::raw::c_void,\n"
      << "    my_memory_pool_2: *mut ::std::os::raw::c_void,\n"
      << "}\n"
      << "\n"
      << "impl WorkspacePools {\n"
      << "    pub fn new <'a>(\n"
      << "        my_memory_pool_1: &mut [u8; 100000],\n"
      << "        my_memory_pool_2: &mut [u8; 200000],\n"
      << "    ) -> Self {\n"
      << "        Self {\n"
      << "            my_memory_pool_1: my_memory_pool_1.as_ptr() as *mut ::std::os::raw::c_void,\n"
      << "            my_memory_pool_2: my_memory_pool_2.as_ptr() as *mut ::std::os::raw::c_void,\n"
      << "        }\n"
      << "    }\n"
      << "}\n";

  runtime::Module test_module = InterfaceRustCreate(
      "ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}},
      {allocated_pool_info1, allocated_pool_info2}, {}, {}, {"input"}, {"output"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(workspace_struct.str()));

  ASSERT_THAT(header_source,
              HasSubstr("/// my_memory_pool_1 size for TVM module \"ultimate_cat_spotter\""));

  ASSERT_THAT(header_source,
              HasSubstr("pub const MY_MEMORY_POOL_1_WORKSPACE_POOL_SIZE: usize = 100000;"));

  ASSERT_THAT(header_source,
              HasSubstr("/// my_memory_pool_2 size for TVM module \"ultimate_cat_spotter\""));

  ASSERT_THAT(header_source,
              HasSubstr("pub const MY_MEMORY_POOL_2_WORKSPACE_POOL_SIZE: usize = 200000;"));
}

TEST(RustInterfaceAPI, ContainsWorkspacePoolStructSanitized) {
  PoolInfo pool_info = WorkspacePoolInfo("my_memory_pool+1", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info =
      tir::usmp::AllocatedPoolInfo(pool_info, 100000);

  std::stringstream workspace_struct;

  workspace_struct
      << "/// Workspace pools for TVM module \"ultimate_cat_spotter\"\n"
      << "#[repr(C)]\n"
      << "pub struct WorkspacePools {\n"
      << "    my_memory_pool_1: *mut ::std::os::raw::c_void,\n"
      << "}\n"
      << "\n"
      << "impl WorkspacePools {\n"
      << "    pub fn new <'a>(\n"
      << "        my_memory_pool_1: &mut [u8; 100000],\n"
      << "    ) -> Self {\n"
      << "        Self {\n"
      << "            my_memory_pool_1: my_memory_pool_1.as_ptr() as *mut ::std::os::raw::c_void,\n"
      << "        }\n"
      << "    }\n"
      << "}\n";

  runtime::Module test_module =
      InterfaceRustCreate("ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}},
                          {allocated_pool_info}, {}, {}, {"input"}, {"output"}, 0);
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(workspace_struct.str()));

  ASSERT_THAT(header_source,
              HasSubstr("/// my_memory_pool_1 size for TVM module \"ultimate_cat_spotter\""));

  ASSERT_THAT(header_source,
              HasSubstr("pub const MY_MEMORY_POOL_1_WORKSPACE_POOL_SIZE: usize = 100000;"));
}

TEST(RustInterfaceAPI, ContainsWorkspacePoolStructClash) {
  PoolInfo pool_info1 = WorkspacePoolInfo("my_memory_pool+", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info1 =
      tir::usmp::AllocatedPoolInfo(pool_info1, 100000);
  PoolInfo pool_info2 = WorkspacePoolInfo("my_memory_pool-", {});
  tir::usmp::AllocatedPoolInfo allocated_pool_info2 =
      tir::usmp::AllocatedPoolInfo(pool_info2, 200000);

  runtime::Module test_module = InterfaceRustCreate(
      "ultimate_cat_spotter", {{"input", TestIO()}}, {{"output", TestIO()}},
      {allocated_pool_info1, allocated_pool_info2}, {}, {}, {"input"}, {"output"}, 0);
  ASSERT_THROW(test_module->GetSource(), InternalError);
}

}  // namespace
}  // namespace codegen
}  // namespace tvm
