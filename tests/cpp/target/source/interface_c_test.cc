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

using ::testing::HasSubstr;

namespace tvm {
namespace codegen {

runtime::Module InterfaceCCreate(std::string module_name, Array<String> inputs,
                                 Array<String> outputs, Array<String> devices);

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

  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {});
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

  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {});
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
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {"device"});
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(run_function.str()));
}

TEST(InterfaceAPI, ContainsInputStructSingle) {
  std::stringstream input_struct;

  input_struct << "/*!\n"
               << " * \\brief Input tensor pointers for TVM module \"ultimate_cat_spotter\" \n"
               << " */\n"
               << "struct tvmgen_ultimate_cat_spotter_inputs {\n"
               << "  void* input;\n"
               << "};\n\n";

  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {});
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
      InterfaceCCreate("ultimate_cat_spotter", {"input1", "input2"}, {"output"}, {});
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
      InterfaceCCreate("ultimate_cat_spotter", {"input+1", "input+2"}, {"output"}, {});
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(input_struct.str()));
}

TEST(InterfaceAPI, ContainsInputStructClash) {
  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input+", "input-"}, {"output"}, {});
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

  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {});
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
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output1", "output2"}, {});
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
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output+1", "output-2"}, {});
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(output_struct.str()));
}

TEST(InterfaceAPI, ContainsOutputStructClash) {
  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output+", "output-"}, {});
  ASSERT_THROW(test_module->GetSource(), InternalError);
}

TEST(InterfaceAPI, NoDeviceAPIStructIfNoDevices) {
  std::stringstream device_struct;

  device_struct << "/*!\n"
                << " * \\brief Device context pointers for TVM module \"ultimate_cat_spotter\" \n"
                << " */\n"
                << "struct tvmgen_ultimate_cat_spotter_devices {\n"
                << "};\n\n";

  runtime::Module test_module = InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {});
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
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {"device"});
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(device_struct.str()));
}

TEST(InterfaceAPI, ContainsDeviceStructMany) {
  std::stringstream device_struct;

  device_struct << "struct tvmgen_ultimate_cat_spotter_devices {\n"
                << "  void* device1;\n"
                << "  void* device2;\n"
                << "};\n\n";

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {"device1", "device2"});
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(device_struct.str()));
}

TEST(InterfaceAPI, ContainsDeviceStructSanitised) {
  std::stringstream device_struct;

  device_struct << "struct tvmgen_ultimate_cat_spotter_devices {\n"
                << "  void* device_1;\n"
                << "  void* device_2;\n"
                << "};\n\n";

  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {"device+1", "device+2"});
  std::string header_source = test_module->GetSource();

  ASSERT_THAT(header_source, HasSubstr(device_struct.str()));
}

TEST(InterfaceAPI, ContainsDeviceStructClash) {
  runtime::Module test_module =
      InterfaceCCreate("ultimate_cat_spotter", {"input"}, {"output"}, {"device+", "device-"});
  ASSERT_THROW(test_module->GetSource(), InternalError);
}

}  // namespace
}  // namespace codegen
}  // namespace tvm
