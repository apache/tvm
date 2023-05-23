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

/*!
 * \file main.cc
 * \brief CLML Model execution application.
 */

#include "clml_runner.h"

using namespace tvm::runtime;

/*!
 * \brief Auto generated model file (clml_models.cc) entry function definition.
 * \param args The tool arguments to forward
 * \param arg_platform OpenCL platform
 * \param arg_context OpenCL context
 * \param arg_device_id OpenCL device id
 * \param queue OpenCL queue
 * \return List of CLMLRunner objects corresponding to all sub graphs of a TVM module.
 */
std::vector<CLMLRunner> BuildModules(ToolArgs& args, cl_platform_id arg_platform,
                                     cl_context arg_context, cl_device_id arg_device_id,
                                     cl_command_queue queue);

static const std::string kUsage =
    "Command line usage\n"
    "--input        - Numpy file for the model input (optional and we use random of not given)\n"
    "--output       - Numpy file name to dump the model output as numpy\n"
    "--params       - Numpy file with params\n"
    "--dump-meta    - Dump model meta information\n"
    "\n"
    "  Example\n"
    "  ./clml_run --dump-meta\n"
    "  ./clml_run --params=clmlparams.npz\n"
    "  ./clml_run --input=input.npz --output=output.npz --params=clml_params.npz\n"
    "\n";

/*!
 * \brief PrintArgs print the contents of ToolArgs
 * \param args ToolArgs structure
 */
void PrintArgs(const ToolArgs& args) {
  LOG(INFO) << "Input         = " << args.input;
  LOG(INFO) << "Output        = " << args.output;
  LOG(INFO) << "Params        = " << args.params;
  LOG(INFO) << "DumpMeta      = " << args.dump_meta;
}

#if defined(__linux__) || defined(__ANDROID__)
/*!
 * \brief CtrlCHandler, exits if Ctrl+C is pressed
 * \param s signal
 */
void CtrlCHandler(int s) {
  LOG(INFO) << "User pressed Ctrl+C, Exiting";
  exit(1);
}

/*!
 * \brief HandleCtrlC Register for handling Ctrl+C event.
 */
void HandleCtrlC() {
  // Ctrl+C handler
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = CtrlCHandler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, nullptr);
}
#endif
/*!
 * \brief GetCmdOption Parse and find the command option.
 * \param argc arg counter
 * \param argv arg values
 * \param option command line option to search for.
 * \param key whether the option itself is key
 * \return value corresponding to option.
 */
std::string GetCmdOption(int argc, char* argv[], std::string option, bool key = false) {
  std::string cmd;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find(option) == 0) {
      if (key) {
        cmd = argv[i];
        return cmd;
      }
      // We assume "=" is the end of option.
      // ICHECK_EQ(*option.rbegin(), '=');
      cmd = arg.substr(arg.find('=') + 1);
      return cmd;
    }
  }
  return cmd;
}

/*!
 * \brief ParseCmdArgs parses the command line arguments.
 * \param argc arg counter
 * \param argv arg values
 * \param args the output structure which holds the parsed values
 */
void ParseCmdArgs(int argc, char* argv[], struct ToolArgs& args) {
  const std::string input = GetCmdOption(argc, argv, "--input=");
  if (!input.empty()) {
    args.input = input;
  }

  const std::string output = GetCmdOption(argc, argv, "--output=");
  if (!output.empty()) {
    args.output = output;
  }

  const std::string params = GetCmdOption(argc, argv, "--params=");
  if (!params.empty()) {
    args.params = params;
  }

  const std::string pmeta = GetCmdOption(argc, argv, "--dump-meta", true);
  if (!pmeta.empty()) {
    args.dump_meta = true;
  }
}

/*!
 * \brief Check CLML extension availability in the CL device.
 * \param platform_id OpenCL platform
 * \param device_id OpenCL device id
 * \return true if extension present else false.
 */
bool ExtensionStringPresent(cl_platform_id platform_id, cl_device_id device_id) {
  cl_int result = 0;
  size_t reqd_size = 0;
  result = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, 0, nullptr, &reqd_size);
  CLML_SDK_TEST_AND_EXIT(reqd_size > 0u && result == CL_SUCCESS);

  std::vector<char> buf(reqd_size);
  result = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, reqd_size, buf.data(), nullptr);
  CLML_SDK_TEST_AND_EXIT(result == CL_SUCCESS);

  std::string extensions(buf.data());
  LOG(WARNING) << "OpenCL Extensions:" << extensions;
  return (extensions.find("cl_qcom_ml_ops") != std::string::npos);
}

/*!
 * \brief Loads and Executes the model on given Target.
 * \param args tool arguments
 * \return result of operation.
 */
int ExecuteModel(ToolArgs& args) {
#if defined(__linux__) || defined(__ANDROID__)
  // Ctrl+C handler
  HandleCtrlC();
#endif

  // Init OpenCL Environment
  cl_int result;
  cl_event readEvent = nullptr;
  cl_platform_id platform = nullptr;
  cl_context context = nullptr;
  cl_device_id device_id = nullptr;
  cl_command_queue queue = nullptr;

  // Initialize Context and Command Queue
  result = clGetPlatformIDs(1, &platform, nullptr);
  CLML_SDK_TEST_AND_EXIT(result == CL_SUCCESS);

  uint32_t num_devices = 0;
  result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
  CLML_SDK_TEST_AND_EXIT(result == CL_SUCCESS && num_devices == 1);

  result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
  CLML_SDK_TEST_AND_EXIT(device_id && result == CL_SUCCESS);

  CLML_SDK_TEST_AND_EXIT(ExtensionStringPresent(platform, device_id) == true);

  context = clCreateContext(0, 1, &device_id, nullptr, nullptr, &result);
  CLML_SDK_TEST_AND_EXIT(result == CL_SUCCESS);

  cl_command_queue_properties queue_props = 0;

  queue = clCreateCommandQueue(context, device_id, queue_props, &result);
  CLML_SDK_TEST_AND_EXIT(queue && result == CL_SUCCESS);

  // Populate the runner with model
  LOG(INFO) << "Call Build Modules\n";
  auto runners = BuildModules(args, platform, context, device_id, queue);

  LOG(INFO) << "Loop Through the Modules";
  for (auto runner : runners) {
    if (args.dump_meta) {
      // Print Meta Information
      runner.PrintMetaInfo();
    }

    // Run the model
    runner.Run();
  }

  return 0;
}

/*!
 * \brief main The main function.
 * \param argc arg counter
 * \param argv arg values
 * \return result of operation.
 */
int main(int argc, char* argv[]) {
  if (argc <= 1) {
    LOG(INFO) << kUsage;
    return 0;
  }

  ToolArgs args;
  ParseCmdArgs(argc, argv, args);
  PrintArgs(args);

  if (ExecuteModel(args)) {
    PrintArgs(args);
    LOG(INFO) << kUsage;
    return -1;
  }
  return 0;
}
