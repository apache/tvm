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
 * \brief TVM runtime utility for TVM.
 */
#include <csignal>
#include <cstdio>
#include <cstdlib>
#if defined(__linux__) || defined(__ANDROID__)
#include <unistd.h>
#endif
#include <dmlc/logging.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#include "../../src/support/socket.h"
#include "../../src/support/utils.h"
#include "tvm_runner.h"

#if defined(_WIN32)
#include "win32_process.h"
#endif

using namespace std;
using namespace tvm::runtime;
using namespace tvm::support;

static const string kUsage =
    "Command line usage\n"
    "--model        - The folder containing tvm artifacts(mod.so, mod.param, mod.json) \n"
    "--device       - The target device to use {llvm, opencl, cpu, cuda, metal, rocm, vpi, "
    "oneapi}\n"
    "--input        - Numpy file for the model input (optional and we use random of not given)\n"
    "--output       - Numpy file name to dump the model output as numpy\n"
    "--dump-meta    - Dump model meta information\n"
    "--pre-compiled - The file name of a file where pre-compiled programs should be stored"
    "\n"
    "  Example\n"
    "  ./rtvm --model=keras-resnet50 --device=\"opencl\" --dump-meta\n"
    "  ./rtvm --model=keras-resnet50 --device=\"opencl\" --input input.npz --output=output.npz\n"
    "\n";

/*!
 * \brief Tool Arguments.
 * \arg model The tvm artifact to load & run
 * \arg device The target device to use {llvm, cl, ...etc.}
 * \arg input Numpy file for the model input
 * \arg output Numpy file name to dump the model output as numpy
 * \arg pre_compiled File name where pre-compiled programs should be stored
 */
struct ToolArgs {
  string model;
  string device;
  string input;
  string output;
  string pre_compiled;
  bool dump_meta = false;
};

/*!
 * \brief PrintArgs print the contents of ToolArgs
 * \param args ToolArgs structure
 */
void PrintArgs(const ToolArgs& args) {
  LOG(INFO) << "Model         = " << args.model;
  LOG(INFO) << "Device        = " << args.device;
  LOG(INFO) << "Input         = " << args.input;
  LOG(INFO) << "Output        = " << args.output;
  LOG(INFO) << "Pre-compiled  = " << args.pre_compiled;
  LOG(INFO) << "Dump Metadata = " << ((args.dump_meta) ? ("True") : ("False"));
}

#if defined(__linux__) || defined(__ANDROID__)
/*!
 * \brief CtrlCHandler, exits if Ctrl+C is pressed
 * \param s signal
 */
void CtrlCHandler(int s) {
  LOG(INFO) << "\nUser pressed Ctrl+C, Exiting";
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
string GetCmdOption(int argc, char* argv[], string option, bool key = false) {
  string cmd;
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg.find(option) == 0) {
      if (key) {
        cmd = argv[i];
        return cmd;
      }
      // We assume "=" is the end of option.
      ICHECK_EQ(*option.rbegin(), '=');
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
  const string model = GetCmdOption(argc, argv, "--model=");
  if (!model.empty()) {
    args.model = model;
  } else {
    LOG(INFO) << kUsage;
    exit(0);
  }

  const string device = GetCmdOption(argc, argv, "--device=");
  if (!device.empty()) {
    args.device = device;
  } else {
    LOG(INFO) << kUsage;
    exit(0);
  }

  const string input = GetCmdOption(argc, argv, "--input=");
  if (!input.empty()) {
    args.input = input;
  }

  const string output = GetCmdOption(argc, argv, "--output=");
  if (!output.empty()) {
    args.output = output;
  }

  const string pmeta = GetCmdOption(argc, argv, "--dump-meta", true);
  if (!pmeta.empty()) {
    args.dump_meta = true;
  }

  args.pre_compiled = GetCmdOption(argc, argv, "--pre-compiled=");
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

  // Initialize TVM Runner
  TVMRunner runner = TVMRunner(args.model, args.device);

  // Load the model
  runner.Load();
  if (!args.pre_compiled.empty()) {
    runner.UsePreCompiledPrograms(args.pre_compiled);
  }

  // Query Model meta Information
  TVMMetaInfo mInfo = runner.GetMetaInfo();

  // Print Meta Information
  if (args.dump_meta) runner.PrintMetaInfo();

  if (args.input.empty() || args.output.empty()) {
    LOG(INFO) << "Executing dry run ... ";
    // Set random input for all inputs
    for (auto& elem : mInfo.input_info) {
      LOG(INFO) << "Set Random Input for :" << elem.first;
      auto shape = elem.second.first;
      size_t ssize = runner.GetInputMemSize(elem.first);
      char* data = (char*)malloc(ssize);
      LOG(INFO) << "Random Input Size:" << ssize << "  bytes";
      runner.SetInput(elem.first, data);
      free(data);
    }

    // Run the model
    runner.Run();

    // Get Output and dump few values
    for (auto& elem : mInfo.output_info) {
      LOG(INFO) << "Get Output for :" << elem.first;
      auto shape = elem.second.first;
      size_t ssize = runner.GetOutputMemSize(elem.first);
      char* data = (char*)malloc(ssize);
      runner.GetOutput(elem.first, data);
      LOG(INFO) << "Output Size:" << ssize << "  bytes";
      free(data);
    }
  } else {
    LOG(INFO) << "Executing with Input:" << args.input << " Output:" << args.output;

    // Set Input from Numpy Input
    runner.SetInput(args.input);

    // Run the model
    runner.Run();

    // Get Output as Numpy dump
    runner.GetOutput(args.output);
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
