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

#include <tvm/runtime/logging.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "tvm_runner.h"

using namespace std;
using namespace tvm::runtime;

static const string kUsage =
    "Command line usage\n"
    "--model        - The tvm compiled Relax VM module (mod.so)\n"
    "--device       - The target device to use {llvm, opencl, cpu, cuda, metal, rocm, vpi, "
    "oneapi}\n"
    "--input        - Folder with one <param_name>.npy file per main() parameter\n"
    "--output       - Folder to dump the model outputs as <index>.npy\n"
    "--dump-meta    - Dump model meta information\n"
    "--pre-compiled - The file name of a file where pre-compiled programs should be stored\n"
    "--profile      - Profile over all execution\n"
    "--dry-run      - Profile after given dry runs, default 10\n"
    "--run-count    - Profile for given runs, default 50\n"
    "--zero-copy    - Profile with zero copy api\n"
    "\n"
    "  Example\n"
    "  ./rtvm --model=opencl_vm_mod.so --device=opencl --input=./inputs --dump-meta\n"
    "  ./rtvm --model=opencl_vm_mod.so --device=opencl --input=./inputs --output=./out\n"
    "\n";

/*!
 * \brief Tool Arguments.
 * \arg model The tvm artifact to load & run
 * \arg device The target device to use {llvm, cl, ...etc.}
 * \arg input Numpy file for the model input
 * \arg output Numpy file name to dump the model output as numpy
 * \arg pre_compiled File name where pre-compiled programs should be stored
 * \arg profile Do we profile overall execution
 */
struct ToolArgs {
  string model;
  string device;
  string input;
  string output;
  string pre_compiled;
  bool dump_meta{false};
  bool profile{false};
  int dry_run{10};
  int run_count{50};
  bool zero_copy{false};
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
  LOG(INFO) << "Profile       = " << ((args.profile) ? ("True") : ("False"));
  LOG(INFO) << "Dry Run       = " << args.dry_run;
  LOG(INFO) << "Run Count     = " << args.run_count;
  LOG(INFO) << "Zero Copy     = " << ((args.zero_copy) ? ("True") : ("False"));
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
      TVM_FFI_ICHECK_EQ(*option.rbegin(), '=');
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
  } else {
    LOG(INFO) << kUsage;
    exit(0);
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

  const string pprofile = GetCmdOption(argc, argv, "--profile", true);
  if (!pprofile.empty()) {
    args.profile = true;
  }

  const string pdry_run = GetCmdOption(argc, argv, "--dry-run=");
  if (!pdry_run.empty()) {
    args.dry_run = stoi(pdry_run);
  }

  const string prun = GetCmdOption(argc, argv, "--run-count=");
  if (!prun.empty()) {
    args.run_count = stoi(prun);
  }

  const string pzcopy = GetCmdOption(argc, argv, "--zero-copy", true);
  if (!pzcopy.empty()) {
    args.zero_copy = true;
  }
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
  auto runner = std::make_unique<TVMRunner>(args.model, args.device);

  // Load the model
  runner->Load();

  // Query Model meta Information
  runner->GetMetaInfo();

  // // Print Meta Information
  if (args.dump_meta) runner->PrintMetaInfo();

  // Create input NDArray tensor with give input numpy files
  runner->CreateInputNDArrayFromFile(args.input);

  double total_exec_time = 0;

  if (args.profile) {
    if (args.dry_run) {
      runner->SetInput();
      for (int ii = 0; ii < args.dry_run; ++ii) {
        runner->Run();
      }
      DeviceAPI::Get(DLDevice{GetTVMDevice(args.device), 0})
          ->StreamSync(DLDevice{GetTVMDevice(args.device), 0}, nullptr);
    }
    // Timer start
    auto tstart = std::chrono::high_resolution_clock::now();

    for (int ii = 0; ii < args.run_count; ++ii) {
      // Set for all input

      runner->SetInput();
      // Run the model
      runner->Run();
    }
    // Just wait for the run to complete.
    DeviceAPI::Get(DLDevice{GetTVMDevice(args.device), 0})
        ->StreamSync(DLDevice{GetTVMDevice(args.device), 0}, nullptr);
    //  Timer end
    auto tend = std::chrono::high_resolution_clock::now();
    total_exec_time += std::chrono::duration<double, std::milli>(tend - tstart).count();
  } else {
    LOG(INFO) << "Executing with Input:" << args.input << " Output:" << args.output;
    // Set Input from Numpy Input
    runner->SetInput();
    // Run the model
    runner->Run();
    // Device kernels may execute asynchronously.  Ensure outputs are ready before copying them.
    DeviceAPI::Get(DLDevice{GetTVMDevice(args.device), 0})
        ->StreamSync(DLDevice{GetTVMDevice(args.device), 0}, nullptr);
    // Get Output as Numpy dump
    runner->GetOutput(args.output);
  }

  if (args.profile) {
    // Print Stats
    runner->PrintStats();
  }
  auto tstart = std::chrono::high_resolution_clock::now();
  runner.reset();
  auto tend = std::chrono::high_resolution_clock::now();

  if (args.profile) {
    LOG(INFO) << "Average ExecTime :" << total_exec_time / args.run_count << " ms";
    LOG(INFO) << "Unload Time      :"
              << std::chrono::duration<double, std::milli>(tend - tstart).count() << " ms";
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

  try {
    ToolArgs args;
    ParseCmdArgs(argc, argv, args);
    PrintArgs(args);

    if (ExecuteModel(args)) {
      PrintArgs(args);
      LOG(INFO) << kUsage;
      return -1;
    }
  } catch (const tvm::ffi::Error& error) {
    LOG(ERROR) << error.message();
    return -1;
  } catch (const std::exception& error) {
    LOG(ERROR) << error.what();
    return -1;
  }
  return 0;
}
