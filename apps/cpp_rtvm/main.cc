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

#include <chrono>
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
    "--pre-compiled - The file name of a file where pre-compiled programs should be stored\n"
    "--profile      - Profile over all execution\n"
    "--dry-run      - Profile after given dry runs, default 10\n"
    "--run-count    - Profile for given runs, default 50\n"
    "--zero-copy    - Profile with zero copy api\n"
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
  auto runner = new TVMRunner(args.model, args.device);

  // Load the model
  runner->Load();
  if (!args.pre_compiled.empty()) {
    runner->UsePreCompiledPrograms(args.pre_compiled);
  }

  // Query Model meta Information
  TVMMetaInfo mInfo = runner->GetMetaInfo();

  // Print Meta Information
  if (args.dump_meta) runner->PrintMetaInfo();

  int total_exec_time = 0;

  if (args.profile) {
    if (args.dry_run) {
      for (int ii = 0; ii < args.dry_run; ++ii) {
        runner->Run();
      }
      TVMSynchronize(GetTVMDevice(args.device), 0, nullptr);
    }
    int total_time = 0;
    std::map<std::string, NDArray> input_data_even, input_data_odd;
    std::map<std::string, NDArray> output_data_even, output_data_odd;

    std::map<std::string, char*> input_data;
    std::map<std::string, char*> output_data;

    // Alloc / populate and keep input data ready
    for (auto& elem : mInfo.input_info) {
      if (args.zero_copy) {
        auto ndarr =
            NDArray::Empty(elem.second.first, tvm::runtime::String2DLDataType(elem.second.second),
                           DLDevice{GetTVMDevice(args.device), 0});
        input_data_even.insert({elem.first, ndarr});

        ndarr =
            NDArray::Empty(elem.second.first, tvm::runtime::String2DLDataType(elem.second.second),
                           DLDevice{GetTVMDevice(args.device), 0});
        input_data_odd.insert({elem.first, ndarr});
      } else {
        char* data = (char*)malloc(runner->GetInputMemSize(elem.first));
        input_data.insert({elem.first, data});
      }
    }

    // Alloc and keep output bufers ready
    for (auto& elem : mInfo.output_info) {
      if (args.zero_copy) {
        auto ndarr =
            NDArray::Empty(elem.second.first, tvm::runtime::String2DLDataType(elem.second.second),
                           DLDevice{GetTVMDevice(args.device), 0});
        output_data_even.insert({elem.first, ndarr});

        ndarr =
            NDArray::Empty(elem.second.first, tvm::runtime::String2DLDataType(elem.second.second),
                           DLDevice{GetTVMDevice(args.device), 0});
        output_data_odd.insert({elem.first, ndarr});
      } else {
        char* data = (char*)malloc(runner->GetOutputMemSize(elem.first));
        output_data.insert({elem.first, data});
      }
    }

    for (int ii = 0; ii < args.run_count; ++ii) {
      // Timer start
      auto tstart = std::chrono::high_resolution_clock::now();
      // Set random input for all input
      for (auto& elem : mInfo.input_info) {
        if (args.zero_copy) {
          if (ii % 2) {
            runner->SetInput(elem.first, input_data_even[elem.first]);
          } else {
            runner->SetInput(elem.first, input_data_odd[elem.first]);
          }
        } else {
          runner->SetInput(elem.first, input_data[elem.first]);
        }
      }

      if (args.zero_copy) {
        // With zero copy set the result NDArray up front
        for (auto& elem : mInfo.output_info) {
          if (ii % 2) {
            runner->SetOutput(elem.first, output_data_even[elem.first]);
          } else {
            runner->SetOutput(elem.first, output_data_odd[elem.first]);
          }
        }
      }

      // Run the model
      runner->Run();

      if (!args.zero_copy) {
        // W/o zero copy we need to invoke explicite data copy
        for (auto& elem : mInfo.output_info) {
          runner->GetOutput(elem.first, output_data[elem.first]);
        }
      } else {
        // Just wait for the run to complete.
        TVMSynchronize(GetTVMDevice(args.device), 0, nullptr);
      }

      // Timer end
      auto tend = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "Exec Time:" << static_cast<double>((tend - tstart).count()) / 1e6;
      total_exec_time += static_cast<double>((tend - tstart).count()) / 1e6;
    }

    // Free input bufers
    for (auto& elem : mInfo.input_info) {
      free(input_data[elem.first]);
    }

    // Free output bufers
    for (auto& elem : mInfo.output_info) {
      free(output_data[elem.first]);
    }
  } else if (!args.input.empty() && !args.output.empty()) {
    LOG(INFO) << "Executing with Input:" << args.input << " Output:" << args.output;
    // Set Input from Numpy Input
    runner->SetInput(args.input);
    // Run the model
    runner->Run();
    // Get Output as Numpy dump
    runner->GetOutput(args.output);
  } else {
    LOG(INFO) << "Executing dry run ... ";
    // Set random input for all inputs
    for (auto& elem : mInfo.input_info) {
      LOG(INFO) << "Set Random Input for :" << elem.first;
      auto shape = elem.second.first;
      size_t ssize = runner->GetInputMemSize(elem.first);
      char* data = (char*)malloc(ssize);
      LOG(INFO) << "Random Input Size:" << ssize << "  bytes";
      runner->SetInput(elem.first, data);
      free(data);
    }
    // Run the model
    runner->Run();
    // Get Output and dump few values
    for (auto& elem : mInfo.output_info) {
      LOG(INFO) << "Get Output for :" << elem.first;
      auto shape = elem.second.first;
      size_t ssize = runner->GetOutputMemSize(elem.first);
      char* data = (char*)malloc(ssize);
      runner->GetOutput(elem.first, data);
      LOG(INFO) << "Output Size:" << ssize << "  bytes";
      free(data);
    }
  }

  if (args.profile) {
    // Print Stats
    runner->PrintStats();
  }
  auto tstart = std::chrono::high_resolution_clock::now();
  delete runner;
  auto tend = std::chrono::high_resolution_clock::now();

  if (args.profile) {
    LOG(INFO) << "Average ExecTime :" << total_exec_time / args.run_count << " ms";
    LOG(INFO) << "Unload Time      :" << static_cast<double>((tend - tstart).count()) / 1e6
              << " ms";
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
