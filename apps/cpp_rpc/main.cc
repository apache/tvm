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
 * \file rpc_server.cc
 * \brief RPC Server for TVM.
 */
#include <stdlib.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <dmlc/logging.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <sstream>

#include "../../src/support/util.h"
#include "../../src/support/socket.h"
#include "rpc_server.h"

using namespace std;
using namespace tvm::runtime;
using namespace tvm::support;

static const string kUSAGE = \
"Command line usage\n" \
" server       - Start the server\n" \
"--host        - The hostname of the server, Default=0.0.0.0\n" \
"--port        - The port of the RPC, Default=9090\n" \
"--port-end    - The end search port of the RPC, Default=9199\n" \
"--tracker     - The RPC tracker address in host:port format e.g. 10.1.1.2:9190 Default=\"\"\n" \
"--key         - The key used to identify the device type in tracker. Default=\"\"\n" \
"--custom-addr - Custom IP Address to Report to RPC Tracker. Default=\"\"\n" \
"--silent      - Whether to run in silent mode. Default=False\n" \
"\n" \
"  Example\n" \
"  ./tvm_rpc server --host=0.0.0.0 --port=9000 --port-end=9090 "
" --tracker=127.0.0.1:9190 --key=rasp" \
"\n";

/*!
 * \brief RpcServerArgs.
 * \arg host The hostname of the server, Default=0.0.0.0
 * \arg port The port of the RPC, Default=9090
 * \arg port_end The end search port of the RPC, Default=9199
 * \arg tracker The address of RPC tracker in host:port format e.g. 10.77.1.234:9190 Default=""
 * \arg key The key used to identify the device type in tracker. Default=""
 * \arg custom_addr Custom IP Address to Report to RPC Tracker. Default=""
 * \arg silent Whether run in silent mode. Default=False
 */
struct RpcServerArgs {
  string host = "0.0.0.0";
  int port = 9090;
  int port_end = 9099;
  string tracker;
  string key;
  string custom_addr;
  bool silent = false;
};

/*!
 * \brief PrintArgs print the contents of RpcServerArgs
 * \param args RpcServerArgs structure
 */
void PrintArgs(struct RpcServerArgs args) {
  LOG(INFO) << "host        = " << args.host;
  LOG(INFO) << "port        = " << args.port;
  LOG(INFO) << "port_end    = " << args.port_end;
  LOG(INFO) << "tracker     = " << args.tracker;
  LOG(INFO) << "key         = " << args.key;
  LOG(INFO) << "custom_addr = " << args.custom_addr;
  LOG(INFO) << "silent      = " << ((args.silent) ? ("True"): ("False"));
}

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
      CHECK_EQ(*option.rbegin(), '=');
      cmd = arg.substr(arg.find("=") + 1);
      return cmd;
    }
  }
  return cmd;
}

/*!
 * \brief ValidateTracker Check the tracker address format is correct and changes the format.
 * \param tracker The tracker input.
 * \return result of operation.
 */
bool ValidateTracker(string &tracker) {
  vector<string> list = Split(tracker, ':');
  if ((list.size() != 2) || (!ValidateIP(list[0])) || (!IsNumber(list[1]))) {
    return false;
  }
  ostringstream ss;
  ss << "('" << list[0] << "', " << list[1] << ")";
  tracker = ss.str();
  return true;
}

/*!
 * \brief ParseCmdArgs parses the command line arguments.
 * \param argc arg counter
 * \param argv arg values
 * \param args, the output structure which holds the parsed values
 */
void ParseCmdArgs(int argc, char * argv[], struct RpcServerArgs &args) {
  string silent = GetCmdOption(argc, argv, "--silent", true);
  if (!silent.empty()) {
    args.silent = true;
    // Only errors and fatal is logged
    dmlc::InitLogging("--minloglevel=2");
  }

  string host = GetCmdOption(argc, argv, "--host=");
  if (!host.empty()) {
    if (!ValidateIP(host)) {
      LOG(WARNING) << "Wrong host address format.";
      LOG(INFO) << kUSAGE;
      exit(1);
    }
    args.host = host;
  }

  string port = GetCmdOption(argc, argv, "--port=");
  if (!port.empty()) {
    if (!IsNumber(port) || stoi(port) > 65535) {
      LOG(WARNING) << "Wrong port number.";
      LOG(INFO) << kUSAGE;
      exit(1);
    }
    args.port = stoi(port);
  }

  string port_end = GetCmdOption(argc, argv, "--port_end=");
  if (!port_end.empty()) {
    if (!IsNumber(port_end) || stoi(port_end) > 65535) {
      LOG(WARNING) << "Wrong port_end number.";
      LOG(INFO) << kUSAGE;
      exit(1);
    }
    args.port_end = stoi(port_end);
  }

  string tracker = GetCmdOption(argc, argv, "--tracker=");
  if (!tracker.empty()) {
    if (!ValidateTracker(tracker)) {
      LOG(WARNING) << "Wrong tracker address format.";
      LOG(INFO) << kUSAGE;
      exit(1);
    }
    args.tracker = tracker;
  }

  string key = GetCmdOption(argc, argv, "--key=");
  if (!key.empty()) {
    args.key = key;
  }

  string custom_addr = GetCmdOption(argc, argv, "--custom_addr=");
  if (!custom_addr.empty()) {
    if (!ValidateIP(custom_addr)) {
      LOG(WARNING) << "Wrong custom address format.";
      LOG(INFO) << kUSAGE;
      exit(1);
    }
    args.custom_addr = custom_addr;
  }
}

/*!
 * \brief RpcServer Starts the RPC server.
 * \param argc arg counter
 * \param argv arg values
 * \return result of operation.
 */
int RpcServer(int argc, char * argv[]) {
  struct RpcServerArgs args;

  /* parse the command line args */
  ParseCmdArgs(argc, argv, args);
  PrintArgs(args);

  // Ctrl+C handler
  LOG(INFO) << "Starting CPP Server, Press Ctrl+C to stop.";
  HandleCtrlC();
  tvm::runtime::RPCServerCreate(args.host, args.port, args.port_end, args.tracker,
                                args.key, args.custom_addr, args.silent);
  return 0;
}

/*!
 * \brief main The main function.
 * \param argc arg counter
 * \param argv arg values
 * \return result of operation.
 */
int main(int argc, char * argv[]) {
  if (argc <= 1) {
    LOG(INFO) << kUSAGE;
    return 0;
  }

  if (0 == strcmp(argv[1], "server")) {
    RpcServer(argc, argv);
  } else {
    LOG(INFO) << kUSAGE;
  }

  return 0;
}
