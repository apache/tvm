/*!
 *  Copyright (c) 2018 by Contributors
 * \file rpc_server.cc
 * \brief RPC Server for TVM.
 */
#include <stdlib.h>
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <sstream>

#include "../../src/runtime/rpc/rpc_base.h"
#include "../../src/runtime/rpc/rpc_server.h"

using namespace tvm::runtime;

#define USAGE \
"Command line usage\n" \
" server       - Start the server\n" \
"--host        - The hostname of the server, Default=0.0.0.0\n" \
"--port        - The port of the RPC, Default=9090\n" \
"--port-end    - The end search port of the RPC, Default=9199\n" \
"--tracker     - The RPC tracker address in host:port format e.g. 10.1.1.2:9190 Default=\"\"\n" \
"--key         - The key used to identify the device type in tracker. Default=\"\"\n" \
"--custom-addr - Custom IP Address to Report to RPC Tracker. Default=\"\"\n" \
"--silent      - Whether run in silent mode. Default=True\n" \
"--proxy       - Whether to run in proxy mode. Default=False\n" \
"\n" \
"  Example\n" \
"  ./tvm_rpc server --host=0.0.0.0 --port=9000 --port-end=9090 --tracker=127.0.0.1:9190 --key=rasp" \
"\n"

/*!
 * \brief RpcServerArgs.
 * \arg host The hostname of the server, Default=0.0.0.0
 * \arg port The port of the RPC, Default=9090
 * \arg port_end The end search port of the RPC, Default=9199
 * \arg tracker The address of RPC tracker in host:port format e.g. 10.77.1.234:9190 Default=""
 * \arg key The key used to identify the device type in tracker. Default=""
 * \arg custom_addr Custom IP Address to Report to RPC Tracker. Default=""
 * \arg silent Whether run in silent mode. Default=True
 * \arg isProxy Whether to run in proxy mode. Default=False
 */
struct RpcServerArgs {
  std::string host = "0.0.0.0";
  int port = 9090;
  int port_end = 9099;
  std::string tracker = "";
  std::string key = "";
  std::string custom_addr = "";
  bool silent = false;
  bool isProxy = false;
};

/*!
 * \brief PrintArgs print the contents of RpcServerArgs
 * \param args RpcServerArgs structure
 */
void PrintArgs(struct RpcServerArgs args) {
  printf("host        = %s\n", args.host.c_str());
  printf("port        = %d\n", args.port);
  printf("port_end    = %d\n", args.port_end);
  printf("tracker     = %s\n", args.tracker.c_str());
  printf("key         = %s\n", args.key.c_str());
  printf("custom_addr = %s\n", args.custom_addr.c_str());
  printf("silent      = %s\n", ((args.silent) ? ("True"): ("False")));
  printf("proxy       = %s\n", ((args.isProxy) ? ("True"): ("False")));
}

/*!
 * \brief CtrlCHandler, exits if Ctrl+C is pressed
 * \param s signal
 */
void CtrlCHandler(int s){
  printf("\nUser pressed Ctrl+C, Exiting\n");
  exit(1);
}

/*!
 * \brief HandleCtrlC Register for handling Ctrl+C event.
 */
void HandleCtrlC() {
  //Ctrl+C handler
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = CtrlCHandler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);
}

/*!
 * \brief GetCmdOption Parse and find the command option.
 * \param argc arg counter
 * \param argv arg values
 * \param option command line option to search for.
 * \param key whether the option itself is key
 * \return value corresponding to option.
 */
std::string GetCmdOption(int argc, char* argv[], std::string option, bool key=false) {
  std::string cmd;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find(option) == 0) {
      if (key) {
        cmd = argv[i];
        return cmd;
      }
      cmd = arg.substr(arg.find_last_of(option) + 1);
      printf("cmd =%s\n", cmd.c_str());
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
bool ValidateTracker(std::string &tracker) {
  std::vector<std::string> list = SplitString(tracker, ':');
  if ((list.size() != 2) || (!ValidateIP(list[0])) || (!IsNumber(list[1]))) {
    return false;
  }
  std::ostringstream ss;
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
void ParseCmdArgs(int argc, char * argv[], struct RpcServerArgs &args){
  std::string host = GetCmdOption(argc, argv, "--host=");
  if (!host.empty()) {
    if (!ValidateIP(host)) {
      printf("Wrong host address format.\n");
      printf("%s", USAGE);
      exit(1);
    }
    args.host = host;
  }

  std::string port = GetCmdOption(argc, argv, "--port=");
  if (!port.empty()) {
    if (!IsNumber(port) || std::stoi(port) > 65535) {
      printf("Wrong port number.\n");
      printf("%s", USAGE);
      exit(1);
    }
    args.port = std::stoi(port);
  }

  std::string port_end = GetCmdOption(argc, argv, "--port_end=");
  if (!port_end.empty()) {
    if (!IsNumber(port_end) || std::stoi(port_end) > 65535) {
      printf("Wrong port_end number.\n");
      printf("%s", USAGE);
      exit(1);
    }
    args.port_end = std::stoi(port_end);
  }

  std::string tracker = GetCmdOption(argc, argv, "--tracker=");
  if (!tracker.empty()) {
    if (!ValidateTracker(tracker)) {
      printf("Wrong tracker address format.\n");
      printf("%s", USAGE);
      exit(1);
    }
    args.tracker = tracker;
  }

  std::string key = GetCmdOption(argc, argv, "--key=");
  if (!key.empty()) {
    args.key = key;
  }

  std::string custom_addr = GetCmdOption(argc, argv, "--custom_addr=");
  if (!custom_addr.empty()) {
    if (!ValidateIP(custom_addr)) {
      printf("Wrong custom address format.\n");
      printf("%s", USAGE);
      exit(1);
    }
    args.custom_addr = custom_addr;
  }

  std::string silent = GetCmdOption(argc, argv, "--silent", true);
  if (!silent.empty()) {
    args.silent = true;
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

  //Ctrl+C handler
  printf("Starting CPP Server, Press Ctrl+C to stop.\n");
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
      printf("%s", USAGE);
      return 0;
   }

  if (0 == strcmp(argv[1], "server")){
    RpcServer(argc, argv);
  } else {
    printf("%s", USAGE);
  }

  return 0;
}
