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

#include "../../src/runtime/rpc/rpc_server.h"

#define USAGE \
"Command line usage\n" \
" server    - Start the server\n" \
"--host     - The hostname of the server, Default=0.0.0.0\n" \
"--port     - The port of the RPC, Default=9090\n" \
"--port-end - The end search port of the RPC, Default=9199\n" \
"--tracker  - The address of RPC tracker in host:port format e.g. 10.77.1.234:9190 Default=\"\"\n" \
"--key      - The key used to identify the device type in tracker. Default=\"\"\n" \
"--silent   - Whether run in silent mode. Default=True\n" \
"--custom-addr - Custom IP Address to Report to RPC Tracker. Default=\"\"\n" \
"\n" \
"  Example\n" \
"  ./tvm_rpc server --host=0.0.0.0 --port=9000 --port-end=9090 --tracker=127.0.0.1:9190 --key=rasp"

/*!
 * \brief xxxx.
 * \param xxxx
 * \return xxxx.
 */
struct RpcServerArgs {
  std::string host = "0.0.0.0";
  int port = 9090;
  int port_end = 9099;
  std::string tracker = "";
  std::string key = "";
  std::string custom_addr = "";
  bool silent = false;
};

/*!
 * \brief xxxx.
 * \param xxxx
 * \return xxxx.
 */
void printArgs(struct RpcServerArgs args) {
  printf("host        = %s\n", args.host.c_str());
  printf("port        = %d\n", args.port);
  printf("port_end    = %d\n", args.port_end);
  printf("tracker     = %s\n", args.tracker.c_str());
  printf("key         = %s\n", args.key.c_str());
  printf("custom_addr = %s\n", args.custom_addr.c_str());
  printf("silent      = %s\n", ((args.silent) ? ("True"): ("False")));
}

/*!
 * \brief xxxx.
 * \param xxxx
 * \return xxxx.
 */
void ctrlCHandler(int s){
  printf("\nUser pressed Ctrl+C, Exiting\n");
  exit(1);
}

/*!
 * \brief xxxx.
 * \param xxxx
 * \return xxxx.
 */
void handleCtrlC() {
  //Ctrl+C handler
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = ctrlCHandler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);
}

/*!
 * \brief xxxx.
 * \param xxxx
 * \return xxxx.
 */
std::string getCmdOption(int argc, char* argv[], std::string option, bool key=false) {
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
 * \brief xxxx.
 * \param xxxx
 * \return xxxx.
 */
bool isNumber(const std::string& str) {
  return !str.empty() &&
    (str.find_first_not_of("[0123456789]") == std::string::npos);
}

/*!
 * \brief xxxx.
 * \param xxxx
 * \return xxxx.
 */
std::vector<std::string> split(const std::string& str, char delim) {
    auto i = 0;
    std::vector<std::string> list;
    auto pos = str.find(delim);
    while (pos != std::string::npos) {
      list.push_back(str.substr(i, pos - i));
      i = ++pos;
      pos = str.find(delim, pos);
    }
    list.push_back(str.substr(i, str.length()));
    return list;
}

/*!
 * \brief xxxx.
 * \param xxxx
 * \return xxxx.
 */
bool validateIP(std::string ip) {
    std::vector<std::string> list = split(ip, '.');
    if (list.size() != 4)
        return false;
    for (std::string str : list) {
      if (!isNumber(str) || std::stoi(str) > 255 || std::stoi(str) < 0)
        return false;
    }
    return true;
}

/*!
 * \brief xxxx.
 * \param xxxx
 * \return xxxx.
 */
bool validateTracker(std::string &tracker) {
  std::vector<std::string> list = split(tracker, ':');
  if ((list.size() != 2) || (!validateIP(list[0])) || (!isNumber(list[1]))) {
    return false;
  }
  std::ostringstream ss;
  ss << "('" << list[0] << "', " << list[1] << ")";
  tracker = ss.str();
  return true;
}

/*!
 * \brief xxxx.
 * \param xxxx
 * \return xxxx.
 */
void parseCmdArgs(int argc, char * argv[], struct RpcServerArgs &args){
  std::string host = getCmdOption(argc, argv, "--host=");
  if (!host.empty()) {
    if (!validateIP(host)) {
      printf("Wrong host address format.\n");
      printf("%s", USAGE);
      exit(1);
    }
    args.host = host;
  }

  std::string port = getCmdOption(argc, argv, "--port=");
  if (!port.empty()) {
    if (!isNumber(port) || std::stoi(port) > 65535) {
      printf("Wrong port number.\n");
      printf("%s", USAGE);
      exit(1);
    }
    args.port = std::stoi(port);
  }

  std::string port_end = getCmdOption(argc, argv, "--port_end=");
  if (!port_end.empty()) {
    if (!isNumber(port_end) || std::stoi(port_end) > 65535) {
      printf("Wrong port_end number.\n");
      printf("%s", USAGE);
      exit(1);
    }
    args.port_end = std::stoi(port_end);
  }

  std::string tracker = getCmdOption(argc, argv, "--tracker=");
  if (!tracker.empty()) {
    if (!validateTracker(tracker)) {
      printf("Wrong tracker address format.\n");
      printf("%s", USAGE);
      exit(1);
    }
    args.tracker = tracker;
  }

  std::string key = getCmdOption(argc, argv, "--key=");
  if (!key.empty()) {
    args.key = key;
  }

  std::string custom_addr = getCmdOption(argc, argv, "--custom_addr=");
  if (!custom_addr.empty()) {
    if (!validateIP(custom_addr)) {
      printf("Wrong custom address format.\n");
      printf("%s", USAGE);
      exit(1);
    }
    args.custom_addr = custom_addr;
  }

  std::string silent = getCmdOption(argc, argv, "--silent", true);
  if (!silent.empty()) {
    args.silent = true;
  }
}

/*!
 * \brief xxxx.
 * \param xxxx
 * \return xxxx.
 */
int rpcServer(int argc, char * argv[]) {
  struct RpcServerArgs args;

  /* parse the command line args */
  parseCmdArgs(argc, argv, args);

  //Ctrl+C handler
  printf("Starting CPP Server, Press Ctrl+C to stop.\n");
  handleCtrlC();
  printArgs(args);
  tvm::runtime::RPCServerCreate(args.host, args.port, args.port_end, args.tracker,
                                args.key, args.custom_addr, args.silent);
  printf("Server functionality\n");
  return 0;
}

/*!
 * \brief xxxx.
 * \param xxxx
 * \return xxxx.
 */
int main(int argc, char * argv[]) {
  if (argc <= 1) {
      printf("%s", USAGE);
      return 0;
   }

  if (0 == strcmp(argv[1], "server")){
    rpcServer(argc, argv);
  } else {
    printf("%s", USAGE);
  }

  return 0;
}
