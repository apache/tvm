/*!
 *  Copyright (c) 2018 by Contributors
 * \file rpc_server_env.h
 * \brief Server environment of the RPC.
 */
#ifndef TVM_RUNTIME_RPC_RPC_SERVER_ENV_H_
#define TVM_RUNTIME_RPC_RPC_SERVER_ENV_H_

#include <tvm/runtime/registry.h>
#if defined(__linux__)
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <string>

namespace tvm {
namespace runtime {

/*!
 * \brief Load Load module from file
          This function will automatically call
          cc.create_shared if the path is in format .o or .tar
          High level handling for .o and .tar file.
          We support this to be consistent with RPC module load.
 * \param file The input file
 * \param file The format of file
 * \return Module The loaded module
 */
Module Load(std::string *path, const std::string fmt = "");

/*!
 * \brief CleanDir Removes the files from the directory
 * \param dirname THe name of the directory
 */
void CleanDir(const std::string dirname);

/*!
 * \brief RPCEnv The RPC Environment parameters for c++ rpc server
 */
struct RPCEnv {
 public:
  RPCEnv() {
    #if defined(__linux__) || defined(__ANDROID__)
      base_ = "rpc";
      mkdir(&base_[0], 0777);

      TVM_REGISTER_GLOBAL("tvm.rpc.server.workpath")
      .set_body([](TVMArgs args, TVMRetValue* rv) {
          static RPCEnv env;
          *rv = env.GetPath(args[0]);
        });

      TVM_REGISTER_GLOBAL("tvm.rpc.server.load_module")
      .set_body([](TVMArgs args, TVMRetValue *rv) {
          std::string file_name = "rpc/" + args[0].operator std::string();
          *rv = Load(&file_name, "");
          LOG(INFO) << "Load module from " << file_name << " ...";
        });
    #else
      LOG(FATAL) << "Only support RPC in linux environment";
    #endif
  }

  /*!
   * \brief GetPath To get the workpath from packed function
   * \param name The file name
   * \return The full path of file.
   */
  std::string GetPath(const std::string& file_name) {
    return base_ + "/" + file_name;
  }

  /*!
   * \brief Remove The RPC Environment cleanup function
   */
  void Remove() {
    #if defined(__linux__) || defined(__ANDROID__)
      CleanDir(&base_[0]);
      rmdir(&base_[0]);
    #else
      LOG(FATAL) << "Only support RPC in linux environment";
    #endif
  }

 private:
  std::string base_;
};  // RPCEnv

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_SERVER_ENV_H_
