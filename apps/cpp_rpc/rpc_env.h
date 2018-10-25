/*!
 *  Copyright (c) 2018 by Contributors
 * \file rpc_env.h
 * \brief Server environment of the RPC.
 */
#ifndef TVM_APPS_CPP_RPC_ENV_H_
#define TVM_APPS_CPP_RPC_ENV_H_

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
void CleanDir(const std::string &dirname);

/*!
 * \brief RPCEnv The RPC Environment parameters for c++ rpc server
 */
struct RPCEnv {
 public:
   /*!
    * \brief Constructor Init The RPC Environment initialize function
    */
  RPCEnv();
  /*!
   * \brief GetPath To get the workpath from packed function
   * \param name The file name
   * \return The full path of file.
   */
  std::string GetPath(const std::string& file_name);
  /*!
   * \brief Remove The RPC Environment cleanup function
   */
  void Remove();

  /*!
   * \base_ Holds the environment path.
   */
  std::string base_;
};  // RPCEnv

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_APPS_CPP_RPC_ENV_H_