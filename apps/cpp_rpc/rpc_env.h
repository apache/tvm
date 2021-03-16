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
 * \file rpc_env.h
 * \brief Server environment of the RPC.
 */
#ifndef TVM_APPS_CPP_RPC_ENV_H_
#define TVM_APPS_CPP_RPC_ENV_H_

#include <tvm/runtime/registry.h>

#include <string>

namespace tvm {
namespace runtime {

/*!
 * \brief RPCEnv The RPC Environment parameters for c++ rpc server
 */
struct RPCEnv {
 public:
  /*!
   * \brief Constructor Init The RPC Environment initialize function
   */
  RPCEnv(const std::string& word_dir = "");
  /*!
   * \brief GetPath To get the workpath from packed function
   * \param name The file name
   * \return The full path of file.
   */
  std::string GetPath(const std::string& file_name) const;
  /*!
   * \brief The RPC Environment cleanup function
   */
  void CleanUp() const;

 private:
  /*!
   * \brief Holds the environment path.
   */
  std::string base_;
};  // RPCEnv

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_APPS_CPP_RPC_ENV_H_
