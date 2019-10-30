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
#ifndef TVM_RUNTIME_CONTRIB_DNNL_DNNL_H_
#define TVM_RUNTIME_CONTRIB_DNNL_DNNL_H_

#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <string>
#include "../extern_common.h"

namespace tvm {
namespace runtime {
namespace contrib {

/*!
 * \brief Defined a data structure to save dnnl subgraph args.
 */
typedef struct {
  void** data;
} DnnlPackedArgs;

constexpr const char* kDnnlPrefix = "dnnl_";

typedef void (*DnnlSubgraphFunc)(DnnlPackedArgs in, float* out);

class DNNLModule : public ExternModuleBase {
 public:
  explicit DNNLModule(const std::string& lib_path) : lib_path_(lib_path) {}

  const char* type_key() const final {
    return "DNNLModule";
  }

  runtime::PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final;

  void Init() final;

 private:
  /*! \brief The path to the compiled dnnl library.*/
  std::string lib_path_;
};

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_DNNL_DNNL_H_
