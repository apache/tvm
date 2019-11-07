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
#ifndef TVM_RUNTIME_CONTRIB_GCC_GCC_H_
#define TVM_RUNTIME_CONTRIB_GCC_GCC_H_

#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <string>

#include "../external_util.h"
#include "../../dso_module.h"

namespace tvm {
namespace runtime {
namespace contrib {

constexpr const char* kGccPrefix = "gcc_";

/*!
 * \brief Defined a data structure to save subgraph args.
 */
typedef struct {
  float** data;
} GccPackedArgs;

typedef void (*GccSubgraphFunc)(GccPackedArgs in, float* out);

class GccModuleNode : public DSOModuleNode {
 public:
  const char* type_key() const final {
    return "GccModule";
  }

  runtime::PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final;
};

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_GCC_GCC_H_
