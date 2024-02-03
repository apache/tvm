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
 * \file src/contrib/msc/framework/torch/codegen_utils.h
 * \brief Utils for torch codegen.
 */
#ifndef TVM_CONTRIB_MSC_FRAMEWORK_TORCH_CODEGEN_UTILS_H_
#define TVM_CONTRIB_MSC_FRAMEWORK_TORCH_CODEGEN_UTILS_H_

#include <string>

#include "../../core/codegen/base_codegen.h"
#include "../../core/codegen/codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

/*!
 * \brief CodeGen helper for torch codegen
 */
class TorchCodeGenHelper : public BaseCodeGenHelper {
 public:
  /*! \brief Get describe for default node input*/
  const String IdxOutputBase(const MSCJoint& node, const String& prefix = "", int idx = 0,
                             const String& suffix = "", bool mark_exit = false) final {
    if ((node->optype == "max" || node->optype == "min") && node->OutputAt(0)->Ndim() > 0) {
      ICHECK(idx == 0) << "max and min op only support 1 outputs, get " << node;
      return IdxNodeBase(node, prefix, suffix) + ".values";
    }
    return BaseCodeGenHelper::IdxOutputBase(node, prefix, idx, suffix, mark_exit);
  }
};

/*!
 * \brief CodeGen config for torch codegen
 */
struct TorchCodeGenConfig {
  CODEGEN_CONFIG_MEMBERS
  void Load(dmlc::JSONReader* reader) {
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      CODEGEN_CONFIG_PARSE
    }
  }
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_FRAMEWORK_TORCH_CODEGEN_UTILS_H_
