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
 * \file src/contrib/msc/framework/tensorflow/codegen_utils.h
 * \brief Utils for tensorflow codegen.
 */
#ifndef TVM_CONTRIB_MSC_FRAMEWORK_TENSORFLOW_CODEGEN_UTILS_H_
#define TVM_CONTRIB_MSC_FRAMEWORK_TENSORFLOW_CODEGEN_UTILS_H_

#include <string>

#include "../../core/codegen/base_codegen.h"
#include "../../core/codegen/codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

/*!
 * \brief CodeGen helper for tensorrt codegen
 */
class TFV1CodeGenHelper : public BaseCodeGenHelper {};

/*!
 * \brief CodeGen config for tensorflow codegen
 */
struct TensorflowCodeGenConfig {
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
#endif  // TVM_CONTRIB_MSC_FRAMEWORK_TENSORFLOW_CODEGEN_UTILS_H_
