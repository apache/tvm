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
 *  Copyright (c) 2018 by Contributors
 * \file relay/backend/utils.h
 * \brief Utils function for backend
 */
#ifndef TVM_RELAY_BACKEND_UTILS_H_
#define TVM_RELAY_BACKEND_UTILS_H_

#include <dmlc/json.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/type.h>
#include <tvm/build_module.h>
#include <tvm/codegen.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>

#include <typeinfo>
#include <string>

namespace tvm {
namespace relay {
namespace backend {
/*!
 * \brief Get the Packed Func
 *
 * \param func_name
 * \return const PackedFunc*
 */
inline const PackedFunc* GetPackedFunc(const std::string& func_name) {
  return tvm::runtime::Registry::Get(func_name);
}
/*!
 * \brief Convert type to string
 *
 * \param typ
 * \return std::string string format of type
 */
inline std::string DType2String(const tvm::Type typ) {
  std::ostringstream os;
  auto tvm_type = Type2TVMType(typ);
  if (tvm_type.code == kDLFloat) {
    os << "float";
  } else if (tvm_type.code == kDLInt) {
    os << "int";
  } else if (tvm_type.code == kDLUInt) {
    os << "uint";
  } else {
    LOG(FATAL) << "Unknown type";
  }
  os << typ.bits();
  return os.str();
}

}  // namespace backend
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_UTILS_H_
