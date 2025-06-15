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
#include "registry.h"

#include <tvm/ffi/function.h>
#include <tvm/runtime/data_type.h>

namespace tvm {
namespace datatype {

using ffi::Any;
using ffi::PackedArgs;

TVM_FFI_REGISTER_GLOBAL("dtype.register_custom_type")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* ret) {
      datatype::Registry::Global()->Register(args[0].cast<std::string>(),
                                             static_cast<uint8_t>(args[1].cast<int>()));
    });

TVM_FFI_REGISTER_GLOBAL("dtype.get_custom_type_code")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* ret) {
      *ret = datatype::Registry::Global()->GetTypeCode(args[0].cast<std::string>());
    });

TVM_FFI_REGISTER_GLOBAL("dtype.get_custom_type_name")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* ret) {
      *ret = Registry::Global()->GetTypeName(args[0].cast<int>());
    });

TVM_FFI_REGISTER_GLOBAL("runtime._datatype_get_type_registered")
    .set_body_packed([](ffi::PackedArgs args, ffi::Any* ret) {
      *ret = Registry::Global()->GetTypeRegistered(args[0].cast<int>());
    });

Registry* Registry::Global() {
  static Registry inst;
  return &inst;
}

void Registry::Register(const std::string& type_name, uint8_t type_code) {
  ICHECK(type_code >= DataType::kCustomBegin)
      << "Please choose a type code >= DataType::kCustomBegin for custom types";
  code_to_name_[type_code] = type_name;
  name_to_code_[type_name] = type_code;
}

uint8_t Registry::GetTypeCode(const std::string& type_name) {
  ICHECK(name_to_code_.find(type_name) != name_to_code_.end())
      << "Type name " << type_name << " not registered";
  return name_to_code_[type_name];
}

std::string Registry::GetTypeName(uint8_t type_code) {
  ICHECK(code_to_name_.find(type_code) != code_to_name_.end())
      << "Type code " << static_cast<unsigned>(type_code) << " not registered";
  return code_to_name_[type_code];
}

std::optional<tvm::ffi::Function> GetCastLowerFunc(const std::string& target, uint8_t type_code,
                                                   uint8_t src_type_code) {
  std::ostringstream ss;
  ss << "tvm.datatype.lower.";
  ss << target << ".";
  ss << "Cast"
     << ".";

  if (datatype::Registry::Global()->GetTypeRegistered(type_code)) {
    ss << datatype::Registry::Global()->GetTypeName(type_code);
  } else {
    ss << ffi::details::DLDataTypeCodeAsCStr(static_cast<DLDataTypeCode>(type_code));
  }

  ss << ".";

  if (datatype::Registry::Global()->GetTypeRegistered(src_type_code)) {
    ss << datatype::Registry::Global()->GetTypeName(src_type_code);
  } else {
    ss << ffi::details::DLDataTypeCodeAsCStr(static_cast<DLDataTypeCode>(src_type_code));
  }
  return tvm::ffi::Function::GetGlobal(ss.str());
}

std::optional<tvm::ffi::Function> GetMinFunc(uint8_t type_code) {
  std::ostringstream ss;
  ss << "tvm.datatype.min.";
  ss << datatype::Registry::Global()->GetTypeName(type_code);
  return tvm::ffi::Function::GetGlobal(ss.str());
}

std::optional<tvm::ffi::Function> GetFloatImmLowerFunc(const std::string& target,
                                                       uint8_t type_code) {
  std::ostringstream ss;
  ss << "tvm.datatype.lower.";
  ss << target;
  ss << ".FloatImm.";
  ss << datatype::Registry::Global()->GetTypeName(type_code);
  return tvm::ffi::Function::GetGlobal(ss.str());
}

std::optional<tvm::ffi::Function> GetIntrinLowerFunc(const std::string& target,
                                                     const std::string& name, uint8_t type_code) {
  std::ostringstream ss;
  ss << "tvm.datatype.lower.";
  ss << target;
  ss << ".Call.intrin.";
  ss << name;
  ss << ".";
  ss << datatype::Registry::Global()->GetTypeName(type_code);
  return tvm::ffi::Function::GetGlobal(ss.str());
}

uint64_t ConvertConstScalar(uint8_t type_code, double value) {
  std::ostringstream ss;
  ss << "tvm.datatype.convertconstscalar.float.";
  ss << datatype::Registry::Global()->GetTypeName(type_code);
  auto make_const_scalar_func = tvm::ffi::Function::GetGlobal(ss.str());
  return (*make_const_scalar_func)(value).cast<uint64_t>();
}

}  // namespace datatype
}  // namespace tvm
