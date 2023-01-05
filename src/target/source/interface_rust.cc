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
 * \file interface_rust.cc
 * \brief Generates a Rust interface header for a given modules inputs and outputs
 * which works on top of the C interface API
 */

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/name_transforms.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/usmp/utils.h>

#include <string>

#include "../../relay/backend/name_transforms.h"
#include "codegen_params.h"

namespace tvm {
namespace codegen {

using runtime::PackedFunc;
using namespace tvm::relay::backend;
using namespace tvm::runtime;

class InterfaceRustNode : public runtime::ModuleNode {
 public:
  InterfaceRustNode(std::string module_name, Array<String> input_names, Array<String> output_names,
                    Array<tir::usmp::AllocatedPoolInfo> pools,
                    Map<String, tir::usmp::PoolAllocation> io_pool_allocations,
                    Array<String> devices, int workspace_size,
                    Map<String, Map<String, ObjectRef>> inputs,
                    Map<String, Map<String, ObjectRef>> outputs)
      : module_name_(module_name),
        inputs_(inputs),
        outputs_(outputs),
        devices_(devices),
        input_names_(input_names),
        output_names_(output_names),
        pools_(FilterExternalPools(pools)),
        io_pool_allocations_(io_pool_allocations),
        workspace_size_(workspace_size) {
    ICHECK(io_pool_allocations_.empty()) << "Workspace Memory Pools IO unsupported";
  }
  const char* type_key() const final { return "h"; }

  std::string GetSource(const std::string& format) final {
    std::stringstream code;

    EmitBrief(code, "Input tensors");
    EmitDataStruct(code, "inputs", inputs_, input_names_);
    EmitBrief(code, "Output tensors");
    EmitDataStruct(code, "outputs", outputs_, output_names_);

    if (!devices_.empty()) {
      EmitBrief(code, "Device context pointers");
      EmitDeviceStruct(code, "devices", devices_);
    }

    if (!pools_.empty()) {
      EmitBrief(code, "Workspace pools");
      EmitWorkspacePoolsStruct(code);
    }

    EmitRustRunFunction(code);
    EmitCRunFunction(code);

    EmitIntegerValueConst(code, "Workspace size", "WORKSPACE_SIZE", workspace_size_);
    EmitMemoryPools(code);

    return code.str();
  }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    return PackedFunc();
  }

 private:
  constexpr static const char* _macro_workspace_pool_size_postfix = "_WORKSPACE_POOL_SIZE";
  constexpr static const char* _macro_constant_pool_size_postfix = "_CONSTANT_POOL_SIZE";
  constexpr static const char* _macro_constant_pool_data_postfix = "_constant_pool_data";

  void EmitBrief(std::stringstream& code_stream, const std::string& description) {
    code_stream << "/// " << description << " for TVM module \"" << module_name_ << "\"\n";
  }

  void EmitMemoryPools(std::stringstream& code) {
    for (const tir::usmp::AllocatedPoolInfo pool : pools_) {
      String pool_name = pool->pool_info->pool_name;
      Integer pool_size = pool->allocated_size;
      if (const auto* pool_info = pool->pool_info.as<ConstantPoolInfoNode>()) {
        EmitConstantPool(code, SanitizeName(pool_name) + " initialization data", pool_info);
      } else {
        EmitIntegerValueConst(code, SanitizeName(pool_name) + " size",
                              SanitizeName(pool_name) + _macro_workspace_pool_size_postfix,
                              pool_size->value);
      }
    }
  }

  void EmitConstantPool(std::stringstream& code_, const std::string& brief_description,
                        const ConstantPoolInfoNode* pool_info) {
    EmitBrief(code_, brief_description);
    std::string const_name_prefixed = ToRustConstantStyle(SanitizeName(pool_info->pool_name));
    std::string macro_name_prefixed = ToRustMacroStyle(SanitizeName(pool_info->pool_name));

    if (pool_info->constant_info_array.size() > 0) {
      std::vector<ConstantInfo> const_info_vec(pool_info->constant_info_array.begin(),
                                               pool_info->constant_info_array.end());
      std::sort(const_info_vec.begin(), const_info_vec.end(),
                [](const ConstantInfo& a, const ConstantInfo& b) {
                  return a->byte_offset->value < b->byte_offset->value;
                });
      int64_t accumulated_pool_len =
          const_info_vec.back()->byte_offset.IntValue() +
          runtime::GetDataSize(*const_info_vec.back()->data.operator->());
      const auto& accumulated_pool = runtime::NDArray::Empty(
          {accumulated_pool_len}, DataType::UInt(8), const_info_vec.back()->data->device);
      for (const auto& const_info : const_info_vec) {
        const auto& data = const_info->data;
        const auto& offs = const_info->byte_offset;
        data.CopyToBytes(static_cast<uint8_t*>(accumulated_pool->data) + offs.IntValue(),
                         runtime::GetDataSize(*data.operator->()));
      }

      code_ << "pub const " << const_name_prefixed << _macro_constant_pool_size_postfix
            << ": u32 = " << accumulated_pool_len << ";\n";
      code_ << "#[macro_export]\n"
            << "macro_rules! " << macro_name_prefixed << _macro_constant_pool_data_postfix << " {\n"
            << "    () => {[\n";
      codegen::NDArrayDataToC(accumulated_pool, 8, code_, "\\\n");
      code_ << "    ]};\n";
      code_ << "}\n";
    } else {
      LOG(FATAL) << "No constant data in constant pool found "
                 << PrettyPrint(GetRef<ObjectRef>(pool_info));
    }
  }

  void EmitStruct(std::stringstream& code_stream, const std::string& struct_name,
                  std::unordered_map<std::string, std::pair<std::string, std::string>> properties,
                  std::vector<std::string> property_names_ordered) {
    std::unordered_map<std::string, std::pair<std::string, std::string>> sanitized_properties;
    std::vector<std::string> sanitized_property_names_ordered;

    for (const std::string& property_name : property_names_ordered) {
      std::string sanitized_property = SanitizeName(property_name);
      std::pair<std::string, std::string> property_values = properties.at(property_name);
      ICHECK(std::find(sanitized_property_names_ordered.begin(),
                       sanitized_property_names_ordered.end(),
                       sanitized_property) == sanitized_property_names_ordered.end())
          << "Sanitized input tensor name clash" << sanitized_property;

      sanitized_properties.emplace(sanitized_property, property_values);
      sanitized_property_names_ordered.push_back(sanitized_property);
    }
    std::reverse(sanitized_property_names_ordered.begin(), sanitized_property_names_ordered.end());

    code_stream << "#[repr(C)]\n";
    code_stream << "pub struct " << ToRustStructStyle(struct_name) << " {\n";
    for (const std::string& property_name : sanitized_property_names_ordered) {
      code_stream << "    " << property_name << ": *mut ::std::os::raw::c_void,\n";
    }
    code_stream << "}\n\n"
                << "impl " << ToRustStructStyle(struct_name) << " {\n"
                << "    pub fn new <'a>(\n";
    for (const std::string& property_name : sanitized_property_names_ordered) {
      std::string rust_data_type = sanitized_properties.at(property_name).first;
      code_stream << "        " << property_name << ": " << rust_data_type << ",\n";
    }
    code_stream << "    ) -> Self {\n"
                << "        Self {\n";
    for (const std::string& property_name : sanitized_property_names_ordered) {
      std::string struct_conversion = sanitized_properties.at(property_name).second;
      code_stream << "            " << property_name << ": " << property_name << struct_conversion
                  << ",\n";
    }
    code_stream << "        }\n"
                << "    }\n"
                << "}\n";
  }

  void EmitWorkspacePoolsStruct(std::stringstream& code_stream) {
    std::unordered_map<std::string, std::pair<std::string, std::string>> struct_properties;
    std::vector<std::string> property_names_ordered;
    for (const tir::usmp::AllocatedPoolInfo pool : pools_) {
      int64_t allocated_size = pool->allocated_size.IntValue();
      std::string rust_type = "&mut [u8; " + std::to_string(allocated_size) + "]";
      struct_properties.emplace(
          pool->pool_info->pool_name,
          std::make_pair(rust_type, ".as_ptr() as *mut ::std::os::raw::c_void"));
      property_names_ordered.push_back(pool->pool_info->pool_name);
    }
    std::reverse(property_names_ordered.begin(), property_names_ordered.end());

    EmitStruct(code_stream, "workspace_pools", struct_properties, property_names_ordered);
  }

  std::string DTypeToRust(std::string dtype) {
    std::string width;
    std::copy_if(dtype.begin(), dtype.end(), std::back_inserter(width), ::isdigit);
    return dtype[0] + width;
  }

  std::string NumElements(std::string dtype, int64_t size) {
    std::string width;
    std::copy_if(dtype.begin(), dtype.end(), std::back_inserter(width), ::isdigit);
    return std::to_string(size * 8 / std::stoi(width));
  }

  void EmitDataStruct(std::stringstream& code_stream, const std::string& struct_name,
                      Map<String, Map<String, ObjectRef>> properties,
                      Array<String> property_names) {
    std::unordered_map<std::string, std::pair<std::string, std::string>> struct_properties;
    for (const auto& property : properties) {
      Map<String, ObjectRef> values = property.second;
      std::string dtype = Downcast<String>(values.Get("dtype"));
      int64_t size = Downcast<Integer>(values.Get("size")).IntValue();
      std::string rust_dtype = DTypeToRust(dtype);
      std::string num_elements = NumElements(dtype, size);

      struct_properties.emplace(property.first,
                                std::make_pair("&mut [" + rust_dtype + "; " + num_elements + "]",
                                               ".as_ptr() as *mut ::std::os::raw::c_void"));
    }

    std::vector<std::string> property_names_ordered;
    for (const String& property_name : property_names) {
      property_names_ordered.push_back(property_name);
    }
    std::reverse(property_names_ordered.begin(), property_names_ordered.end());

    EmitStruct(code_stream, struct_name, struct_properties, property_names_ordered);
  }

  void EmitDeviceStruct(std::stringstream& code_stream, const std::string& struct_name,
                        Array<String> devices) {
    std::unordered_map<std::string, std::pair<std::string, std::string>> struct_properties;
    std::vector<std::string> property_names_ordered;
    for (const auto& device : devices) {
      struct_properties.emplace(device, std::make_pair("*mut ::std::os::raw::c_void", ""));
      property_names_ordered.push_back(device);
    }
    std::reverse(property_names_ordered.begin(), property_names_ordered.end());

    EmitStruct(code_stream, struct_name, struct_properties, property_names_ordered);
  }

  void EmitIntegerValueConst(std::stringstream& code_stream, const std::string& brief_description,
                             const std::string& macro_name, int macro_value) {
    EmitBrief(code_stream, brief_description);
    std::string macro_name_prefixed = ToRustConstantStyle(macro_name);
    code_stream << "pub const " << macro_name_prefixed << ": usize = " << macro_value << ";\n";
  }

  void EmitCRunFunction(std::stringstream& code_stream) {
    std::string run_function = ToCVariableStyle(PrefixGeneratedName({module_name_, "run"}));
    code_stream << "extern \"C\" {\n"
                << "    pub fn " << run_function << "(\n"
                << "        inputs: *mut Inputs,\n"
                << "        outputs: *mut Outputs,\n";
    if (!pools_.empty()) {
      code_stream << "        workspace_pools: *mut WorkspacePools,\n";
    }
    if (!devices_.empty()) {
      code_stream << "        devices: *mut Devices,\n";
    }
    code_stream << "    ) -> i32;\n"
                << "}\n";
  }

  void EmitRustRunFunction(std::stringstream& code_stream) {
    std::string run_function = ToCVariableStyle(PrefixGeneratedName({module_name_, "run"}));
    code_stream << "/// Entrypoint function for TVM module \"" << module_name_ << "\"\n"
                << "/// # Arguments\n";
    if (io_pool_allocations_.empty()) {
      code_stream << "/// * `inputs` - Input tensors for the module\n";
      code_stream << "/// * `outputs` - Output tensors for the module\n";
    }

    if (!pools_.empty()) {
      code_stream << "/// * `workspace_pools` - Workspace memory pools for the module\n";
    }
    if (!devices_.empty()) {
      code_stream << "/// * `devices` - Device context pointers for the module\n";
    }
    code_stream << "pub fn run(\n"
                << "    inputs: &mut Inputs,\n"
                << "    outputs: &mut Outputs,\n";
    if (!pools_.empty()) {
      code_stream << "    workspace_pools: &mut WorkspacePools,\n";
    }
    if (!devices_.empty()) {
      code_stream << "    devices: &mut Devices,\n";
    }
    code_stream << ") -> Result<(), ()> {\n"
                << "    unsafe {\n"
                << "        let ret = " << run_function << "(\n"
                << "            inputs,\n"
                << "            outputs,\n";
    if (!pools_.empty()) {
      code_stream << "            workspace_pools,\n";
    }
    if (!devices_.empty()) {
      code_stream << "            devices,\n";
    }
    code_stream << "        );\n"
                << "        if ret == 0 {\n"
                << "            Ok(())\n"
                << "        } else {\n"
                << "            Err(())\n"
                << "        }\n"
                << "    }\n"
                << "}\n\n";
  }

  void EmitRunFunction(std::stringstream& code_stream) {
    EmitRustRunFunction(code_stream);
    EmitCRunFunction(code_stream);
  }

  Array<tir::usmp::AllocatedPoolInfo> FilterExternalPools(
      const Array<tir::usmp::AllocatedPoolInfo>& pools) {
    Array<tir::usmp::AllocatedPoolInfo> external_pools;
    for (tir::usmp::AllocatedPoolInfo pool : pools) {
      if (!pool->pool_info->is_internal) {
        external_pools.push_back(pool);
      }
    }
    return external_pools;
  }

  std::string module_name_;
  Map<String, Map<String, ObjectRef>> inputs_;
  Map<String, Map<String, ObjectRef>> outputs_;
  Array<String> devices_;
  Array<String> input_names_;
  Array<String> output_names_;
  Array<tir::usmp::AllocatedPoolInfo> pools_;
  Map<String, tir::usmp::PoolAllocation> io_pool_allocations_;
  int workspace_size_;
};  // namespace codegen

runtime::Module InterfaceRustCreate(std::string module_name,
                                    Map<String, Map<String, ObjectRef>> inputs,
                                    Map<String, Map<String, ObjectRef>> outputs,
                                    Array<tir::usmp::AllocatedPoolInfo> pools,
                                    Map<String, tir::usmp::PoolAllocation> io_pool_allocations,
                                    Array<String> devices, Array<String> input_names,
                                    Array<String> output_names, int workspace_size) {
  auto n =
      make_object<InterfaceRustNode>(module_name, input_names, output_names, pools,
                                     io_pool_allocations, devices, workspace_size, inputs, outputs);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.InterfaceRustCreate").set_body_typed(InterfaceRustCreate);

}  // namespace codegen
}  // namespace tvm
