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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/relax/analysis.h>
#include <tvm/script/ir_builder/ir/ir.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace ir {

IRModuleFrame IRModule() {
  ObjectPtr<IRModuleFrameNode> n = ffi::make_object<IRModuleFrameNode>();
  n->global_var_map.clear();
  n->functions.clear();
  return IRModuleFrame(n);
}

inline relax::StructInfo GetGlobalVarStructInfo(const BaseFunc& func) {
  if (func->struct_info_.defined()) {
    return tvm::relax::GetStructInfo(func);
  } else if (const auto* prim_func = func.as<tvm::tir::PrimFuncNode>()) {
    return tvm::relax::FuncStructInfo::OpaqueFunc(
        tvm::relax::StructInfoFromType(prim_func->ret_type));
  } else {
    LOG(FATAL) << "Unsupported function type: " << func->GetTypeKey();
  }
}

GlobalVar DeclFunction(const ffi::String& func_name, const BaseFunc& func_signature) {
  IRModuleFrame frame = FindModuleFrame();
  CHECK(!frame->global_var_map.count(func_name))
      << "ValueError: function " << func_name << " already exists";

  auto gvar_type = [&]() -> Type {
    if (auto prim_func = func_signature.as<tir::PrimFuncNode>()) {
      ffi::Array<Type> arg_types =
          prim_func->params.Map([](const auto& var) { return GetType(var); });
      return FuncType(arg_types, prim_func->ret_type);
    }

    return {};
  }();

  GlobalVar gv = GlobalVar(func_name);
  gv->struct_info_ = GetGlobalVarStructInfo(func_signature);
  CHECK(frame->functions.find(gv) == frame->functions.end())
      << "ValueError: function " << func_name << " has already been defined.";
  frame->global_var_map.Set(func_name, gv);
  frame->functions.Set(gv, func_signature);
  return gv;
}

void DefFunction(const ffi::String& func_name, const BaseFunc& func) {
  IRModuleFrame frame = FindModuleFrame();
  auto it = frame->global_var_map.find(func_name);
  CHECK(it != frame->global_var_map.end())
      << "ValueError: function " << func_name << " does not exist, please declare it first.";
  const GlobalVar& gv = (*it).second;
  frame->functions.Set(gv, func);
  gv->struct_info_ = GetGlobalVarStructInfo(func);
}

void ModuleAttrs(ffi::Map<ffi::String, Any> attrs, bool allow_overwrite) {
  if (IRBuilder::IsInScope()) {
    // TODO(hongyi): add comments to explain why we need to check if the module frame is in scope
    IRModuleFrame frame = FindModuleFrame("I.ModuleAttr");
    if (!allow_overwrite && !frame->attrs.empty()) {
      LOG(FATAL) << "ValueError: Duplicate module attrs, previous one is:\n" << frame->attrs;
    }
    frame->attrs = attrs;
  }
}

ffi::Optional<ObjectRef> ModuleGetAttr(const ffi::String& key) {
  if (IRBuilder::IsInScope()) {
    IRModuleFrame frame = FindModuleFrame();
    if (frame->attrs.find(key) != frame->attrs.end()) {
      return frame->attrs[key].cast<ObjectRef>();
    }
  }
  return std::nullopt;
}

void ModuleSetAttr(const ffi::String& key, const ffi::Optional<ObjectRef>& value,
                   bool allow_override) {
  if (IRBuilder::IsInScope()) {
    IRModuleFrame frame = FindModuleFrame();
    if (!allow_override && frame->attrs.find(key) != frame->attrs.end() && value.defined()) {
      LOG(FATAL) << "ValueError: Duplicate module attr " << key;
    }
    if (value.defined()) {
      frame->attrs.Set(key, value.value());
    } else {
      frame->attrs.erase(key);
    }
  } else {
    LOG(FATAL) << "ValueError: Currently in in the scope of a module.";
  }
}

void ModuleGlobalInfos(ffi::Map<ffi::String, ffi::Array<GlobalInfo>> global_infos) {
  if (IRBuilder::IsInScope()) {
    IRModuleFrame frame = FindModuleFrame("I.ModuleGlobalInfos");
    if (!frame->global_infos.empty()) {
      LOG(FATAL) << "ValueError: Duplicate module global_infos, previous one is:\n"
                 << frame->global_infos;
    }
    frame->global_infos = global_infos;
  }
}

VDevice LookupVDevice(ffi::String target_kind, int device_index) {
  if (IRBuilder::IsInScope()) {
    IRModuleFrame frame = FindModuleFrame();
    if (frame->global_infos.empty()) {
      LOG(FATAL) << "ValueError: The GlobalInfos in the IRModule is not defined.";
    }
    ffi::Array<GlobalInfo> vdevices = frame->global_infos["vdevice"];
    if (vdevices.empty() || device_index < 0 ||
        static_cast<size_t>(device_index) >= vdevices.size()) {
      LOG(FATAL) << "ValueError: The target VDevice in the GlobalInfos was not found.";
    }
    if (target_kind == "vdevice") {
      return Downcast<VDevice>(vdevices[device_index]);
    }
    int count = 0;
    for (auto vdevice : vdevices) {
      auto vdev = Downcast<VDevice>(vdevice);
      if (vdev->target->kind->name == target_kind) {
        if (count == device_index) {
          return vdev;
        }
        count++;
      }
    }
  }
  LOG(WARNING) << "The annotated device was not found, please check your vdevice list.";
  return VDevice();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("script.ir_builder.ir.IRModule", IRModule)
      .def("script.ir_builder.ir.DeclFunction", DeclFunction)
      .def("script.ir_builder.ir.DefFunction", DefFunction)
      .def("script.ir_builder.ir.ModuleAttrs", ModuleAttrs)
      .def("script.ir_builder.ir.ModuleGetAttr", ModuleGetAttr)
      .def("script.ir_builder.ir.ModuleSetAttr", ModuleSetAttr)
      .def("script.ir_builder.ir.ModuleGlobalInfos", ModuleGlobalInfos)
      .def("script.ir_builder.ir.LookupVDevice", LookupVDevice);
}

}  // namespace ir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
