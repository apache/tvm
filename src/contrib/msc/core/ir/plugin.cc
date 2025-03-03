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
 * \file src/contrib/msc/core/ir/plugin.cc
 */

#include "plugin.h"

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <utility>

namespace tvm {
namespace contrib {
namespace msc {

PluginAttr::PluginAttr(const String& name, const String& type, const String& default_value,
                       const String& describe) {
  ObjectPtr<PluginAttrNode> n = make_object<PluginAttrNode>();
  n->name = std::move(name);
  n->type = std::move(type);
  n->default_value = std::move(default_value);
  n->describe = std::move(describe);
  data_ = std::move(n);
}

PluginAttr::PluginAttr(const JsonPluginAttr& j_attr) {
  ObjectPtr<PluginAttrNode> n = make_object<PluginAttrNode>();
  n->FromJson(j_attr);
  data_ = std::move(n);
}

PluginAttr::PluginAttr(const std::string& json_str) {
  ObjectPtr<PluginAttrNode> n = make_object<PluginAttrNode>();
  n->FromJson(json_str);
  data_ = std::move(n);
}

const JsonPluginAttr PluginAttrNode::ToJson() const {
  JsonPluginAttr j_attr;
  j_attr.name = name;
  j_attr.type = type;
  j_attr.default_value = default_value;
  j_attr.describe = describe;
  return j_attr;
}

void PluginAttrNode::FromJson(const JsonPluginAttr& j_attr) {
  name = j_attr.name;
  type = j_attr.type;
  default_value = j_attr.default_value;
  describe = j_attr.describe;
}

void PluginAttrNode::FromJson(const std::string& json_str) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JsonPluginAttr j_attr;
  reader.Read(&j_attr);
  FromJson(j_attr);
}

PluginTensor::PluginTensor(const String& name, const String& dtype, const Integer& ndim,
                           const String& device, const String& describe) {
  ObjectPtr<PluginTensorNode> n = make_object<PluginTensorNode>();
  n->name = std::move(name);
  n->dtype = std::move(dtype);
  n->ndim = std::move(ndim);
  n->device = std::move(device);
  n->describe = std::move(describe);
  data_ = std::move(n);
}

PluginTensor::PluginTensor(const JsonPluginTensor& j_tensor) {
  ObjectPtr<PluginTensorNode> n = make_object<PluginTensorNode>();
  n->FromJson(j_tensor);
  data_ = std::move(n);
}

PluginTensor::PluginTensor(const std::string& json_str) {
  ObjectPtr<PluginTensorNode> n = make_object<PluginTensorNode>();
  n->FromJson(json_str);
  data_ = std::move(n);
}

const JsonPluginTensor PluginTensorNode::ToJson() const {
  JsonPluginTensor j_tensor;
  j_tensor.name = name;
  j_tensor.dtype = dtype;
  j_tensor.ndim = ndim->value;
  j_tensor.device = device;
  j_tensor.describe = describe;
  return j_tensor;
}

void PluginTensorNode::FromJson(const JsonPluginTensor& j_tensor) {
  name = j_tensor.name;
  dtype = j_tensor.dtype;
  ndim = Integer(j_tensor.ndim);
  device = j_tensor.device;
  describe = j_tensor.describe;
}

void PluginTensorNode::FromJson(const std::string& json_str) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JsonPluginTensor j_tensor;
  reader.Read(&j_tensor);
  FromJson(j_tensor);
}

PluginExtern::PluginExtern(const String& name, const String& header, const String& source,
                           const String& lib, const String& describe) {
  ObjectPtr<PluginExternNode> n = make_object<PluginExternNode>();
  n->name = std::move(name);
  n->header = std::move(header);
  n->source = std::move(source);
  n->lib = std::move(lib);
  n->describe = std::move(describe);
  data_ = std::move(n);
}

PluginExtern::PluginExtern(const JsonPluginExtern& j_extern) {
  ObjectPtr<PluginExternNode> n = make_object<PluginExternNode>();
  n->FromJson(j_extern);
  data_ = std::move(n);
}

PluginExtern::PluginExtern(const std::string& json_str) {
  ObjectPtr<PluginExternNode> n = make_object<PluginExternNode>();
  n->FromJson(json_str);
  data_ = std::move(n);
}

const JsonPluginExtern PluginExternNode::ToJson() const {
  JsonPluginExtern j_extern;
  j_extern.name = name;
  j_extern.header = header;
  j_extern.source = source;
  j_extern.lib = lib;
  j_extern.describe = describe;
  return j_extern;
}

void PluginExternNode::FromJson(const JsonPluginExtern& j_extern) {
  name = j_extern.name;
  header = j_extern.header;
  source = j_extern.source;
  lib = j_extern.lib;
  describe = j_extern.describe;
}

void PluginExternNode::FromJson(const std::string& json_str) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JsonPluginExtern j_extern;
  reader.Read(&j_extern);
  FromJson(j_extern);
}

Plugin::Plugin(const String& name, const String& version, const String& describe,
               const Array<PluginAttr>& attrs, const Array<PluginTensor>& inputs,
               const Array<PluginTensor>& outputs, const Array<PluginTensor>& buffers,
               const Map<String, PluginExtern>& externs,
               const Map<String, Array<String>>& support_dtypes,
               const Map<String, String>& options) {
  ObjectPtr<PluginNode> n = make_object<PluginNode>();
  n->name = std::move(name);
  n->version = std::move(version);
  n->describe = std::move(describe);
  n->attrs = std::move(attrs);
  n->inputs = std::move(inputs);
  n->outputs = std::move(outputs);
  n->buffers = std::move(buffers);
  n->externs = std::move(externs);
  n->support_dtypes = std::move(support_dtypes);
  n->options = std::move(options);
  data_ = std::move(n);
}

Plugin::Plugin(const JsonPlugin& j_plugin) {
  ObjectPtr<PluginNode> n = make_object<PluginNode>();
  n->FromJson(j_plugin);
  data_ = std::move(n);
}

Plugin::Plugin(const std::string& json_str) {
  ObjectPtr<PluginNode> n = make_object<PluginNode>();
  n->FromJson(json_str);
  data_ = std::move(n);
}

const JsonPlugin PluginNode::ToJson() const {
  JsonPlugin j_plugin;
  j_plugin.name = name;
  j_plugin.version = version;
  j_plugin.describe = describe;
  for (const auto& a : attrs) {
    j_plugin.attrs.push_back(a->ToJson());
  }
  for (const auto& t : inputs) {
    j_plugin.inputs.push_back(t->ToJson());
  }
  for (const auto& t : outputs) {
    j_plugin.inputs.push_back(t->ToJson());
  }
  for (const auto& t : buffers) {
    j_plugin.inputs.push_back(t->ToJson());
  }
  for (const auto& pair : externs) {
    j_plugin.externs[pair.first] = pair.second->ToJson();
  }
  for (const auto& pair : support_dtypes) {
    std::vector<std::string> dtypes;
    for (const auto& d : pair.second) {
      dtypes.push_back(d);
    }
    j_plugin.support_dtypes[pair.first] = dtypes;
  }
  for (const auto& pair : options) {
    j_plugin.options[pair.first] = pair.second;
  }
  return j_plugin;
}

void PluginNode::FromJson(const JsonPlugin& j_plugin) {
  name = j_plugin.name;
  version = j_plugin.version;
  describe = j_plugin.describe;
  for (const auto& a : j_plugin.attrs) {
    attrs.push_back(PluginAttr(a));
  }
  for (const auto& t : j_plugin.inputs) {
    inputs.push_back(PluginTensor(t));
  }
  for (const auto& t : j_plugin.outputs) {
    outputs.push_back(PluginTensor(t));
  }
  for (const auto& t : j_plugin.buffers) {
    buffers.push_back(PluginTensor(t));
  }
  for (const auto& pair : j_plugin.externs) {
    externs.Set(pair.first, PluginExtern(pair.second));
  }
  for (const auto& pair : j_plugin.support_dtypes) {
    Array<String> dtypes;
    for (const auto& d : pair.second) {
      dtypes.push_back(d);
    }
    support_dtypes.Set(pair.first, dtypes);
  }
  for (const auto& pair : j_plugin.options) {
    options.Set(pair.first, pair.second);
  }
}

void PluginNode::FromJson(const std::string& json_str) {
  std::istringstream is(json_str);
  dmlc::JSONReader reader(&is);
  JsonPlugin j_plugin;
  reader.Read(&j_plugin);
  FromJson(j_plugin);
}

int PluginNode::FindDtypeRefIdx(const PluginTensor& tensor) const {
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i]->dtype == tensor->dtype) {
      return i;
    }
  }
  return -1;
}

int PluginNode::FindDeviceRefIdx(const PluginTensor& tensor) const {
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i]->device == tensor->device) {
      return i;
    }
  }
  return -1;
}

const Array<String> ListPluginNames() { return PluginRegistry::Global()->ListAllNames(); }

const Plugin GetPlugin(const String& name) { return PluginRegistry::Global()->Get(name); }

bool IsPlugin(const String& name) { return PluginRegistry::Global()->Registered(name); }

TVM_REGISTER_GLOBAL("msc.core.RegisterPlugin")
    .set_body_typed([](const String& name, const String& json_str) {
      PluginRegistry::Global()->Register(name, json_str);
    });

TVM_REGISTER_GLOBAL("msc.core.ListPluginNames").set_body_typed([]() -> Array<String> {
  return ListPluginNames();
});

TVM_REGISTER_GLOBAL("msc.core.GetPlugin").set_body_typed([](const String& name) -> Plugin {
  return GetPlugin(name);
});

TVM_REGISTER_GLOBAL("msc.core.IsPlugin").set_body_typed([](const String& name) -> Bool {
  return Bool(IsPlugin(name));
});

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
