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
 * \file src/contrib/msc/core/ir/plugin.h
 * \brief Plugin describe for msc.
 */
#ifndef TVM_CONTRIB_MSC_CORE_IR_PLUGIN_H_
#define TVM_CONTRIB_MSC_CORE_IR_PLUGIN_H_

#include <dmlc/json.h>
#include <tvm/tir/data_layout.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../../../../node/attr_registry.h"
#include "../utils.h"

namespace tvm {
namespace contrib {
namespace msc {

/*!
 * \brief Json serialize and deserialize for Plugin Attribute.
 */
struct JsonPluginAttr {
  std::string name;
  std::string type;
  std::string default_value;
  std::string describe;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("type", type);
    writer->WriteObjectKeyValue("default_value", default_value);
    writer->WriteObjectKeyValue("describe", describe);
    writer->EndObject();
  }

  void Load(dmlc::JSONReader* reader) {
    int bitmask = 0;
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "name") {
        reader->Read(&name);
        bitmask |= 1;
      } else if (key == "type") {
        reader->Read(&type);
        bitmask |= 2;
      } else if (key == "default_value") {
        reader->Read(&default_value);
      } else if (key == "describe") {
        reader->Read(&describe);
      }
    }
    ICHECK_EQ(bitmask, 1 | 2) << "name and type should be given for plugin attr";
    if (describe.size() == 0) {
      describe = "Plugin attribute " + name + "(" + type + ")";
    }
  }
};

/*!
 * \brief Json serialize and deserialize for Plugin Tensor.
 */
struct JsonPluginTensor {
  std::string name;
  std::string dtype;
  int64_t ndim{-1};
  std::string device{"default"};
  std::string describe;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("dtype", dtype);
    writer->WriteObjectKeyValue("ndim", ndim);
    writer->WriteObjectKeyValue("device", device);
    writer->WriteObjectKeyValue("describe", describe);
    writer->EndObject();
  }

  void Load(dmlc::JSONReader* reader) {
    int bitmask = 0;
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "name") {
        reader->Read(&name);
        bitmask |= 1;
      } else if (key == "dtype") {
        reader->Read(&dtype);
      } else if (key == "ndim") {
        reader->Read(&ndim);
      } else if (key == "device") {
        reader->Read(&device);
      } else if (key == "describe") {
        reader->Read(&describe);
      }
    }
    ICHECK_EQ(bitmask, 1) << "name should be given for plugin tensor";
    if (describe.size() == 0) {
      describe = "Plugin tensor " + name + "(" + dtype + " on " + device + ")";
    }
  }
};

/*!
 * \brief Json serialize and deserialize for Plugin Extern.
 */
struct JsonPluginExtern {
  std::string name;
  std::string header;
  std::string source;
  std::string lib;
  std::string describe;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("header", header);
    writer->WriteObjectKeyValue("source", source);
    writer->WriteObjectKeyValue("lib", lib);
    writer->WriteObjectKeyValue("describe", describe);
    writer->EndObject();
  }

  void Load(dmlc::JSONReader* reader) {
    int bitmask = 0;
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "name") {
        reader->Read(&name);
        bitmask |= 1;
      } else if (key == "header") {
        reader->Read(&header);
      } else if (key == "source") {
        reader->Read(&source);
      } else if (key == "lib") {
        reader->Read(&lib);
      } else if (key == "describe") {
        reader->Read(&describe);
      }
    }
    ICHECK_EQ(bitmask, 1) << "name should be given for plugin extern";
    if (describe.size() == 0) {
      describe = "Plugin function " + name + "(from " + header + ")";
    }
  }
};

/*!
 * \brief Json serialize and deserialize for Plugin.
 */
struct JsonPlugin {
  std::string name;
  std::string version;
  std::string describe;
  std::vector<JsonPluginAttr> attrs;
  std::vector<JsonPluginTensor> inputs;
  std::vector<JsonPluginTensor> outputs;
  std::vector<JsonPluginTensor> buffers;
  std::unordered_map<std::string, JsonPluginExtern> externs;
  std::unordered_map<std::string, std::vector<std::string>> support_dtypes;
  std::unordered_map<std::string, std::string> options;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("version", version);
    writer->WriteObjectKeyValue("describe", describe);
    writer->WriteObjectKeyValue("attrs", attrs);
    writer->WriteObjectKeyValue("inputs", inputs);
    writer->WriteObjectKeyValue("outputs", outputs);
    writer->WriteObjectKeyValue("buffers", buffers);
    writer->WriteObjectKeyValue("externs", externs);
    writer->WriteObjectKeyValue("support_dtypes", support_dtypes);
    writer->WriteObjectKeyValue("options", options);
    writer->EndObject();
  }

  void Load(dmlc::JSONReader* reader) {
    int bitmask = 0;
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "name") {
        reader->Read(&name);
        bitmask |= 1;
      } else if (key == "version") {
        reader->Read(&version);
      } else if (key == "describe") {
        reader->Read(&describe);
      } else if (key == "attrs") {
        reader->Read(&attrs);
      } else if (key == "inputs") {
        reader->Read(&inputs);
        bitmask |= 2;
      } else if (key == "outputs") {
        reader->Read(&outputs);
        bitmask |= 4;
      } else if (key == "buffers") {
        reader->Read(&buffers);
      } else if (key == "externs") {
        reader->Read(&externs);
      } else if (key == "support_dtypes") {
        reader->Read(&support_dtypes);
      } else if (key == "options") {
        reader->Read(&options);
      }
    }
    ICHECK_EQ(bitmask, 1 | 2 | 4) << "name, inputs and outputs should be given for plugin";
    if (externs.size() > 0) {
      ICHECK(externs.count("infer_output")) << "infer_output should be given as extern";
      bool has_compute = false;
      for (const auto& pair : externs) {
        if (StringUtils::EndsWith(pair.first, "_compute")) {
          has_compute = true;
        }
      }
      ICHECK(has_compute) << "No compute function found, please check";
    }
    if (describe.size() == 0) {
      describe = "Plugin " + name + "(" + version + ")";
    }
  }
};

/*!
 * \brief Attribute in Plugin.
 */
class PluginAttrNode : public Object {
 public:
  /*! \brief The name of attribute. */
  String name;
  /*! \brief The type of attribute. */
  String type;
  /*! \brief The default_value of attribute. */
  String default_value;
  /*! \brief The describe of attribute. */
  String describe;

  /*! \brief Export attribute to json. */
  const JsonPluginAttr ToJson() const;
  /*! \brief Load attribute from json struct. */
  void FromJson(const JsonPluginAttr& j_attr);
  /*! \brief Load attribute from json string. */
  void FromJson(const std::string& json_str);

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("type", &type);
    v->Visit("default_value", &default_value);
    v->Visit("describe", &describe);
  }

  bool SEqualReduce(const PluginAttrNode* other, SEqualReducer equal) const {
    return equal(name, other->name) && equal(type, other->type) &&
           equal(default_value, other->default_value) && equal(describe, other->describe);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(type);
    hash_reduce(default_value);
    hash_reduce(describe);
  }

  static constexpr const char* _type_key = "msc.core.PluginAttr";
  TVM_DECLARE_FINAL_OBJECT_INFO(PluginAttrNode, Object);
};

/*!
 * \brief Managed reference to PluginAttrNode.
 * \sa PluginAttrNode
 */
class PluginAttr : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of the attribute.
   * \param type The type of the attribute.
   * \param default_value The default_value of the attribute.
   * \param describe The describe of the attribute.
   */
  TVM_DLL PluginAttr(const String& name, const String& type, const String& default_value,
                     const String& describe);

  /*!
   * \brief The json constructor.
   * \param j_attr The json describe of the attribute.
   */
  TVM_DLL PluginAttr(const JsonPluginAttr& j_attr);

  /*!
   * \brief The json constructor.
   * \param json_str The json describe of the attribute.
   */
  TVM_DLL PluginAttr(const std::string& json_str);

  TVM_DEFINE_OBJECT_REF_METHODS(PluginAttr, ObjectRef, PluginAttrNode);
};

/*!
 * \brief Tensor in Plugin.
 */
class PluginTensorNode : public Object {
 public:
  /*! \brief The name of tensor. */
  String name;
  /*! \brief The dtype of tensor. */
  String dtype;
  /*! \brief The ndim of tensor. */
  Integer ndim;
  /*! \brief The device of tensor. */
  String device;
  /*! \brief The describe of tensor. */
  String describe;

  /*! \brief Export tensor to json. */
  const JsonPluginTensor ToJson() const;
  /*! \brief Load tensor from json struct. */
  void FromJson(const JsonPluginTensor& j_attr);
  /*! \brief Load tensor from json string. */
  void FromJson(const std::string& json_str);

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("dtype", &dtype);
    v->Visit("ndim", &ndim);
    v->Visit("device", &device);
    v->Visit("describe", &describe);
  }

  bool SEqualReduce(const PluginTensorNode* other, SEqualReducer equal) const {
    return equal(name, other->name) && equal(dtype, other->dtype) && equal(ndim, other->ndim) &&
           equal(device, other->device) && equal(describe, other->describe);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(dtype);
    hash_reduce(ndim);
    hash_reduce(device);
    hash_reduce(describe);
  }

  static constexpr const char* _type_key = "msc.core.PluginTensor";
  TVM_DECLARE_FINAL_OBJECT_INFO(PluginTensorNode, Object);
};

/*!
 * \brief Managed reference to PluginTensorNode.
 * \sa PluginTensorNode
 */
class PluginTensor : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of the tensor.
   * \param dtype The dtype of the tensor.
   * \param ndim The ndim of the tensor.
   * \param device The device of the tensor.
   * \param describe The describe of the tensor.
   */
  TVM_DLL PluginTensor(const String& name, const String& dtype, const Integer& ndim,
                       const String& device, const String& describe);

  /*!
   * \brief The json constructor.
   * \param j_tensor The json describe of the tensor.
   */
  TVM_DLL PluginTensor(const JsonPluginTensor& j_tensor);

  /*!
   * \brief The json constructor.
   * \param json_str The json describe of the tensor.
   */
  TVM_DLL PluginTensor(const std::string& json_str);

  TVM_DEFINE_OBJECT_REF_METHODS(PluginTensor, ObjectRef, PluginTensorNode);
};

/*!
 * \brief Extern symbol in Plugin.
 */
class PluginExternNode : public Object {
 public:
  /*! \brief The name of extern. */
  String name;
  /*! \brief The header of extern. */
  String header;
  /*! \brief The source of extern. */
  String source;
  /*! \brief The lib of extern. */
  String lib;
  /*! \brief The describe of extern. */
  String describe;

  /*! \brief Export extern to json. */
  const JsonPluginExtern ToJson() const;
  /*! \brief Load extern from json struct. */
  void FromJson(const JsonPluginExtern& j_attr);
  /*! \brief Load extern from json string. */
  void FromJson(const std::string& json_str);

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("header", &header);
    v->Visit("source", &source);
    v->Visit("lib", &lib);
    v->Visit("describe", &describe);
  }

  bool SEqualReduce(const PluginExternNode* other, SEqualReducer equal) const {
    return equal(name, other->name) && equal(header, other->header) &&
           equal(source, other->source) && equal(lib, other->lib) &&
           equal(describe, other->describe);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(header);
    hash_reduce(source);
    hash_reduce(lib);
    hash_reduce(describe);
  }

  static constexpr const char* _type_key = "msc.core.PluginExtern";
  TVM_DECLARE_FINAL_OBJECT_INFO(PluginExternNode, Object);
};

/*!
 * \brief Managed reference to PluginExternNode.
 * \sa PluginExternNode
 */
class PluginExtern : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of the extern.
   * \param header The header of the extern.
   * \param source The source of the extern.
   * \param lib The lib of the extern.
   * \param describe The describe of the extern.
   */
  TVM_DLL PluginExtern(const String& name, const String& header, const String& source,
                       const String& lib, const String& describe);

  /*!
   * \brief The json constructor.
   * \param j_extern The json describe of the extern.
   */
  TVM_DLL PluginExtern(const JsonPluginExtern& j_extern);

  /*!
   * \brief The json constructor.
   * \param json_str The json describe of the extern.
   */
  TVM_DLL PluginExtern(const std::string& json_str);

  TVM_DEFINE_OBJECT_REF_METHODS(PluginExtern, ObjectRef, PluginExternNode);
};

/*!
 * \brief The Plugin in MSC.
 */
class PluginNode : public Object {
 public:
  /*! \brief The name of plugin. */
  String name;
  /*! \brief The version of plugin. */
  String version;
  /*! \brief The describe of plugin. */
  String describe;
  /*! \brief The attributes of plugin. */
  Array<PluginAttr> attrs;
  /*! \brief The inputs of plugin. */
  Array<PluginTensor> inputs;
  /*! \brief The outputs of plugin. */
  Array<PluginTensor> outputs;
  /*! \brief The buffers of plugin. */
  Array<PluginTensor> buffers;
  /*! \brief The externs of plugin. */
  Map<String, PluginExtern> externs;
  /*! \brief The support_dtypes of plugin. */
  Map<String, Array<String>> support_dtypes;
  /*! \brief The options of plugin. */
  Map<String, String> options;

  /*! \brief Export plugin to json. */
  const JsonPlugin ToJson() const;
  /*! \brief Load plugin from json struct. */
  void FromJson(const JsonPlugin& j_attr);
  /*! \brief Load plugin from json string. */
  void FromJson(const std::string& json_str);

  /*! \brief Find input ref index for dtype. */
  int FindDtypeRefIdx(const PluginTensor& tensor) const;
  /*! \brief Find input ref index for device. */
  int FindDeviceRefIdx(const PluginTensor& tensor) const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("version", &version);
    v->Visit("describe", &describe);
    v->Visit("attrs", &attrs);
    v->Visit("inputs", &inputs);
    v->Visit("outputs", &outputs);
    v->Visit("buffers", &buffers);
    v->Visit("externs", &externs);
    v->Visit("support_dtypes", &support_dtypes);
    v->Visit("options", &options);
  }

  bool SEqualReduce(const PluginNode* other, SEqualReducer equal) const {
    return equal(name, other->name) && equal(version, other->version) &&
           equal(describe, other->describe) && equal(attrs, other->attrs) &&
           equal(inputs, other->inputs) && equal(outputs, other->outputs) &&
           equal(buffers, other->buffers) && equal(externs, other->externs) &&
           equal(support_dtypes, other->support_dtypes) && equal(options, other->options);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(version);
    hash_reduce(describe);
    hash_reduce(attrs);
    hash_reduce(inputs);
    hash_reduce(outputs);
    hash_reduce(buffers);
    hash_reduce(externs);
    hash_reduce(externs);
    hash_reduce(support_dtypes);
    hash_reduce(options);
  }

  static constexpr const char* _type_key = "msc.core.Plugin";
  TVM_DECLARE_FINAL_OBJECT_INFO(PluginNode, Object);
};

/*!
 * \brief Managed reference to PluginNode.
 * \sa PluginNode
 */
class Plugin : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of the plugin.
   * \param version The version of the plugin.
   * \param describe The describe of the plugin.
   * \param attrs The attrs of the plugin.
   * \param inputs The inputs of the plugin.
   * \param outputs The outputs of the plugin.
   * \param buffers The buffers of the plugin.
   * \param externs The externs of the plugin.
   * \param support_dtypes The support_dtypes of the plugin.
   * \param options The options of the plugin.
   */
  TVM_DLL Plugin(const String& name, const String& version, const String& describe,
                 const Array<PluginAttr>& attrs, const Array<PluginTensor>& inputs,
                 const Array<PluginTensor>& outputs, const Array<PluginTensor>& buffers,
                 const Map<String, PluginExtern>& externs,
                 const Map<String, Array<String>>& support_dtypes,
                 const Map<String, String>& options);

  /*!
   * \brief The json constructor.
   * \param j_plugin The json describe of the plugin.
   */
  TVM_DLL Plugin(const JsonPlugin& j_plugin);

  /*!
   * \brief The json constructor.
   * \param json_str The json describe of the plugin.
   */
  TVM_DLL Plugin(const std::string& json_str);

  TVM_DEFINE_OBJECT_REF_METHODS(Plugin, ObjectRef, PluginNode);
};

class PluginRegistry {
 public:
  /*!
   * \brief Register a new plugin.
   * \param name The name of the item.
   * \param json_str The json_str.
   * \return The corresponding entry.
   */
  bool Register(const String& name, const String& json_str) {
    plugin_map_[name] = Plugin(json_str);
    return true;
  }

  /*!
   * \brief Check if an plugin is registered.
   * \param name The name of the item.
   * \return Whether the plugin is registered.
   */
  bool Registered(const String& name) const {
    auto it = plugin_map_.find(name);
    return it != plugin_map_.end();
  }

  /*!
   * \brief Get an plugin from the registry.
   * \param name The name of the item.
   * \return The corresponding plugin.
   */
  const Plugin Get(const String& name) const {
    auto it = plugin_map_.find(name);
    ICHECK(it != plugin_map_.end()) << "Can not find plugin " << name;
    return it->second;
  }

  /*!
   * \brief List all the plugin names in the registry.
   * \return The plugin names.
   */
  Array<String> ListAllNames() const {
    Array<String> names;
    for (const auto& kv : plugin_map_) {
      names.push_back(kv.first);
    }
    return names;
  }

  /*!
   * \return a global singleton of the registry.
   */
  static PluginRegistry* Global() {
    static PluginRegistry* inst = new PluginRegistry();
    return inst;
  }

 private:
  // map from name to plugins.
  std::unordered_map<String, Plugin> plugin_map_;
};

/*!
 * \brief List all plugin names.
 * \return the corresponding plugin names.
 */
const Array<String> ListPluginNames();

/*!
 * \brief Get the registered plugin.
 * \param name The name of the Plugin.
 * \return the corresponding plugin.
 */
const Plugin GetPlugin(const String& name);

/*!
 * \brief Check if an plugin is registered.
 * \param name The name of the item.
 * \return Whether the plugin is registered.
 */
bool IsPlugin(const String& name);

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_IR_PLUGIN_H_
