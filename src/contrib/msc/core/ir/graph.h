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
 * \file src/contrib/msc/core/ir/graph.h
 * \brief Core MSCGraph.
 */
#ifndef TVM_CONTRIB_MSC_CORE_IR_GRAPH_H_
#define TVM_CONTRIB_MSC_CORE_IR_GRAPH_H_

#include <dmlc/json.h>
#include <tvm/tir/data_layout.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace contrib {
namespace msc {

/*!
 * \brief Json serialize and deserialize for MSCTensor.
 *  MSCTensor is edge in MSCGraph with name, dtype and shape
 */
struct JsonMSCTensor {
  std::string name;
  std::string alias;
  std::string dtype;
  std::string layout;
  std::vector<int64_t> shape;
  std::vector<std::string> prims;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("alias", alias);
    writer->WriteObjectKeyValue("dtype", dtype);
    writer->WriteObjectKeyValue("layout", layout);
    writer->WriteObjectKeyValue("shape", shape);
    writer->WriteObjectKeyValue("prims", prims);
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
      } else if (key == "alias") {
        reader->Read(&alias);
      } else if (key == "dtype") {
        reader->Read(&dtype);
        bitmask |= 2;
      } else if (key == "layout") {
        reader->Read(&layout);
      } else if (key == "shape") {
        reader->Read(&shape);
        bitmask |= 4;
      } else if (key == "prims") {
        reader->Read(&prims);
      }
    }
    ICHECK_EQ(bitmask, 1 | 2 | 4) << "name, dtype and shape should be given";
  }
};

/*!
 * \brief Json serialize and deserialize for MSCJoint.
 *  MSCJoint is node in MSCGraph with name, optype and attrbutes.
 *  MSCJoint has MSCTensors as inputs, outputs and weights.
 */
struct JsonMSCJoint {
  size_t index;
  std::string name;
  std::string shared_ref;
  std::string optype;
  std::vector<std::string> scope;
  std::vector<std::string> parents;
  std::vector<std::string> inputs;
  std::vector<JsonMSCTensor> outputs;
  std::unordered_map<std::string, std::string> attrs;
  std::unordered_map<std::string, JsonMSCTensor> weights;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("index", index);
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("shared_ref", shared_ref);
    writer->WriteObjectKeyValue("optype", optype);
    writer->WriteObjectKeyValue("parents", parents);
    writer->WriteObjectKeyValue("inputs", inputs);
    writer->WriteObjectKeyValue("outputs", outputs);
    writer->WriteObjectKeyValue("attrs", attrs);
    writer->WriteObjectKeyValue("weights", weights);
    writer->EndObject();
  }

  void Load(dmlc::JSONReader* reader) {
    int bitmask = 0;
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "index") {
        reader->Read(&index);
        bitmask |= 1;
      } else if (key == "name") {
        reader->Read(&name);
        bitmask |= 2;
      } else if (key == "shared_ref") {
        reader->Read(&shared_ref);
      } else if (key == "optype") {
        reader->Read(&optype);
        bitmask |= 4;
      } else if (key == "parents") {
        reader->Read(&parents);
      } else if (key == "inputs") {
        reader->Read(&inputs);
      } else if (key == "outputs") {
        reader->Read(&outputs);
        bitmask |= 8;
      } else if (key == "attrs") {
        reader->Read(&attrs);
      } else if (key == "weights") {
        reader->Read(&weights);
      }
    }
    ICHECK_EQ(bitmask, 1 | 2 | 4 | 8) << "index, name, optype and outputs should be given";
  }
};

/*!
 * \brief Json serialize and deserialize for MSCPrim.
 *  MSCPrim is node in MSCGraph with name, op and attrbutes.
 */
struct JsonMSCPrim {
  size_t index;
  std::string name;
  std::string optype;
  std::vector<std::string> parents;
  std::unordered_map<std::string, std::string> attrs;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("index", index);
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("optype", optype);
    writer->WriteObjectKeyValue("parents", parents);
    writer->WriteObjectKeyValue("attrs", attrs);
    writer->EndObject();
  }

  void Load(dmlc::JSONReader* reader) {
    int bitmask = 0;
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "index") {
        reader->Read(&index);
        bitmask |= 1;
      } else if (key == "name") {
        reader->Read(&name);
        bitmask |= 2;
      } else if (key == "optype") {
        reader->Read(&optype);
        bitmask |= 4;
      } else if (key == "parents") {
        reader->Read(&parents);
      } else if (key == "attrs") {
        reader->Read(&attrs);
      }
    }
    ICHECK_EQ(bitmask, 1 | 2 | 4) << "index, name and optype should be given";
  }
};

/*!
 * \brief Json serialize and deserialize for WeightJoint.
 *  WeightJoint is node in WeightGraph with name, wtype and attrbutes.
 *  WeightJoint has MSCTensor as weight.
 */
struct JsonWeightJoint {
  size_t index;
  std::string name;
  std::string shared_ref;
  std::string weight_type;
  JsonMSCTensor weight;
  std::vector<std::string> parents;
  std::vector<std::string> friends;
  std::unordered_map<std::string, std::string> attrs;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("index", index);
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("shared_ref", shared_ref);
    writer->WriteObjectKeyValue("weight_type", weight_type);
    writer->WriteObjectKeyValue("weight", weight);
    writer->WriteObjectKeyValue("parents", parents);
    writer->WriteObjectKeyValue("friends", friends);
    writer->WriteObjectKeyValue("attrs", attrs);
    writer->EndObject();
  }

  void Load(dmlc::JSONReader* reader) {
    int bitmask = 0;
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "index") {
        reader->Read(&index);
        bitmask |= 1;
      } else if (key == "name") {
        reader->Read(&name);
        bitmask |= 2;
      } else if (key == "shared_ref") {
        reader->Read(&shared_ref);
      } else if (key == "weight_type") {
        reader->Read(&weight_type);
        bitmask |= 4;
      } else if (key == "weight") {
        reader->Read(&weight);
        bitmask |= 8;
      } else if (key == "parents") {
        reader->Read(&parents);
      } else if (key == "friends") {
        reader->Read(&friends);
      } else if (key == "attrs") {
        reader->Read(&attrs);
      }
    }
    ICHECK_EQ(bitmask, 1 | 2 | 4 | 8) << "index, name, weight_type and weight should be given";
  }
};

/*!
 * \brief Json serialize and deserialize for MSCGraph.
 *  MSCGraph is core of MSC.
 *  MSCGraph contains MSCJoints as nodes and MSCTensors as edges.
 */
struct JsonMSCGraph {
  std::string name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<JsonMSCJoint> nodes;
  std::vector<JsonMSCPrim> prims;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("inputs", inputs);
    writer->WriteObjectKeyValue("outputs", outputs);
    writer->WriteObjectKeyValue("nodes", nodes);
    writer->WriteObjectKeyValue("prims", prims);
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
      } else if (key == "inputs") {
        reader->Read(&inputs);
        bitmask |= 2;
      } else if (key == "outputs") {
        reader->Read(&outputs);
        bitmask |= 4;
      } else if (key == "nodes") {
        reader->Read(&nodes);
        bitmask |= 8;
      } else if (key == "prims") {
        reader->Read(&prims);
      }
    }
    ICHECK_EQ(bitmask, 1 | 2 | 4 | 8) << "name, inputs, outputs and nodes should be given";
  }
};

/*!
 * \brief Json serialize and deserialize for WeightGraph.
 *  WeightGraph is core of MSC.prune.
 *  WeightGraph contains WeightJoints as nodes.
 */
struct JsonWeightGraph {
  std::string name;
  std::vector<JsonWeightJoint> nodes;

  void Save(dmlc::JSONWriter* writer) const {
    writer->BeginObject();
    writer->WriteObjectKeyValue("name", name);
    writer->WriteObjectKeyValue("nodes", nodes);
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
      } else if (key == "nodes") {
        reader->Read(&nodes);
        bitmask |= 2;
      }
    }
    ICHECK_EQ(bitmask, 1 | 2) << "name and nodes should be given";
  }
};

/*!
 * \brief Tensor in MSCGraph.
 */
class MSCTensorNode : public Object {
 public:
  /*! \brief The name of tensor. */
  String name;
  /*! \brief The alias of tensor, can be changed. */
  mutable String alias;
  /*! \brief The data type of tensor. */
  DataType dtype;
  /*! \brief The layout of tensor. */
  tvm::tir::Layout layout;
  /*! \brief The shape of tensor. */
  Array<Integer> shape;
  /*! \brief The prims of tensor. */
  Array<String> prims;
  /*! \brief Export tensor to json. */
  const JsonMSCTensor ToJson() const;
  /*! \brief Load tensor from json struct. */
  void FromJson(const JsonMSCTensor& j_tensor);
  /*! \brief Load tensor from json string. */
  void FromJson(const std::string& json_str);
  /*! \brief Get the ndim of tensor. */
  size_t Ndim() const;
  /*! \brief Get dim at given index. */
  const Integer DimAt(int index) const;
  /*! \brief Get dim at given axis. */
  const Integer DimAt(const String& axis) const;
  /*! \brief Get prim at given index. */
  const String PrimAt(int index) const;
  /*! \brief Get prim at given axis. */
  const String PrimAt(const String& axis) const;
  /*! \brief Get layout index of given axis. */
  int32_t LayoutOf(const String& axis) const;
  /*! \brief Get size of the tensor. */
  const Integer GetSize() const;
  /*! \brief Get name of the dtype. */
  const String DTypeName() const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("alias", &alias);
    v->Visit("dtype", &dtype);
    v->Visit("layout", &layout);
    v->Visit("shape", &shape);
    v->Visit("prims", &prims);
  }

  bool SEqualReduce(const MSCTensorNode* other, SEqualReducer equal) const {
    return equal(name, other->name) && equal(dtype, other->dtype) && equal(shape, other->shape) &&
           equal(layout, other->layout) && equal(prims, other->prims);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(dtype);
    hash_reduce(shape);
    hash_reduce(layout);
    hash_reduce(prims);
  }

  static constexpr const char* _type_key = "msc.core.MSCTensor";
  TVM_DECLARE_FINAL_OBJECT_INFO(MSCTensorNode, Object);
};

/*!
 * \brief Managed reference to MSCTensorNode.
 * \sa MSCTensorNode
 */
class MSCTensor : public ObjectRef {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of the tensor.
   * \param dtype The data type the tensor.
   * \param layout The layout of the tensor.
   * \param shape The shape of the tensor.
   * \param alias The alias of the tensor.
   * \param prims The prims of the tensor shape.
   */
  TVM_DLL MSCTensor(const String& name, const DataType& dtype, const String& layout,
                    const Array<Integer>& shape, const String& alias = "",
                    const Array<String>& prims = Array<String>());

  /*!
   * \brief The json constructor.
   * \param j_tensor The json describe of the tensor.
   */
  TVM_DLL MSCTensor(const JsonMSCTensor& j_tensor);

  /*!
   * \brief The json constructor.
   * \param json_str The json describe of the tensor.
   */
  TVM_DLL MSCTensor(const std::string& json_str);

  TVM_DEFINE_OBJECT_REF_METHODS(MSCTensor, ObjectRef, MSCTensorNode);
};

/*!
 * \brief Basic node in MSCGraph and WeightGraph.
 */
class BaseJoint;
class BaseJointNode : public Object {
 public:
  /*! \brief The index of node, can be changed. */
  mutable int index;
  /*! \brief The name of node. */
  String name;
  /*! \brief The shared_ref of node, can be changed. */
  String shared_ref;
  /*! \brief The attributes of node. */
  mutable Map<String, String> attrs;
  /*! \brief The parents of node. */
  Array<ObjectRef> parents;
  /*! \brief The children of node. */
  mutable Array<ObjectRef> children;
  /*! \brief Add child to the node. */
  size_t AddChild(const BaseJoint& child) const;
  /*! \brief Get parent from the node. */
  const BaseJoint ParentAt(int index) const;
  /*! \brief Get child from the node. */
  const BaseJoint ChildAt(int index) const;
  /*! \brief Check if has the attribute. */
  bool HasAttr(const String& key) const;
  /*! \brief Get the attribute by type. */
  bool GetAttr(const String& key, std::string* val) const;
  bool GetAttr(const String& key, int* val) const;
  bool GetAttr(const String& key, int64_t* val) const;
  bool GetAttr(const String& key, float* val) const;
  bool GetAttr(const String& key, bool* val) const;
  bool GetAttr(const String& key, std::vector<std::string>* val) const;
  bool GetAttr(const String& key, std::vector<int>* val) const;
  bool GetAttr(const String& key, std::vector<int64_t>* val) const;
  bool GetAttr(const String& key, std::vector<float>* val) const;
  bool GetAttr(const String& key, std::vector<bool>* val) const;
  /*! \brief Check and get the attribute by type. */
  template <typename T>
  const T GetTypeAttr(const String& key) const {
    T val;
    ICHECK(GetAttr(key, &val)) << "Can not get attr " << key;
    return val;
  }
  template <typename T>
  const std::vector<T> GetTypeArrayAttr(const String& key) const {
    std::vector<T> val;
    ICHECK(GetAttr(key, &val)) << "Can not get attr " << key;
    return val;
  }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("index", &index);
    v->Visit("name", &name);
    v->Visit("shared_ref", &shared_ref);
    v->Visit("attrs", &attrs);
    v->Visit("parents", &parents);
    v->Visit("children", &children);
  }

  bool SEqualReduce(const BaseJointNode* other, SEqualReducer equal) const {
    return equal(name, other->name) && equal(shared_ref, other->shared_ref) &&
           equal(attrs, other->attrs) && equal(parents, other->parents) &&
           equal(children, other->children);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(shared_ref);
    hash_reduce(attrs);
    hash_reduce(parents);
    hash_reduce(children);
  }

  static constexpr const char* _type_key = "msc.core.BaseJoint";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const uint32_t _type_child_slots = 2;
  TVM_DECLARE_BASE_OBJECT_INFO(BaseJointNode, Object);
};

/*!
 * \brief Managed reference to BaseJointNode.
 * \sa BaseJointNode
 */
class BaseJoint : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(BaseJoint, ObjectRef, BaseJointNode);
};

/*!
 * \brief Node in MSCGraph.
 */
class MSCJoint;
class MSCJointNode : public BaseJointNode {
 public:
  /*! \brief The op type of node. */
  String optype;
  /*! \brief The scope of node. */
  Array<String> scope;
  /*! \brief The inputs of node, can be changed. */
  Array<Array<Integer>> inputs;
  /*! \brief The outputs of node. */
  Array<MSCTensor> outputs;
  /*! \brief The weights of node. */
  Map<String, MSCTensor> weights;
  /*! \brief Export node to json. */
  const JsonMSCJoint ToJson() const;
  /*! \brief Load node from json struct. */
  void FromJson(const JsonMSCJoint& j_joint, const Map<String, BaseJoint>& nodes);
  /*! \brief Load node from json string. */
  void FromJson(const std::string& json_str, const Map<String, BaseJoint>& nodes);
  /*! \brief Get input from the node. */
  const MSCTensor InputAt(int index) const;
  /*! \brief Get inputs from the node. */
  const Array<MSCTensor> GetInputs() const;
  /*! \brief Get output from the node. */
  const MSCTensor OutputAt(int index) const;
  /*! \brief Get outputs from the node. */
  const Array<MSCTensor> GetOutputs() const;
  /*! \brief Get weight from the node. */
  const MSCTensor WeightAt(const String& wtype) const;
  /*! \brief Get parent from the node. */
  const MSCJoint ParentAt(int index) const;
  /*! \brief Get child from the node. */
  const MSCJoint ChildAt(int index) const;
  /*! \brief Get Producer of the input. */
  const MSCJoint ProducerOf(int index) const;
  const MSCJoint ProducerOf(const String& input_name) const;
  const MSCJoint ProducerOf(const MSCTensor& input) const;
  /*! \brief Get Producer and out index of the input. */
  const std::pair<MSCJoint, size_t> ProducerAndIdxOf(int index) const;
  const std::pair<MSCJoint, size_t> ProducerAndIdxOf(const String& name) const;
  const std::pair<MSCJoint, size_t> ProducerAndIdxOf(const MSCTensor& input) const;

  void VisitAttrs(AttrVisitor* v) {
    BaseJointNode::VisitAttrs(v);
    v->Visit("optype", &optype);
    v->Visit("scope", &scope);
    v->Visit("inputs", &inputs);
    v->Visit("outputs", &outputs);
    v->Visit("weights", &weights);
  }

  bool SEqualReduce(const MSCJointNode* other, SEqualReducer equal) const {
    return BaseJointNode::SEqualReduce(other, equal) && equal(optype, other->optype) &&
           equal(scope, other->scope) && equal(inputs, other->inputs) &&
           equal(outputs, other->outputs) && equal(weights, other->weights);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    BaseJointNode::SHashReduce(hash_reduce);
    hash_reduce(optype);
    hash_reduce(scope);
    hash_reduce(inputs);
    hash_reduce(outputs);
    hash_reduce(weights);
  }

  static constexpr const char* _type_key = "msc.core.MSCJoint";
  TVM_DECLARE_FINAL_OBJECT_INFO(MSCJointNode, BaseJointNode);
};

/*!
 * \brief Managed reference to MSCJointNode.
 * \sa MSCJointNode
 */
class MSCJoint : public BaseJoint {
 public:
  /*!
   * \brief The constructor.
   * \param index The index of the node.
   * \param name The name of the node.
   * \param shared_ref The shared_ref of the node.
   * \param optype The op type the node.
   * \param attrs The attributes of the node.
   * \param inputs The inputs of the node.
   * \param outputs The outputs of the node.
   * \param weights The weights of the node.
   */
  TVM_DLL MSCJoint(int index, const String& name, const String& shared_ref, const String& optype,
                   const Map<String, String>& attrs, const Array<String>& scope,
                   const std::vector<std::pair<BaseJoint, size_t>>& inputs,
                   const Array<MSCTensor>& outputs, const Map<String, MSCTensor>& weights);

  /*!
   * \brief The json constructor.
   * \param j_joint The json describe of the node.
   */
  TVM_DLL MSCJoint(const JsonMSCJoint& j_joint, const Map<String, BaseJoint>& nodes);

  /*!
   * \brief The json constructor.
   * \param json_str The json describe of the node.
   */
  TVM_DLL MSCJoint(const std::string& json_str, const Map<String, BaseJoint>& nodes);

  /*! \brief Clone the node. */
  TVM_DLL static const MSCJoint Clone(const MSCJoint& node,
                                      const std::vector<std::pair<BaseJoint, size_t>>& inputs);

  TVM_DEFINE_OBJECT_REF_METHODS(MSCJoint, BaseJoint, MSCJointNode);
};

/*!
 * \brief MSCPrim in MSCGraph.
 */
class MSCPrim;
class MSCPrimNode : public BaseJointNode {
 public:
  /*! \brief The op of prim. */
  String optype;
  /*! \brief Export prim to json. */
  const JsonMSCPrim ToJson() const;
  /*! \brief Load prim from json struct. */
  void FromJson(const JsonMSCPrim& j_prim, const Map<String, BaseJoint>& prims);
  /*! \brief Load prim from json string. */
  void FromJson(const std::string& json_str, const Map<String, BaseJoint>& prims);
  /*! \brief Get parent from the prim. */
  const MSCPrim ParentAt(int index) const;
  /*! \brief Get child from the prim. */
  const MSCPrim ChildAt(int index) const;

  void VisitAttrs(AttrVisitor* v) {
    BaseJointNode::VisitAttrs(v);
    v->Visit("optype", &optype);
  }

  bool SEqualReduce(const MSCPrimNode* other, SEqualReducer equal) const {
    return BaseJointNode::SEqualReduce(other, equal) && equal(optype, other->optype);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    BaseJointNode::SHashReduce(hash_reduce);
    hash_reduce(optype);
  }

  static constexpr const char* _type_key = "msc.core.MSCPrim";
  TVM_DECLARE_FINAL_OBJECT_INFO(MSCPrimNode, BaseJointNode);
};

/*!
 * \brief Managed reference to MSCPrimNode.
 * \sa MSCPrimNode
 */
class MSCPrim : public BaseJoint {
 public:
  /*!
   * \brief The constructor.
   * \param index The index of the prim.
   * \param name The name of the prim.
   * \param optype The optype of the prim.
   * \param parents The parents of the prim.
   * \param attrs The attributes of the prim.
   */
  TVM_DLL MSCPrim(int index, const String& name, const String& optype,
                  const Array<BaseJoint>& parents,
                  const Map<String, String>& attrs = Map<String, String>());

  /*!
   * \brief The json constructor.
   * \param j_prim The json describe of the prim.
   */
  TVM_DLL MSCPrim(const JsonMSCPrim& j_prim, const Map<String, BaseJoint>& prims);

  /*!
   * \brief The json constructor.
   * \param json_str The json describe of the prim.
   */
  TVM_DLL MSCPrim(const std::string& json_str, const Map<String, BaseJoint>& prims);

  TVM_DEFINE_OBJECT_REF_METHODS(MSCPrim, BaseJoint, MSCPrimNode);
};

/*!
 * \brief Node in WeightGraph.
 */
class WeightJoint;
class WeightJointNode : public BaseJointNode {
 public:
  /*! \brief The weight reference of weight node. */
  String weight_type;
  /*! \brief The weight of weight node. */
  MSCTensor weight;
  /*! \brief The friends of weight node. */
  mutable Array<BaseJoint> friends;
  /*! \brief Export node to json. */
  const JsonWeightJoint ToJson() const;
  /*! \brief Load node from json struct. */
  void FromJson(const JsonWeightJoint& j_joint, const Map<String, BaseJoint>& nodes);
  /*! \brief Load node from json string. */
  void FromJson(const std::string& json_str, const Map<String, BaseJoint>& nodes);
  /*! \brief Get parent from the node. */
  const WeightJoint ParentAt(int index) const;
  /*! \brief Get child from the node. */
  const WeightJoint ChildAt(int index) const;

  void VisitAttrs(AttrVisitor* v) {
    BaseJointNode::VisitAttrs(v);
    v->Visit("weight_type", &weight_type);
    v->Visit("weight", &weight);
    v->Visit("friends", &friends);
  }

  bool SEqualReduce(const WeightJointNode* other, SEqualReducer equal) const {
    return BaseJointNode::SEqualReduce(other, equal) && equal(weight_type, other->weight_type) &&
           equal(weight, other->weight) && equal(friends, other->friends);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    BaseJointNode::SHashReduce(hash_reduce);
    hash_reduce(weight_type);
    hash_reduce(weight);
    hash_reduce(friends);
  }

  static constexpr const char* _type_key = "msc.core.WeightJoint";
  TVM_DECLARE_FINAL_OBJECT_INFO(WeightJointNode, BaseJointNode);
};

/*!
 * \brief Managed reference to WeightJointNode.
 * \sa WeightJointNode
 */
class WeightJoint : public BaseJoint {
 public:
  /*!
   * \brief The constructor.
   * \param index The index of the node.
   * \param name The name of the node.
   * \param shared_ref The shared_ref of the node.
   * \param weight_type The weight type of the node.
   * \param weight The weight tensor of the node.
   * \param parents The parents of the node.
   * \param attrs The attributes of the node.
   * \param friends The friends of the node.
   */
  TVM_DLL WeightJoint(int index, const String& name, const String& shared_ref,
                      const String& weight_type, const MSCTensor& weight,
                      const Array<BaseJoint> parents,
                      const Map<String, String>& attrs = Map<String, String>(),
                      const Array<BaseJoint>& friends = Array<BaseJoint>());

  /*!
   * \brief The json constructor.
   * \param j_joint The json describe of the node.
   */
  TVM_DLL WeightJoint(const JsonWeightJoint& j_joint, const Map<String, BaseJoint>& nodes);

  /*!
   * \brief The json constructor.
   * \param json_str The json describe of the node.
   */
  TVM_DLL WeightJoint(const std::string& json_str, const Map<String, BaseJoint>& nodes);

  TVM_DEFINE_OBJECT_REF_METHODS(WeightJoint, BaseJoint, WeightJointNode);
};

/*!
 * \brief Basic graph class (MSCGraph and WeightGraph).
 */
class BaseGraphNode : public Object {
 public:
  /*! \brief The name of graph. */
  String name;
  /*! \brief The node names in graph, can be changed. */
  Array<String> node_names;
  /*! \brief The nodes in graph, can be changed. */
  Map<String, BaseJoint> nodes;
  /*! \brief Check if node in the graph. */
  const bool HasNode(const String& name) const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("nodes", &nodes);
    v->Visit("node_names", &node_names);
  }

  bool SEqualReduce(const BaseGraphNode* other, SEqualReducer equal) const {
    return equal(name, other->name) && equal(nodes, other->nodes) &&
           equal(node_names, other->node_names);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(name);
    hash_reduce(nodes);
    hash_reduce(node_names);
  }

  static constexpr const char* _type_key = "msc.core.BaseGraph";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const uint32_t _type_child_slots = 2;
  TVM_DECLARE_BASE_OBJECT_INFO(BaseGraphNode, Object);
};

/*!
 * \brief Managed reference to BaseGraphNode.
 * \sa BaseGraphNode
 */
class BaseGraph : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(BaseGraph, ObjectRef, BaseGraphNode);
};

/*!
 * \brief MSCGraph.
 */
class MSCGraph;
class MSCGraphNode : public BaseGraphNode {
 public:
  /*! \brief The shape node names in graph. */
  Array<String> prim_names;
  /*! \brief The shape nodes in graph. */
  Map<String, MSCPrim> prims;
  /*! \brief The input names of graph. */
  Array<String> input_names;
  /*! \brief The output names of graph. */
  Array<String> output_names;
  /*! \brief The tensor alias in graph, get by AnalysisGraph. */
  mutable Map<String, String> tensor_alias;
  /*! \brief The weights in graph, get by AnalysisGraph. */
  Map<String, Array<String>> weight_holders;
  /*! \brief Export graph to json. */
  const JsonMSCGraph ToJson() const;
  /*! \brief Load graph from json. */
  void FromJson(const JsonMSCGraph& json_str);
  /*! \brief Load graph from json string. */
  void FromJson(const std::string& json_str);
  /*! \brief Export graph to prototxt. */
  const String ToPrototxt() const;
  /*! \brief Find node in graph. */
  const MSCJoint FindNode(const String& name) const;
  /*! \brief Find prim in graph. */
  const MSCPrim FindPrim(const String& name) const;
  /*! \brief Get input from the graph. */
  const MSCTensor InputAt(int index) const;
  /*! \brief Get inputs from the graph. */
  const Array<MSCTensor> GetInputs() const;
  /*! \brief Get output from the graph. */
  const MSCTensor OutputAt(int index) const;
  /*! \brief Get outputs from the graph. */
  const Array<MSCTensor> GetOutputs() const;
  /*! \brief Get entries from the graph. */
  const Array<MSCJoint> GetEntries() const;
  /*! \brief Get exits from the graph. */
  const Array<MSCJoint> GetExits() const;
  /*! \brief Check if tensor in the graph. */
  const bool HasTensor(const String& name) const;
  /*! \brief Find tensor from the graph. */
  const MSCTensor FindTensor(const String& name) const;
  /*! \brief Find producer of tensor from the graph. */
  const MSCJoint FindProducer(const String& name) const;
  /*! \brief Find producer of tensor from the graph. */
  const MSCJoint FindProducer(const MSCTensor& tensor) const;
  /*! \brief Find producer and output index of tensor from the graph. */
  const std::pair<MSCJoint, size_t> FindProducerAndIdx(const String& name) const;
  /*! \brief Find producer and output index of tensor from the graph. */
  const std::pair<MSCJoint, size_t> FindProducerAndIdx(const MSCTensor& tensor) const;
  /*! \brief Find consumers of tensor from the graph. */
  const Array<MSCJoint> FindConsumers(const String& name) const;
  /*! \brief Find consumers of tensor from the graph. */
  const Array<MSCJoint> FindConsumers(const MSCTensor& tensor) const;
  /*! \brief Find consumers and input indices of tensor from the graph. */
  const std::vector<std::pair<MSCJoint, size_t>> FindConsumersAndIndices(const String& name) const;
  /*! \brief Find consumers and input indices of tensor from the graph. */
  const std::vector<std::pair<MSCJoint, size_t>> FindConsumersAndIndices(
      const MSCTensor& tensor) const;
  /*! \brief Analysis the graph and fill info. */
  void AnalysisGraph();

  void VisitAttrs(AttrVisitor* v) {
    BaseGraphNode::VisitAttrs(v);
    v->Visit("prims", &prims);
    v->Visit("prim_names", &prim_names);
    v->Visit("input_names", &input_names);
    v->Visit("output_names", &output_names);
    v->Visit("weight_holders", &weight_holders);
  }

  bool SEqualReduce(const MSCGraphNode* other, SEqualReducer equal) const {
    return BaseGraphNode::SEqualReduce(other, equal) && equal(prims, other->prims) &&
           equal(prim_names, other->prim_names) && equal(input_names, other->input_names) &&
           equal(output_names, other->output_names) && equal(weight_holders, other->weight_holders);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    BaseGraphNode::SHashReduce(hash_reduce);
    hash_reduce(prims);
    hash_reduce(prim_names);
    hash_reduce(input_names);
    hash_reduce(output_names);
    hash_reduce(weight_holders);
  }

  static constexpr const char* _type_key = "msc.core.MSCGraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(MSCGraphNode, BaseGraphNode);
};

/*!
 * \brief Managed reference to MSCGraphNode.
 * \sa MSCGraphNode
 */
class MSCGraph : public BaseGraph {
 public:
  /*!
   * \brief The constructor.
   * \param name The name of the node.
   * \param nodes The nodes in the graph.
   * \param input_names The input names of the graph.
   * \param output_names The output names of the graph.
   * \param prims The prims in the graph.
   */
  TVM_DLL MSCGraph(const String& name, const Array<MSCJoint>& nodes,
                   const Array<String>& input_names, const Array<String>& output_names,
                   const Array<MSCPrim>& prims = Array<MSCPrim>());

  /*!
   * \brief The json constructor.
   * \param j_graph The json describe of the graph.
   */
  TVM_DLL MSCGraph(const JsonMSCGraph& j_graph);

  /*!
   * \brief The json constructor.
   * \param json_str The json describe of the graph.
   */
  TVM_DLL MSCGraph(const std::string& json_str);

  TVM_DEFINE_OBJECT_REF_METHODS(MSCGraph, BaseGraph, MSCGraphNode);
};

/*!
 * \brief WeightGraph.
 */
class WeightGraphNode : public BaseGraphNode {
 public:
  /*! \brief build from MSCGraph. */
  void Build(const MSCGraph& graph, const Map<String, Array<String>>& prunable_types,
             const Map<String, String>& relation_types);
  /*! \brief Find node in graph. */
  const WeightJoint FindNode(const String& name) const;
  /*! \brief Export graph to json. */
  const JsonWeightGraph ToJson() const;
  /*! \brief Load graph from json. */
  void FromJson(const JsonWeightGraph& json_str);
  /*! \brief Load graph from json string. */
  void FromJson(const std::string& json_str);
  /*! \brief Export graph to prototxt. */
  const String ToPrototxt() const;

  void VisitAttrs(AttrVisitor* v) { BaseGraphNode::VisitAttrs(v); }

  bool SEqualReduce(const WeightGraphNode* other, SEqualReducer equal) const {
    return BaseGraphNode::SEqualReduce(other, equal);
  }

  void SHashReduce(SHashReducer hash_reduce) const { BaseGraphNode::SHashReduce(hash_reduce); }

  static constexpr const char* _type_key = "msc.core.WeightGraph";
  TVM_DECLARE_FINAL_OBJECT_INFO(WeightGraphNode, BaseGraphNode);
};

/*!
 * \brief Managed reference to WeightGraphNode.
 * \sa WeightGraphNode
 */
class WeightGraph : public BaseGraph {
 public:
  /*!
   * \brief The constructor based on MSCGraph.
   * \param graph The msc graph.
   * \param prunable_types The prunable types.
   * \param relation_types The relation types.
   */
  TVM_DLL WeightGraph(const MSCGraph& graph, const Map<String, Array<String>>& prunable_types,
                      const Map<String, String>& relation_types);

  /*!
   * \brief The json constructor.
   * \param j_graph The json describe of the graph.
   */
  TVM_DLL WeightGraph(const JsonWeightGraph& j_graph);

  /*!
   * \brief The json constructor.
   * \param json_str The json describe of the graph.
   */
  TVM_DLL WeightGraph(const std::string& json_str);

  TVM_DEFINE_OBJECT_REF_METHODS(WeightGraph, BaseGraph, WeightGraphNode);
};

MSCGraph PruneWeights(const MSCGraph& graph, const Map<String, MSCTensor>& pruned_tensors);

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_IR_GRAPH_H_
