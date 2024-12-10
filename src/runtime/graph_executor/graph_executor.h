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
 * \brief Tiny graph executor that can run graph
 *        containing only tvm PackedFunc.
 * \file graph_executor.h
 */
#ifndef TVM_RUNTIME_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_
#define TVM_RUNTIME_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_

#include <dlpack/dlpack.h>
#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {

using memory::AllocatorType;
using memory::MemoryManager;

/*! \brief macro to do C API call */
#define TVM_CCALL(func)                     \
  {                                         \
    int ret = (func);                       \
    ICHECK_EQ(ret, 0) << TVMGetLastError(); \
  }

/*! \brief operator attributes about tvm op */
struct TVMOpParam {
  std::string func_name;
  std::unordered_map<std::string, ObjectRef> attrs;
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t flatten_data;
};

/*!
 * \brief Tiny graph executor.
 *
 *  This runtime can be accessible in various languages via
 *  TVM runtime PackedFunc API.
 */
class TVM_DLL GraphExecutor : public ModuleNode {
  struct OpArgs {
    std::vector<DLTensor*> args;
    std::vector<TVMValue> arg_values;
    std::vector<int> arg_tcodes;
    std::vector<int64_t> shape_data;
  };

 public:
  using ShapeInfo = Map<String, ObjectRef>;
  using DtypeInfo = Map<String, ObjectRef>;
  /*!
   * \brief Get member function to front-end
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  virtual PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self);

  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const final { return "GraphExecutor"; }
  void Run();

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return ModulePropertyMask::kRunnable; }

  /*!
   * \brief Initialize the graph executor with graph and device.
   * \param graph_json The execution graph.
   * \param module The module containing the compiled functions for the host
   *  processor.
   * \param devs The device of the host and devices where graph nodes will be
   *  executed on.
   * \param lookup_linked_param_func If given, a PackedFunc invoked to lookup linked parameters
   *  by storage_id. If not given, linked parameters are looked-up using an internal implementation,
   *  which is not compatible with RPCModules. Default is nullptr.
   */

  void Init(const std::string& graph_json, tvm::runtime::Module module,
            const std::vector<Device>& devs, const PackedFunc lookup_linked_param_func = nullptr);

  /*!
   * \brief Get the input index given the name of input.
   * \param name The name of the input.
   * \return The index of input.
   */
  int GetInputIndex(const std::string& name);

  /*!
   * \brief Get the input info of Graph by parsing the input nodes.
   * \return The shape and dtype tuple.
   */
  std::tuple<ShapeInfo, DtypeInfo> GetInputInfo() const;

  /*!
   * \brief Get the output info of Graph by parsing the output nodes.
   * \return The shape and dtype tuple.
   */
  std::tuple<ShapeInfo, DtypeInfo> GetOutputInfo() const;

  /*!
   * \brief Get the output index given the name of output.
   * \param name The name of the output.
   * \return The index of output.
   */
  int GetOutputIndex(const std::string& name);

  /*!
   * \brief set index-th input to the graph.
   * \param index The input index.
   * \param data_in The input data.
   */
  void SetInput(int index, DLTensor* data_in);
  /*!
   * \brief set index-th input to the graph without copying the data
   * \param index The input index.
   * \param data_ref The input data that is referred.
   */
  void SetInputZeroCopy(int index, DLTensor* data_ref);
  /*!
   * \brief set index-th output to the graph without copying the data.
   * \param index The output index.
   * \param data_ref The output data that is referred.
   */
  void SetOutputZeroCopy(int index, DLTensor* data_ref);
  /*!
   * \brief Get the number of outputs
   *
   * \return The number of outputs from graph.
   */
  int NumOutputs() const;
  /*!
   * \brief Get the number of inputs
   *
   * \return The number of inputs to the graph.
   */
  int NumInputs() const;
  /*!
   * \brief Return NDArray for given input index.
   * \param index The input index.
   *
   * \return NDArray corresponding to given input node index.
   */
  NDArray GetInput(int index) const;
  /*!
   * \brief Return NDArray for given output index.
   * \param index The output index.
   *
   * \return NDArray corresponding to given output node index.
   */
  NDArray GetOutput(int index) const;
  /*!
   * \brief Copy index-th output to data_out.
   * \param index The output index.
   * \param data_out the output data.
   */
  void CopyOutputTo(int index, DLTensor* data_out);
  /*!
   * \brief Load parameters from binary stream
   * \param strm The input stream.
   */
  void LoadParams(dmlc::Stream* strm);
  /*!
   * \brief Load parameters from parameter blob.
   * \param param_blob A binary blob of parameter.
   */
  void LoadParams(const std::string& param_blob);

  /*!
   * \brief Share parameters from pre-existing GraphExecutor instance.
   * \param other A GraphExecutor instance, previously with |LoadParams| called with the
   * identical input |param_blob|.
   * \param strm The input stream.
   */
  void ShareParams(const GraphExecutor& other, dmlc::Stream* strm);

  /*!
   * \brief Get total number of nodes.
   * \return Total number of nodes.
   */
  uint32_t GetNumOfNodes() const { return static_cast<uint32_t>(nodes_.size()); }

  std::string GetNodeName(uint32_t nid) const { return nodes_[nid].name; }

 protected:
  // Memory pool entry.
  struct PoolEntry {
    int device_type;
    std::vector<int64_t> shape;
    DLDataType dtype;
    int param_data_entry;
    NDArray linked_param;
    std::string scope;
    //    PoolEntry(int s, int dev_type, void* pre_linked_param) :
    //        size(s), device_type(dev_type), pre_linked_param(std::move(pre_linked_param)) {}
  };
  // Node entry
  struct NodeEntry {
    uint32_t node_id;
    uint32_t index;
    uint32_t version;
    inline bool operator==(const NodeEntry& other) const {
      return node_id == other.node_id && index == other.index && version == other.version;
    }
    // JSON Loader
    void Load(dmlc::JSONReader* reader) {
      reader->BeginArray();
      ICHECK(reader->NextArrayItem()) << "invalid json format";
      reader->Read(&node_id);
      ICHECK(reader->NextArrayItem()) << "invalid json format";
      reader->Read(&index);
      if (reader->NextArrayItem()) {
        reader->Read(&version);
        ICHECK(!reader->NextArrayItem()) << "invalid json format";
      } else {
        version = 0;
      }
    }
  };
  // Node
  struct Node {
    // operator type in string
    std::string op_type;
    // name of the op
    std::string name;
    // parameters
    TVMOpParam param;
    // inputs
    std::vector<NodeEntry> inputs;
    // control deps
    std::vector<uint32_t> control_deps;

    // JSON Loader
    void LoadAttrs(dmlc::JSONReader* reader, TVMOpParam* param) {
      int bitmask = 0;
      std::string key, value;
      reader->BeginObject();
      while (reader->NextObjectItem(&key)) {
        reader->Read(&value);
        if (key == "func_name") {
          param->func_name = value;
          bitmask |= 1;
        } else if (key == "num_inputs") {
          param->num_inputs = strtoul(value.c_str(), nullptr, 10);
          bitmask |= 2;
        } else if (key == "num_outputs") {
          param->num_outputs = strtoul(value.c_str(), nullptr, 10);
          bitmask |= 4;
        } else if (key == "flatten_data") {
          param->flatten_data = strtoul(value.c_str(), nullptr, 10);
          bitmask |= 8;
        } else {
          param->attrs[key] = String(value);
        }
      }
      ICHECK_EQ(bitmask, 1 | 2 | 4 | 8) << "invalid format";
    }
    // JSON Loader
    void Load(dmlc::JSONReader* reader) {
      reader->BeginObject();
      int bitmask = 0;
      std::string key;
      while (reader->NextObjectItem(&key)) {
        if (key == "op") {
          reader->Read(&op_type);
          bitmask |= 1;
        } else if (key == "name") {
          reader->Read(&name);
          bitmask |= 2;
        } else if (key == "inputs") {
          reader->Read(&inputs);
          bitmask |= 4;
        } else if (key == "attr" || key == "attrs") {
          this->LoadAttrs(reader, &param);
        } else if (key == "control_deps") {
          reader->Read(&control_deps);
        } else {
          LOG(FATAL) << "do not support key " << key;
        }
      }
      ICHECK_EQ(bitmask, 1 | 2 | 4) << "invalid format";
    }
  };
  struct GraphAttr {
    size_t storage_num_not_alloctaed{0};
    std::vector<int> storage_id;
    std::vector<int> device_index;
    std::vector<std::string> dltype;
    std::vector<std::string> storage_scope;
    std::vector<std::vector<int64_t>> shape;
    // The graph attribute fields.
    void Load(dmlc::JSONReader* reader) {
      reader->BeginObject();
      int bitmask = 0;
      std::string key, type;
      while (reader->NextObjectItem(&key)) {
        if (key == "dltype") {
          reader->BeginArray();
          ICHECK(reader->NextArrayItem());
          reader->Read(&type);
          ICHECK_EQ(type, "list_str");
          ICHECK(reader->NextArrayItem());
          reader->Read(&dltype);
          ICHECK(!reader->NextArrayItem());
          bitmask |= 1;
        } else if (key == "storage_id") {
          reader->BeginArray();
          ICHECK(reader->NextArrayItem());
          reader->Read(&type);
          ICHECK_EQ(type, "list_int");
          ICHECK(reader->NextArrayItem());
          reader->Read(&storage_id);
          ICHECK(!reader->NextArrayItem());
          bitmask |= 2;
        } else if (key == "storage_scope") {
          reader->BeginArray();
          ICHECK(reader->NextArrayItem());
          reader->Read(&type);
          ICHECK_EQ(type, "list_str");
          ICHECK(reader->NextArrayItem());
          reader->Read(&storage_scope);
          ICHECK(!reader->NextArrayItem());
          bitmask |= 1;
        } else if (key == "shape") {
          reader->BeginArray();
          ICHECK(reader->NextArrayItem());
          reader->Read(&type);
          ICHECK_EQ(type, "list_shape");
          ICHECK(reader->NextArrayItem());
          reader->Read(&shape);
          ICHECK(!reader->NextArrayItem());
          bitmask |= 4;
        } else if (key == "device_index") {
          reader->BeginArray();
          ICHECK(reader->NextArrayItem());
          reader->Read(&type);
          ICHECK_EQ(type, "list_int");
          ICHECK(reader->NextArrayItem());
          reader->Read(&device_index);
          ICHECK(!reader->NextArrayItem());
        } else {
          reader->BeginArray();
          ICHECK(reader->NextArrayItem());
          reader->Read(&type);
          if (type == "list_int") {
            ICHECK(reader->NextArrayItem());
            std::vector<int> temp;
            reader->Read(&temp);
          } else if (type == "size_t") {
            ICHECK(reader->NextArrayItem());
            size_t temp;
            reader->Read(&temp);
          } else {
            LOG(FATAL) << "cannot skip graph attr " << key;
          }
          ICHECK(!reader->NextArrayItem());
        }
      }
      ICHECK_EQ(bitmask, 1 | 2 | 4) << "invalid format";
    }
  };
  // The graph attribute fields.
  void Load(dmlc::JSONReader* reader) {
    reader->BeginObject();
    int bitmask = 0;
    std::string key;
    while (reader->NextObjectItem(&key)) {
      if (key == "nodes") {
        reader->Read(&nodes_);
        bitmask |= 1;
      } else if (key == "arg_nodes") {
        reader->Read(&input_nodes_);
        bitmask |= 2;
      } else if (key == "node_row_ptr") {
        reader->Read(&node_row_ptr_);
        bitmask |= 4;
      } else if (key == "heads") {
        reader->Read(&outputs_);
        bitmask |= 8;
      } else if (key == "attrs") {
        reader->Read(&attrs_);
        bitmask |= 16;
      } else if (key == "metadata") {
        break;
      } else {
        LOG(FATAL) << "key " << key << " is not supported";
      }
    }
    ICHECK_EQ(bitmask, 1 | 2 | 4 | 8 | 16) << "invalid format";
  }
  /*! \brief PackedFunc to lookup a linked paramter from a local Module. */
  void DefaultLookupLinkedParam(TVMArgs args, TVMRetValue* rv);
  /*! \brief Delete NDArray::Container with linked (i.e. static) data. */
  static void LinkedNDArrayDeleter(Object* container);
  /*! \brief Setup the temporal storage */
  void SetupStorage();
  /*! \brief Setup the executors. */
  void SetupOpExecs();
  /*!
   * \brief Check the legality of external DLTensor*.
   * \param external The external DLTensor*.
   * \param eid The data_enrty_ index.
   */
  void CheckExternalDLTensor(const DLTensor* external, uint32_t eid) const;
  /*!
   * \brief Create an execution function given input.
   * \param attrs The node attributes.
   * \param args The arguments to the functor, including inputs and outputs.
   * \return The created executor.
   */
  std::pair<std::function<void()>, std::shared_ptr<OpArgs>> CreateTVMOp(
      const TVMOpParam& attrs, const std::vector<DLTensor*>& args);
  // Get node entry index.
  uint32_t entry_id(uint32_t nid, uint32_t index) const { return node_row_ptr_[nid] + index; }
  // Get node entry index.
  uint32_t entry_id(const NodeEntry& e) const { return entry_id(e.node_id, e.index); }
  // Number of node entries.
  uint32_t num_node_entries() const { return node_row_ptr_.back(); }
  /*! \brief The graph nodes. */
  std::vector<Node> nodes_;
  /*! \brief The argument nodes. */
  std::vector<uint32_t> input_nodes_;
  /*! \brief The parameter names. */
  std::unordered_set<std::string> param_names_;
  /*! \brief Map of input names to input indices. */
  std::unordered_map<std::string, uint32_t> input_map_;
  /*! \brief Map of output names to output indices. */
  std::unordered_map<std::string, uint32_t> output_map_;
  /*! \brief Used for quick node input DLTensor* lookup given an input eid. */
  std::vector<std::vector<DLTensor*>> input_dltensors_;
  /*! \brief Used for quick node output DLTensor* lookup given an output eid. */
  std::vector<std::vector<DLTensor*>> output_dltensors_;
  /*! \brief Used for quick node(both model output and op input) DLTensor* lookup given an eid. */
  std::vector<std::vector<DLTensor*>> both_output_opinput_dltensors_;
  /*! \brief Used for quick node output DLTensor* lookup given a nop's input eid. */
  std::unordered_map<int, std::vector<DLTensor*>> node_output_dltensors_;
  /*! \brief Used for quick entry_id lookup given an storage_id. */
  std::vector<std::vector<uint32_t>> sid_to_eid_;
  /*! \brief Used for quick entry indexing. */
  std::vector<uint32_t> node_row_ptr_;
  /*! \brief Output entries. */
  std::vector<NodeEntry> outputs_;
  /*! \brief Additional graph attributes. */
  GraphAttr attrs_;
  /*! \brief The code module that contains both host and device code. */
  tvm::runtime::Module module_;
  /*! \brief Execution context of all devices including the host. */
  std::vector<Device> devices_;
  /*! \brief Common storage pool for all devices. */
  std::vector<NDArray> storage_pool_;
  /*! \brief Data entry of each node. */
  std::vector<NDArray> data_entry_;
  /*! \brief Data alignment of each node. */
  std::vector<size_t> data_alignment_;
  /*! \brief Operator on each node. */
  std::vector<std::function<void()>> op_execs_;
  /*! \brief Linked parameter lookup function. */
  PackedFunc lookup_linked_param_;
  /*! \brief Module's _lookup_linked_param function, used by DefaultLookupLinkedParam. */
  PackedFunc module_lookup_linked_param_;
  /*!
   * \brief True when module_lookup_linked_param_ is valid.
   * When the module does not include linked parmeters, module_lookup_linked_param_ will be nullptr.
   */
  bool module_lookup_linked_param_valid_;
};

std::vector<Device> GetAllDevice(const TVMArgs& args, int dev_start_arg);
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_GRAPH_EXECUTOR_GRAPH_EXECUTOR_H_
