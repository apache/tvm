/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_runtime.cc
 */
#include "graph_runtime.h"

#include <dlpack/dlpack.h>
#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

namespace tvm {
namespace runtime {

/*! \brief Macro to do C API call. */
#define TVM_CCALL(func)                                            \
  {                                                                \
    int ret = (func);                                              \
    CHECK_EQ(ret, 0)                                               \
        << TVMGetLastError();                                      \
  }

/*!
 * \brief Tiny graph runtime.
 *
 *  This runtime can be acccesibly in various language via
 *  TVM runtime PackedFunc API.
 */
class GraphRuntime : public ModuleNode {
 public:
  /*!
   * \brief Get member function to front-end.
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final;

  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const final {
    return "GraphRuntime";
  }
  void Run() {
    // setup the array and requirements.
    for (size_t i = 0; i < op_execs_.size(); ++i) {
      if (op_execs_[i]) op_execs_[i]();
    }
  }
  /*!
   * \brief Initialize the graph executor with graph and context.
   * \param graph_json The execution graph.
   * \param module The module containing the compiled functions for the host
   * processor.
   * \param ctxs The context of the host and devices where graph nodes will be
   * executed on.
   */
  void Init(const std::string& graph_json, const tvm::runtime::Module& module,
            const std::vector<TVMContext>& ctxs) {
#ifndef _LIBCPP_SGX_NO_IOSTREAMS
    std::istringstream is(graph_json);
#else
    std::string is = graph_json;
#endif
    dmlc::JSONReader reader(&is);
    this->Load(&reader);
    module_ = module;
    ctxs_ = ctxs;
    this->SetupStorage();
    this->SetupOpExecs();
  }

  /*!
   * \brief Get the input index given the name of input.
   * \param name The name of the input.
   * \return The index of input.
   */
  int GetInputIndex(const std::string& name) {
    for (size_t i = 0; i< input_nodes_.size(); ++i) {
      uint32_t nid = input_nodes_[i];
      if (nodes_[nid].name == name) {
        return static_cast<int>(i);
      }
    }
    LOG(WARNING) << "Warning: cannot find \"" << name << "\" among input";
    return -1;
  }
  /*!
   * \brief Set index-th input to the graph.
   * \param index The input index.
   * \param data_in The input data.
   */
  void SetInput(int index, DLTensor* data_in) {
    CHECK_LT(static_cast<size_t>(index), input_nodes_.size());
    uint32_t eid = this->entry_id(input_nodes_[index], 0);
    data_entry_[eid].CopyFrom(data_in);
  }
  /*!
   * \brief Get the number of outputs
   *
   * \return The number of outputs from graph.
   */
  int NumOutputs() const {
    return outputs_.size();
  }
  /*!
   * \brief Return NDArray for given input index.
   * \param index The input index.
   *
   * \return NDArray corresponding to given input node index.
   */
  NDArray GetInput(int index) {
    CHECK_LT(static_cast<size_t>(index), input_nodes_.size());
    uint32_t eid = this->entry_id(input_nodes_[index], 0);
    return data_entry_[eid];
  }
  /*!
   * \brief Return NDArray for given output index.
   * \param index The output index.
   *
   * \return NDArray corresponding to given output node index.
   */
  NDArray GetOutput(int index) {
    CHECK_LT(static_cast<size_t>(index), outputs_.size());
    uint32_t eid = this->entry_id(outputs_[index]);
    return data_entry_[eid];
  }
  /*!
   * \brief Copy index-th output to data_out.
   * \param index The output index.
   * \param data_out The output data.
   */
  void CopyOutputTo(int index, DLTensor* data_out) {
    CHECK_LT(static_cast<size_t>(index), outputs_.size());
    uint32_t eid = this->entry_id(outputs_[index]);

    // Check the shapes to avoid receiving in different dimension but same size.
    const NDArray& data = data_entry_[eid];
    CHECK_EQ(data->ndim, data_out->ndim);
    for (int32_t j = 0; j < data->ndim; ++j) {
      CHECK_EQ(data->shape[j], data_out->shape[j]);
    }

    data_entry_[eid].CopyTo(data_out);
  }
#ifdef TVM_GRAPH_RUNTIME_DEBUG
  /*!
   * \brief Get the node index given the name of node.
   * \param name The name of the node.
   * \return The index of node.
   */
  int GetNodeIndex(const std::string& name) {
    for (uint32_t nid = 0; nid< nodes_.size(); ++nid) {
      if (nodes_[nid].name == name) {
        return static_cast<int>(nid);
      }
    }
    LOG(FATAL) << "cannot find " << name << " among nodex";
    return -1;
  }

  /*!
   * \brief Copy index-th node to data_out.
   *
   * This method will do a partial run of the the graph
   * from begining upto the index-th node and return output of index-th node.
   * This is costly operation and suggest to use only for debug porpose.
   *
   * \param index The index of the node.
   * \param data_out The node data.
   */
  void DebugGetNodeOutput(int index, DLTensor* data_out) {
    CHECK_LT(static_cast<size_t>(index), nodes_.size());
    uint32_t eid = index;

    for (size_t i = 0; i < op_execs_.size(); ++i) {
      if (op_execs_[i]) op_execs_[i]();
      if (static_cast<int>(i) == index) break;
    }

    data_entry_[eid].CopyTo(data_out);
  }
#endif
  /*!
   * \brief Load parameters from binary stream.
   * \param strm The input stream.
   */
  void LoadParams(dmlc::Stream* strm);
  /*!
   * \brief Load parameters from parameter blob.
   * \param param_blob A binary blob of parameter.
   */
  void LoadParams(const std::string& param_blob) {
    dmlc::MemoryStringStream strm(const_cast<std::string*>(&param_blob));
    this->LoadParams(&strm);
  }

 private:
  // Memory pool entry.
  struct PoolEntry {
    size_t size;
    int device_type;
    PoolEntry(int s, int dev_type) : size(s), device_type(dev_type) {}
  };
  // Node entry
  struct NodeEntry {
    uint32_t node_id;
    uint32_t index;
    uint32_t version;
    // JSON Loader
    void Load(dmlc::JSONReader *reader) {
      reader->BeginArray();
      CHECK(reader->NextArrayItem()) << "invalid json format";
      reader->Read(&node_id);
      CHECK(reader->NextArrayItem()) << "invalid json format";
      reader->Read(&index);
      if (reader->NextArrayItem()) {
        reader->Read(&version);
        CHECK(!reader->NextArrayItem()) << "invalid json format";
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
    void LoadAttrs(dmlc::JSONReader *reader, TVMOpParam* param) {
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
        }
      }
      CHECK_EQ(bitmask, 1|2|4|8) << "invalid format";
    }
    // JSON Loader
    void Load(dmlc::JSONReader *reader) {
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
      CHECK_EQ(bitmask, 1|2|4) << "invalid format";
    }
  };
  struct GraphAttr {
    size_t storage_num_not_alloctaed{0};
    std::vector<int> storage_id;
    std::vector<int> device_index;
    std::vector<std::string> dltype;
    std::vector<std::vector<int64_t> > shape;
    // The graph attribute fields.
    void Load(dmlc::JSONReader *reader) {
      reader->BeginObject();
      int bitmask = 0;
      std::string key, type;
      while (reader->NextObjectItem(&key)) {
        if (key == "dltype") {
          reader->BeginArray();
          CHECK(reader->NextArrayItem());
          reader->Read(&type);
          CHECK_EQ(type, "list_str");
          CHECK(reader->NextArrayItem());
          reader->Read(&dltype);
          CHECK(!reader->NextArrayItem());
          bitmask |= 1;
        } else if (key == "storage_id") {
          reader->BeginArray();
          CHECK(reader->NextArrayItem());
          reader->Read(&type);
          CHECK_EQ(type, "list_int");
          CHECK(reader->NextArrayItem());
          reader->Read(&storage_id);
          CHECK(!reader->NextArrayItem());
          bitmask |= 2;
        } else if (key == "shape") {
          reader->BeginArray();
          CHECK(reader->NextArrayItem());
          reader->Read(&type);
          CHECK_EQ(type, "list_shape");
          CHECK(reader->NextArrayItem());
          reader->Read(&shape);
          CHECK(!reader->NextArrayItem());
          bitmask |= 4;
        } else if (key == "device_index") {
          reader->BeginArray();
          CHECK(reader->NextArrayItem());
          reader->Read(&type);
          CHECK_EQ(type, "list_int");
          CHECK(reader->NextArrayItem());
          reader->Read(&device_index);
          CHECK(!reader->NextArrayItem());
        } else {
          reader->BeginArray();
          CHECK(reader->NextArrayItem());
          reader->Read(&type);
          if (type == "list_int") {
            CHECK(reader->NextArrayItem());
            std::vector<int> temp;
            reader->Read(&temp);
          } else if (type == "size_t") {
            CHECK(reader->NextArrayItem());
            size_t temp;
            reader->Read(&temp);
          } else {
            LOG(FATAL) << "cannot skip graph attr " << key;
          }
          CHECK(!reader->NextArrayItem());
        }
      }
      CHECK_EQ(bitmask, 1|2|4) << "invalid format";
    }
  };
  // The graph attribute fields.
  void Load(dmlc::JSONReader *reader) {
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
        } else {
          LOG(FATAL) << "key " << key << " is not supported";
        }
      }
      CHECK_EQ(bitmask, 1|2|4|8|16) << "invalid format";
  }
  /*! \brief Setup the temporal storage */
  void SetupStorage();
  /*! \brief Setup the executors. */
  void SetupOpExecs();
  /*!
   * \brief Create a executtion function given input.
   * \param attrs The node attributes.
   * \param args The arguments to the functor, including inputs and outputs.
   * \param num_inputs Number of inputs.
   * \param dev_type The device type of the tvm_op.
   * \return The created executor.
   */
  std::function<void()> CreateTVMOp(const TVMOpParam& attrs,
                                    const std::vector<DLTensor>& args,
                                    size_t num_inputs);
  // Get node entry index.
  uint32_t entry_id(uint32_t nid, uint32_t index) const {
    return node_row_ptr_[nid] + index;
  }
  // Get node entry index.
  uint32_t entry_id(const NodeEntry& e) const {
    return entry_id(e.node_id, e.index);
  }
  // Number of node entries.
  uint32_t num_node_entries() const {
    return node_row_ptr_.back();
  }
  // Number of nodes.
  uint32_t num_nodes() const {
    return static_cast<uint32_t>(nodes_.size());
  }
  /*! \brief The graph nodes. */
  std::vector<Node> nodes_;
  /*! \brief The argument nodes. */
  std::vector<uint32_t> input_nodes_;
  /*! \brief Used for quick entry indexing. */
  std::vector<uint32_t> node_row_ptr_;
  /*! \brief Output entries. */
  std::vector<NodeEntry> outputs_;
  /*! \brief Additional graph attributes. */
  GraphAttr attrs_;
  /*! \brief The code module that contains both host and device code. */
  tvm::runtime::Module module_;
  /*! \brief Execution context of all devices including the host. */
  std::vector<TVMContext> ctxs_;
  /*! \brief Common storage pool for all devices. */
  std::vector<NDArray> storage_pool_;
  /*! \brief Data entry of each node. */
  std::vector<NDArray> data_entry_;
  /*! \brief Operator on each node. */
  std::vector<std::function<void()> > op_execs_;
};

void GraphRuntime::LoadParams(dmlc::Stream* strm) {
  uint64_t header, reserved;
  CHECK(strm->Read(&header))
      << "Invalid parameters file format";
  CHECK(header == kTVMNDArrayListMagic)
      << "Invalid parameters file format";
  CHECK(strm->Read(&reserved))
      << "Invalid parameters file format";

  std::vector<std::string> names;
  CHECK(strm->Read(&names))
      << "Invalid parameters file format";
  uint64_t sz;
  strm->Read(&sz);
  size_t size = static_cast<size_t>(sz);
  CHECK(size == names.size())
      << "Invalid parameters file format";
  for (size_t i = 0; i < size; ++i) {
    int in_idx = GetInputIndex(names[i]);
    CHECK_GE(in_idx, 0) << "Found param for non-existent input: " << names[i];
    uint32_t eid = this->entry_id(input_nodes_[in_idx], 0);
    CHECK_LT(eid, data_entry_.size());

    // The data_entry is allocated on device, NDArray.load always load the array into CPU.
    NDArray temp;
    temp.Load(strm);
    data_entry_[eid].CopyFrom(temp);
  }
}

void GraphRuntime::SetupStorage() {
  // Grab saved optimization plan from graph.
  std::vector<TVMType> vtype;
  for (const std::string& s_type : attrs_.dltype) {
    vtype.push_back(tvm::runtime::String2TVMType(s_type));
  }

  // Size and device type of each storage pool entry.
  std::vector<PoolEntry> pool_entry;
  // Find the maximum space size.
  for (size_t i = 0; i < attrs_.shape.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    // Use the fallback device if no device index is available.
    int device_type = static_cast<int>(ctxs_[0].device_type);
    if (!attrs_.device_index.empty()) {
      device_type = attrs_.device_index[i];
    }
    size_t size = 1;
    for (int64_t sz : attrs_.shape[i]) {
      size *= static_cast<size_t>(sz);
    }
    CHECK_GE(storage_id, 0) << "Do not support runtime shape op";
    DLDataType t = vtype[i];
    size_t bits = t.bits * t.lanes;
    CHECK_EQ(bits % 8U, 0U);
    size_t bytes = (bits / 8U) * size;

    uint32_t sid = static_cast<uint32_t>(storage_id);
    if (sid >= pool_entry.size()) {
      pool_entry.resize(sid + 1, {0, -1});
    } else {
      CHECK(pool_entry[sid].device_type == -1 ||
            pool_entry[sid].device_type == device_type)
          << "The same pool entry cannot be assigned to multiple devices";
    }
    pool_entry[sid].size = std::max(pool_entry[sid].size, bytes);
    pool_entry[sid].device_type = device_type;
  }

  // Allocate the space.
  for (const auto& pit : pool_entry) {
    std::vector<int64_t> shape;
    // This for loop is very fast since there are usually only a couple of
    // devices available on the same hardware.
    const auto& cit =
        std::find_if(ctxs_.begin(), ctxs_.end(), [&pit](const TVMContext& c) {
          return pit.device_type == static_cast<int>(c.device_type);
        });
    TVMContext ctx = cit == ctxs_.end() ? ctxs_[0] : *cit;
    shape.push_back(static_cast<int64_t>(pit.size + 3) / 4);
    storage_pool_.push_back(
        NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, ctx));
  }

  // Assign the pooled entries. A unified memory pool is used to simplifiy
  // memory assignment for each node entry. The allocated memory on each device
  // is mapped to this pool.
  data_entry_.resize(num_node_entries());
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    CHECK_LT(static_cast<size_t>(storage_id), storage_pool_.size());
    data_entry_[i] =
        storage_pool_[storage_id].CreateView(attrs_.shape[i], vtype[i]);
  }
}

void GraphRuntime::SetupOpExecs() {
  op_execs_.resize(this->num_nodes());
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < this->num_nodes(); ++nid) {
    const auto& inode = nodes_[nid];
    if (inode.op_type == "null") continue;
    std::vector<DLTensor> args;
    for (const auto& e : inode.inputs) {
      args.push_back(*(data_entry_[this->entry_id(e)].operator->()));
    }
    for (uint32_t index = 0; index < inode.param.num_outputs; ++index) {
      uint32_t eid = this->entry_id(nid, index);
      args.push_back(*(data_entry_[eid].operator->()));
    }
    CHECK(inode.op_type == "tvm_op") << "Can only take tvm_op as op";

    op_execs_[nid] = CreateTVMOp(inode.param, args, inode.inputs.size());
  }
}

std::function<void()> GraphRuntime::CreateTVMOp(
    const TVMOpParam& param,
    const std::vector<DLTensor>& args,
    size_t num_inputs) {
  struct OpArgs {
    std::vector<DLTensor> args;
    std::vector<TVMValue> arg_values;
    std::vector<int> arg_tcodes;
    std::vector<int64_t> shape_data;
  };
  std::shared_ptr<OpArgs> arg_ptr = std::make_shared<OpArgs>();
  // setup address.
  arg_ptr->args = std::move(args);
  if (param.flatten_data) {
    arg_ptr->shape_data.resize(arg_ptr->args.size());
  }
  for (size_t i = 0; i < arg_ptr->args.size(); ++i) {
    TVMValue v;
    DLTensor* t = &(arg_ptr->args[i]);
    v.v_handle = t;
    arg_ptr->arg_values.push_back(v);
    arg_ptr->arg_tcodes.push_back(kArrayHandle);
    if (param.flatten_data) {
      arg_ptr->shape_data[i] = std::accumulate(
          t->shape, t->shape + t->ndim, 1, std::multiplies<int64_t>());
      t->ndim = 1;
      t->shape = &(arg_ptr->shape_data[i]);
    }
  }

  if (param.func_name == "__nop") {
    return [](){};
  } else if (param.func_name == "__copy") {
    // Perform cross device data copy.
    // Directly copy data from the input to the output.
    auto fexec = [arg_ptr]() {
      DLTensor* from = static_cast<DLTensor*>(arg_ptr->arg_values[0].v_handle);
      DLTensor* to = static_cast<DLTensor*>(arg_ptr->arg_values[1].v_handle);
      TVM_CCALL(TVMArrayCopyFromTo(from, to, nullptr));
    };
    return fexec;
  }

  // Get compiled function from the module that contains both host and device
  // code.
  tvm::runtime::PackedFunc pf = module_.GetFunction(param.func_name, false);
  CHECK(pf != nullptr) << "no such function in module: " << param.func_name;

  auto fexec = [arg_ptr, pf]() {
    TVMRetValue rv;
    TVMArgs targs(arg_ptr->arg_values.data(),
                  arg_ptr->arg_tcodes.data(),
                  static_cast<int>(arg_ptr->arg_values.size()));
    pf.CallPacked(targs, &rv);
  };
  return fexec;
}

PackedFunc GraphRuntime::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  // Return member functions during query.
  if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        if (args[0].type_code() == kStr) {
          int in_idx = this->GetInputIndex(args[0]);
          if (in_idx >= 0) this->SetInput(in_idx, args[1]);
        } else {
          this->SetInput(args[0], args[1]);
        }
      });
  } else if (name == "get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        if (args.num_args == 2) {
          this->CopyOutputTo(args[0], args[1]);
        } else {
          *rv = this->GetOutput(args[0]);
        }
      });
  } else if (name == "get_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        int in_idx = 0;
        if (args[0].type_code() == kStr) {
          in_idx = this->GetInputIndex(args[0]);
        } else {
          in_idx = args[0];
        }
        CHECK_GE(in_idx, 0);
        *rv = this->GetInput(in_idx);
      });
  } else if (name == "get_num_outputs") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->NumOutputs();
      });
#ifdef TVM_GRAPH_RUNTIME_DEBUG
  } else if (name == "debug_get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        if (args[0].type_code() == kStr) {
          this->DebugGetNodeOutput(this->GetNodeIndex(args[0]), args[1]);
        } else {
          this->DebugGetNodeOutput(args[0], args[1]);
        }
      });
#endif
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        this->Run();
      });
  } else if (name == "load_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        this->LoadParams(args[0].operator std::string());
      });
  } else {
    return PackedFunc();
  }
}

Module GraphRuntimeCreate(const std::string& sym_json,
                          const tvm::runtime::Module& m,
                          const std::vector<TVMContext>& ctxs) {
  std::shared_ptr<GraphRuntime> exec = std::make_shared<GraphRuntime>();
  exec->Init(sym_json, m, ctxs);
  return Module(exec);
}

// Get all context for the host and other runtime devices.
std::vector<TVMContext> GetAllContext(const TVMArgs& args) {
  // Reserve the first item as the fallback device.
  std::vector<TVMContext> ret;
  TVMContext ctx;
  for (int i = 2; i < args.num_args; i += 2) {
    int dev_type = args[i];
    ctx.device_type = static_cast<DLDeviceType>(dev_type);
    ctx.device_id = args[i + 1];
    ret.push_back(ctx);
  }
  return ret;
}

// 4-argument version is currently reserved to keep support of calling
// from tvm4j and javascript, since they don't have heterogeneous
// execution support yet. For heterogenenous execution, at least 5 arguments will
// be passed in. The third one is the number of devices.
// Eventually, we will only probably pass TVMContext for all the languages.
TVM_REGISTER_GLOBAL("tvm.graph_runtime.create")
  .set_body([](TVMArgs args, TVMRetValue* rv) {
    CHECK_GE(args.num_args, 4)
        << "The expected number of arguments for graph_runtime.create is "
           "at least 4, but it has "
        << args.num_args;
    const auto& contexts = GetAllContext(args);
    *rv = GraphRuntimeCreate(args[0], args[1], contexts);
  });

TVM_REGISTER_GLOBAL("tvm.graph_runtime.remote_create")
  .set_body([](TVMArgs args, TVMRetValue* rv) {
    CHECK_GE(args.num_args, 4) << "The expected number of arguments for "
                                  "graph_runtime.remote_create is "
                                  "at least 4, but it has "
                               << args.num_args;
    void* mhandle = args[1];
    const auto& contexts = GetAllContext(args);
    *rv = GraphRuntimeCreate(
        args[0], *static_cast<tvm::runtime::Module*>(mhandle), contexts);
  });
}  // namespace runtime
}  // namespace tvm
