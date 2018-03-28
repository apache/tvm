/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_runtime.cc
 */
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <dmlc/memory_io.h>
#include <dmlc/json.h>
#include <numeric>
#include "./graph_runtime.h"

namespace tvm {
namespace runtime {

/*! \brief macro to do C API call */
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
  ~GraphRuntime() {
    for (DLTensor* t : storage_pool_) {
      TVM_CCALL(TVMArrayFree(t));
    }
  }
  /*!
   * \brief Get member function to front-end
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
   * \param module The module containing the compiled functions.
   * \param ctx The context where the graph should sit on
   */
  void Init(const std::string& graph_json,
            tvm::runtime::Module module,
            TVMContext ctx) {
#ifndef _LIBCPP_SGX_NO_IOSTREAMS
    std::istringstream is(graph_json);
#else
    std::string is = graph_json;
#endif
    dmlc::JSONReader reader(&is);
    this->Load(&reader);
    module_ = module;
    ctx_ = ctx;
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
   * \brief set index-th input to the graph.
   * \param index The input index.
   * \param data_in The input data.
   */
  void SetInput(int index, DLTensor* data_in) {
    CHECK_LT(static_cast<size_t>(index), input_nodes_.size());
    uint32_t eid = this->entry_id(input_nodes_[index], 0);
    TVM_CCALL(TVMArrayCopyFromTo(data_in, &data_entry_[eid], nullptr));
  }
  /*!
   * \brief Copy index-th input to data_out
   * \param index The input index.
   * \param data_out The output
   */
  void GetInput(int index, DLTensor* data_out) {
    CHECK_LT(static_cast<size_t>(index), input_nodes_.size());
    uint32_t eid = this->entry_id(input_nodes_[index], 0);
    TVM_CCALL(TVMArrayCopyFromTo(&data_entry_[eid], data_out, nullptr));
  }
  /*!
   * \brief Copy index-th output to data_out.
   * \param index The output index.
   * \param data_out the output data.
   */
  void GetOutput(int index, DLTensor* data_out) {
    CHECK_LT(static_cast<size_t>(index), outputs_.size());
    uint32_t eid = this->entry_id(outputs_[index]);
    TVM_CCALL(TVMArrayCopyFromTo(&data_entry_[eid], data_out, nullptr));
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
   * \param index: The  index of the node.
   * \param data_out the node data.
   */
  void DebugGetNodeOutput(int index, DLTensor* data_out) {
    CHECK_LT(static_cast<size_t>(index), nodes_.size());
    uint32_t eid = index;

    for (size_t i = 0; i < op_execs_.size(); ++i) {
      if (op_execs_[i]) op_execs_[i]();
      if (static_cast<int>(i) == index) break;
    }

    TVM_CCALL(TVMArrayCopyFromTo(&data_entry_[eid], data_out, nullptr));
  }
#endif
  /*!
   * \brief Load parameters from binary stream
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
      std::unordered_map<std::string, std::string> dict;
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
  void LoadDLTensor(dmlc::Stream* strm, DLTensor* tensor);
  /*! \brief Setup the temporal storage */
  void SetupStorage();
  /*! \brief Setup the executors */
  void SetupOpExecs();
  /*!
   * \brief Create a executtion function given input.
   * \param attrs The node attributes
   * \param args The arguments to the functor, including inputs and outputs.
   * \param num_inputs Number of inputs
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
  // Number of node entries
  uint32_t num_node_entries() const {
    return node_row_ptr_.back();
  }
  // Number of nodes.
  uint32_t num_nodes() const {
    return static_cast<uint32_t>(nodes_.size());
  }
  // The graph nodes.
  std::vector<Node> nodes_;
  // The argument nodes.
  std::vector<uint32_t> input_nodes_;
  // used or quick entry indexing
  std::vector<uint32_t> node_row_ptr_;
  // output entries
  std::vector<NodeEntry> outputs_;
  // Additional graph attributes
  GraphAttr attrs_;
  /*! \brief The code module */
  tvm::runtime::Module module_;
  /*! \brief execution context */
  TVMContext ctx_;
  /*! \brief common storage pool */
  std::vector<DLTensor*> storage_pool_;
  /*! \brief data entry of each node */
  std::vector<DLTensor> data_entry_;
  /*! \brief operator on each node */
  std::vector<std::function<void()> > op_execs_;
};


void GraphRuntime::LoadDLTensor(dmlc::Stream* strm, DLTensor* dst) {
  uint64_t header, reserved;
  CHECK(strm->Read(&header, sizeof(header)))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&reserved, sizeof(reserved)))
      << "Invalid DLTensor file format";
  CHECK(header == kTVMNDArrayMagic)
      << "Invalid DLTensor file format";

  DLTensor tensor;
  CHECK(strm->Read(&tensor.ctx, sizeof(tensor.ctx)))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&tensor.ndim, sizeof(tensor.ndim)))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&tensor.dtype, sizeof(tensor.dtype)))
      << "Invalid DLTensor file format";
  std::vector<int64_t> shape(tensor.ndim);
  if (tensor.ndim != 0) {
    CHECK(strm->Read(&shape[0], sizeof(int64_t) * tensor.ndim))
        << "Invalid DLTensor file format";
  }
  CHECK_EQ(tensor.ndim, dst->ndim) << "param dimension mismatch";
  CHECK(tensor.dtype.bits == dst->dtype.bits &&
        tensor.dtype.code == dst->dtype.code &&
        tensor.dtype.lanes == dst->dtype.lanes) << "param type mismatch";
  for (int i = 0; i < tensor.ndim; ++i) {
    CHECK_EQ(shape[i], dst->shape[i]) << "param shape mismatch";
  }
  size_t bits = dst->dtype.bits * dst->dtype.lanes;
  size_t size = (bits + 7) / 8;
  for (int i = 0; i < dst->ndim; ++i) {
    size *= dst->shape[i];
  }
  uint64_t data_byte_size;
  CHECK(strm->Read(&data_byte_size, sizeof(data_byte_size)))
      << "Invalid DLTensor file format";
  CHECK(data_byte_size == size)
      << "Invalid DLTensor file format";
  std::vector<uint8_t> bytes(data_byte_size + 1);
  CHECK(strm->Read(&bytes[0], data_byte_size))
      << "Invalid DLTensor file format";
  TVM_CCALL(TVMArrayCopyFromBytes(dst, &bytes[0], data_byte_size));
}

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
  strm->Read(&sz, sizeof(sz));
  size_t size = static_cast<size_t>(sz);

  CHECK(size == names.size())
      << "Invalid parameters file format";
  for (size_t i = 0; i < size; ++i) {
    int in_idx = GetInputIndex(names[i]);
    CHECK_GE(in_idx, 0) << "Found param for non-existent input: " << names[i];
    uint32_t eid = this->entry_id(input_nodes_[in_idx], 0);
    CHECK_LT(eid, data_entry_.size());
    LoadDLTensor(strm, &data_entry_[eid]);
  }
}

void GraphRuntime::SetupStorage() {
  // Grab saved optimization plan from graph.
  std::vector<TVMType> vtype;
  for (const std::string& s_type : attrs_.dltype) {
    vtype.push_back(tvm::runtime::String2TVMType(s_type));
  }
  data_entry_.resize(num_node_entries());
  // size of each storage pool entry
  std::vector<size_t> pool_entry_bytes;
  // Find the maximum space size.
  for (size_t i = 0; i < attrs_.shape.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    size_t size = 1;
    for (int64_t sz : attrs_.shape[i]) {
      size *= static_cast<size_t>(sz);
    }
    CHECK_GE(storage_id, 0) << "Do not support runtime shape op";
    DLDataType t = vtype[i];
    size_t bits = t.bits * t.lanes;
    CHECK_EQ(bits % 8U, 0U);
    size_t bytes = (bits / 8U) * size;

    size_t sid = static_cast<size_t>(storage_id);
    if (sid >= pool_entry_bytes.size()) {
      pool_entry_bytes.resize(sid + 1, 0);
    }
    pool_entry_bytes[sid] = std::max(pool_entry_bytes[sid], bytes);
  }
  // Allocate the space.
  for (size_t i = 0; i < pool_entry_bytes.size(); ++i) {
    int64_t shape[] = {static_cast<int64_t>(pool_entry_bytes[i] + 3) / 4};
    DLTensor* tensor;
    TVM_CCALL(TVMArrayAlloc(
        shape, 1, kDLFloat, 32, 1, ctx_.device_type, ctx_.device_id, &tensor));
    storage_pool_.push_back(tensor);
  }
  // Assign the pooled entries.
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    CHECK_LT(static_cast<size_t>(storage_id), storage_pool_.size());
    data_entry_[i] = *storage_pool_[storage_id];
    data_entry_[i].shape = const_cast<int64_t*>(attrs_.shape[i].data());
    data_entry_[i].ndim = static_cast<int>(attrs_.shape[i].size());
    data_entry_[i].dtype = vtype[i];
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
      args.push_back(data_entry_[this->entry_id(e)]);
    }
    for (uint32_t index = 0; index < inode.param.num_outputs; ++index) {
      uint32_t eid = this->entry_id(nid, index);
      args.push_back(data_entry_[eid]);
    }
    CHECK_EQ(inode.op_type, "tvm_op")
        << "Can only take tvm_op as op";
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
  }
  // get compiled function from module.
  tvm::runtime::PackedFunc pf = module_.GetFunction(param.func_name, false);
  CHECK(pf != nullptr) << "no such function in module: " << param.func_name;
  auto fexec = [arg_ptr, pf] () {
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
  // return member functions during query.
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
        this->GetOutput(args[0], args[1]);
      });
  } else if (name == "get_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        if (args[0].type_code() == kStr) {
          int in_idx = this->GetInputIndex(args[0]);
          CHECK_GE(in_idx, 0);
          this->GetInput(in_idx, args[1]);
        } else {
          this->GetInput(args[0], args[1]);
        }
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

Module GraphRuntimeCreate(std::string sym_json,
                          tvm::runtime::Module m,
                          int device_type,
                          int device_id) {
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id   = device_id;
  std::shared_ptr<GraphRuntime> exec = std::make_shared<GraphRuntime>();
  exec->Init(sym_json, m, ctx);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.graph_runtime.create")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    *rv = GraphRuntimeCreate(args[0], args[1], args[2], args[3]);
  });

TVM_REGISTER_GLOBAL("tvm.graph_runtime.remote_create")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    void* mhandle = args[1];
    *rv = GraphRuntimeCreate(args[0],
                             *static_cast<tvm::runtime::Module*>(mhandle),
                             args[2], args[3]);
  });
}  // namespace runtime
}  // namespace tvm
