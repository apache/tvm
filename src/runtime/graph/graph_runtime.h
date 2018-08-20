/*!
 *  Copyright (c) 2017 by Contributors
 *
 * \brief Tiny graph runtime that can run graph
 *        containing only tvm PackedFunc.
 * \file graph_runtime.h
 */
#ifndef TVM_RUNTIME_GRAPH_GRAPH_RUNTIME_H_
#define TVM_RUNTIME_GRAPH_GRAPH_RUNTIME_H_

#include <dmlc/memory_io.h>
#include <dmlc/json.h>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

/*! \brief macro to do C API call */
#define TVM_CCALL(func)                                            \
  {                                                                \
    int ret = (func);                                              \
    CHECK_EQ(ret, 0)                                               \
        << TVMGetLastError();                                      \
  }
/*! \brief Magic number for NDArray list file  */
constexpr uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;

/*! \brief operator attributes about tvm op */
struct TVMOpParam {
  std::string func_name;
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t flatten_data;
};

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
  virtual PackedFunc GetFunction(const std::string& name,
                                 const std::shared_ptr<ModuleNode>& sptr_to_self);

  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const final{
    return "GraphRuntime";
  };

  void Run();

  /*!
   * \brief Initialize the graph executor with graph and context.
   * \param graph_json The execution graph.
   * \param module The module containing the compiled functions.
   * \param ctx The context where the graph should sit on
   */
  void Init(const std::string& graph_json,
            tvm::runtime::Module module,
            TVMContext ctx);
  /*!
   * \brief Get the input index given the name of input.
   * \param name The name of the input.
   * \return The index of input.
   */
  int GetInputIndex(const std::string& name);

  /*!
   * \brief set index-th input to the graph.
   * \param index The input index.
   * \param data_in The input data.
   */
  void SetInput(int index, DLTensor* data_in);

  /*!
   * \brief Copy index-th input to data_out
   * \param index The input index.
   * \param data_out The output
   */
  void GetInput(int index, DLTensor* data_out);

  /*!
   * \brief Copy index-th output to data_out.
   * \param index The output index.
   * \param data_out the output data.
   */
  void GetOutput(int index, DLTensor* data_out);

  /*!
   * \brief Get the node index given the name of node.
   * \param name The name of the node.
   * \return The index of node.
   */
  int GetNodeIndex(const std::string& name);

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
  void DebugGetNodeOutput(int index, DLTensor* data_out);

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
   * \brief Get the tensor vector pointer.
   */
  std::vector<DLTensor>& data_entry() {
      return data_entry_;
  }

  /*!
   * \brief Get the execution function pointer.
   */
  std::vector<std::function<void()> >& op_execs() {
        return op_execs_;
  }

  /*!
   * \brief Get the number of outputs of a node for a valid optype.
   * \param index Index of the nodes.
   */
  size_t NumOutputs(int index) {
      return (nodes_[index].op_type == "null") ? 1: nodes_[index].param.num_outputs;
  }

  /*!
   * \brief Get node entry index.
   * \param nid Node id.
   * \param index Index of the nodes.
   */
  uint32_t GetEntryId(uint32_t nid, uint32_t index) const {
    return node_row_ptr_[nid] + index;
  }

  /*!
   * \brief Get Ctx.
   * \return Returns the TVMContext
   */
  TVMContext GetCtx() {
    return ctx_;
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
  uint32_t entry_id(const NodeEntry& e) const {
    return GetEntryId(e.node_id, e.index);
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

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_GRAPH_GRAPH_RUNTIME_H_
