/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_executor.h
 */
#ifndef TVM_GRAPH_EXECUTOR_H_
#define TVM_GRAPH_EXECUTOR_H_

#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <nnvm/graph.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/tuple.h>
#include <nnvm/pass.h>
#include <numeric>
#include <string>

namespace tvm {
namespace contrib {

using tvm::runtime::TVMArgs;
using tvm::runtime::TVMRetValue;
using tvm::runtime::PackedFunc;
using nnvm::StorageVector;
using nnvm::ShapeVector;
using nnvm::TShape;
using nnvm::NodeAttrs;

/*! \brief DLPack compatible data types */
using DLTypeVector = std::vector<DLDataType>;

/*! \brief The executor function */
using FOpExec = std::function<void()>;

/*! \brief macro to do C API call */
#define TVM_CCALL(func)                                            \
  {                                                                \
    int ret = (func);                                              \
    CHECK_EQ(ret, 0)                                               \
        << TVMGetLastError();                                      \
  }

constexpr uint64_t kTVMNDArrayMagic     = 0xDD5E40F096B4A13F;
constexpr uint64_t kTVMNDArrayListMagic = 0xF7E58D4F05049CB7;

/*! \brief Graph Executor with TVM runtime */
class GraphExecutor : public runtime::ModuleNode {
 public:
  const char* type_key() const {
    return "GraphExecutor";
  }
  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self);
  // Destructor
  ~GraphExecutor();
  // Setup with a given graph
  void Init(const nnvm::Graph& g, TVMContext ctx);
  // Get index of variable
  int GetIndex(std::string name);
  // Copy data to index-th input
  void SetInput(int index, DLTensor* data_in);
  // Copy index-th output to data_out
  void GetOutput(int index, DLTensor* data_out);
  // Load parameters from stream
  void LoadParams(dmlc::Stream* strm);
  // Load parameters from binary file blob
  void LoadParamsFromBlob(std::string param_blob);
  // Execute the graph.
  void Run();

 private:
  // functions
  void SetupNameIndex();
  void SetupStorage();
  void SetupOpExecs();
  // Constructor to create TVM op
  FOpExec CreateTVMOp(const nnvm::NodeAttrs& attrs,
                      std::vector<DLTensor> inputs,
                      size_t num_inputs);
  // The graph to be executed.
  nnvm::Graph graph_;
  // The execution context
  TVMContext ctx_;
  // Common storage pool
  std::vector<DLTensor*> storage_pool_;
  // The data shape
  std::vector<TShape> data_shape_;
  // The data entry
  std::vector<DLTensor> data_entry_;
  // The operation lambda on each node
  std::vector<FOpExec> op_execs_;
  // The code module.
  tvm::runtime::Module module_;
  std::unordered_map<std::string, size_t> name_idx_;
};


struct TVMOpParam : public dmlc::Parameter<TVMOpParam> {
  std::string func_name;
  uint32_t num_inputs;
  uint32_t num_outputs;
  bool flatten_data;
  DMLC_DECLARE_PARAMETER(TVMOpParam) {
    DMLC_DECLARE_FIELD(func_name);
    DMLC_DECLARE_FIELD(num_inputs)
    .set_default(1);
    DMLC_DECLARE_FIELD(num_outputs)
    .set_default(1);
    DMLC_DECLARE_FIELD(flatten_data)
    .set_default(false);
  }
};
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_GRAPH_EXECUTOR_H_
