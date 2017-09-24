/*!
 *  Copyright (c) 2017 by Contributors
 *
 *  Runtime module for graph deployment.
 *
 * \file graph_executor.h
 */
#ifndef NNVM_RUNTIME_GRAPH_EXECUTOR_H_
#define NNVM_RUNTIME_GRAPH_EXECUTOR_H_

#include <dmlc/io.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/module.h>
#include <nnvm/graph.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/tuple.h>
#include <nnvm/pass.h>
#include <vector>
#include <string>
#include "./graph_runtime.h"

namespace nnvm {
namespace runtime {

/*!
 * \brief TVM Graph Executor.
 *  This is a minimum graph executor, embedded in TVM runtime
 *  without any framework dependency.
 *
 *  This runtime can be acccesibly in various language via
 *  TVM runtime PackedFunc API.
 */
class GraphExecutor : public ::tvm::runtime::ModuleNode {
 public:
  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const final {
    return "GraphExecutor";
  }
  /*!
   * \brief Get member function to front-end
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  tvm::runtime::PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final;
  /*! \brief destructor */
  ~GraphExecutor();
  /*!
   * \brief Initialize the graph executor with graph and context.
   * \param graph The execution graph.
   * \param module The module containing the compiled functions.
   * \param ctx The context where the graph should sit on
   */
  void Init(Graph graph,
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
   * \param data The input data.
   */
  void SetInput(int index, DLTensor* data);
  /*!
   * \brief Copy index-th output to data_out.
   * \param index The output index.
   * \param data_out the output data.
   */
  void GetOutput(int index, DLTensor* data_out);
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
   * \brief Execute the graph, update output.
   */
  void Run();

 private:
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
  std::function<void()> CreateTVMOp(const NodeAttrs& attrs,
                                    const std::vector<DLTensor>& args,
                                    size_t num_inputs);
  /*! \brief The graph */
  Graph graph_;
  /*! \brief The code module */
  tvm::runtime::Module module_;
  /*! \brief execution context */
  TVMContext ctx_;
  /*! \brief common storage pool */
  std::vector<DLTensor*> storage_pool_;
  /*! \brief data shape of each node entry */
  std::vector<TShape> data_shape_;
  /*! \brief data entry of each node */
  std::vector<DLTensor> data_entry_;
  /*! \brief operator on each node */
  std::vector<std::function<void()> > op_execs_;
};

}  // namespace runtime
}  // namespace nnvm

#endif  // NNVM_RUNTIME_GRAPH_EXECUTOR_H_
