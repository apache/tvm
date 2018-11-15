/*!
 *  Copyright (c) 2018 by Contributors
 *
 * \brief TensorRT executor manager. The manager is owned
 * by GraphRuntime if TensorRT is enabled in the build. The manager
 * owns TensorRT inference engines and contexts.
 * \file tensorrt_executor.h
 */
#ifndef TVM_CONTRIB_SUBGRAPH_TENSORRT_EXECUTOR_H_
#define TVM_CONTRIB_SUBGRAPH_TENSORRT_EXECUTOR_H_

#include <dlpack/dlpack.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <NvInfer.h>

namespace tvm {
namespace contrib {

/*!
 * This class manages TensorRT executors(engines) of all the subgraphs in
 * a computational graph consumed by a TVM GraphRuntime. The TVM GraphRuntime
 * owns the lifecycle of the object of this class.
 */
class TensorRTExecManager {
 public:
  TensorRTExecManager();
  /*!
   * Creates an executor for the subgraph node invoked by the TVM GraphRuntime::Run().
   * @param subgraph_name Subgraph node name
   * @param subgraph Subgraph structure
   * @param args Inputs and outputs of the subgraph in the topological sorted order
   * @return
   */
  std::function<void()> CreateExec(const std::string& subgraph_name,
                                   const Subgraph& subgraph,
                                   const std::vector<DLTensor>& args);
  ~TensorRTExecManager();

 private:
  /*!
   * Create a TensorRT inference engine for a subgraph.
   * @param subgraph Subgraph structure
   * @param data_entries Inputs and outputs of the subgraph in the topological sorted order
   * @param input_data_idx Input data indices in data_entries
   * @param input_data_names Input data names, same order as in input_data_idx
   * @param output_names Output names of the subgraph
   * @return
   */
  nvinfer1::ICudaEngine* CreateInferEngine(
      const Subgraph& subgraph,
      const std::vector<DLTensor>& data_entries,
      std::vector<uint32_t>* input_data_idx,
      std::vector<std::string>* input_data_names,
      std::vector<std::string>* output_names);
  nvinfer1::IBuilder* infer_engine_builder_;
  std::unordered_map<std::string, nvinfer1::ICudaEngine*> infer_engine_map_;
  std::unordered_map<nvinfer1::ICudaEngine*, nvinfer1::IExecutionContext*>
      infer_engine_context_map_;
  std::unordered_map<nvinfer1::ICudaEngine*, std::vector<uint32_t> > input_data_idx_map_;
  std::unordered_map<nvinfer1::ICudaEngine*, std::vector<std::string> > input_data_name_map_;
  std::unordered_map<nvinfer1::ICudaEngine*, std::vector<std::string> > output_name_map_;
  /*! Max temporary memory for TensorRT to run a graph. */
  size_t max_workspace_size_;
  /*! Use FP16 kernels in TensorRT when it is true. */
  bool use_fp16_;
  /*! Use profiler of TensorRT if it is true. */
  bool use_profiler_;
};

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_SUBGRAPH_TENSORRT_EXECUTOR_H_
