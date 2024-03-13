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

#ifndef TVM_RUNTIME_GRAPH_EXECUTOR_DEBUG_GRAPH_EXECUTOR_DEBUG_H_
#define TVM_RUNTIME_GRAPH_EXECUTOR_DEBUG_GRAPH_EXECUTOR_DEBUG_H_

#include <tvm/runtime/profiling.h>

#include <string>
#include <vector>

#include "../graph_executor.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Graph executor with debug .
 *
 *  This is the extension of GraphExecutor class used for debugging
 *  TVM runtime PackedFunc API.
 */
class GraphExecutorDebug : public GraphExecutor {
 public:
  /*!
   * \brief Run each operation in the graph and get the time per op for all ops.
   * \param number The number of times to run this function for taking average.
   * \param repeat The number of times to repeat the measurement.
   *        In total, the function will be invoked (1 + number x repeat) times,
   *        where the first one is warmed up and will be discarded in case
   *        there is lazy initialization.
   * \param min_repeat_ms The minimum duration of one `repeat` in milliseconds.
   *        By default, one `repeat` contains `number` runs. If this parameter is set,
   *        the parameters `number` will be dynamically adjusted to meet the
   *        minimum duration requirement of one `repeat`.
   * \param limit_zero_time_iterations The maximum number of repeats when
   *        measured time is equal to 0.  It helps to avoid hanging during
   *        measurements.
   * \param cooldown_interval_ms The cooldown interval in milliseconds between the number of repeats
   *        defined by `repeats_to_cooldown`.
   * \param repeats_to_cooldown The number of repeats before the
   *        cooldown is activated.
   * \return Returns a string with an encoded byte array. Where the first 8 bytes are int64_t
   * representing the number of layers. Next the encoded real numbers are float32_t in the number of
   * repeat multiplied by the number of layers.
   */
  std::string RunIndividual(int number, int repeat, int min_repeat_ms,
                            int limit_zero_time_iterations, int cooldown_interval_ms,
                            int repeats_to_cooldown);

  std::string RunIndividualNode(int node_index, int number, int repeat, int min_repeat_ms,
                                int limit_zero_time_iterations, int cooldown_interval_ms,
                                int repeats_to_cooldown);

  std::vector<double> RunOpRPC(int index, int number, int repeat, int min_repeat_ms,
                               int limit_zero_time_iterations, int cooldown_interval_ms,
                               int repeats_to_cooldown);

  Timer RunOpHost(int index);

  /*!
   * \brief GetFunction Get the function based on input.
   * \param name The function which needs to be invoked.
   * \param sptr_to_self Packed function pointer.
   */
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self);

  /*!
   * \brief Get the node index given the name of node.
   * \param name The name of the node.
   * \return The index of node.
   */
  int GetNodeIndex(const std::string& name) const;

  /*!
   * \brief Execute index-th node in the network.
   *
   * This method will do a partial run of the graph
   * up to index-th node.
   *
   * \param node: The index of the node.
   */
  void ExecuteNode(int node);

  /*!
   * \brief Returns index-th output of node.
   *
   * This method will return index-th out_ind output
   * of index-th node in the network.
   *
   * \param node: The index of the node.
   * \param out_ind: The index of the output.
   * \return Output array.
   */
  NDArray GetNodeOutput(int node, int out_ind);

  /*!
   * \brief Copy index-th node to data_out.
   *
   * This method will do a partial run of the graph
   * from begining upto the index-th node and return output of index-th node.
   * This is costly operation and suggest to use only for debug porpose.
   *
   * \param index: The  index of the node.
   * \param data_out the node data.
   */
  void DebugGetNodeOutput(int index, DLTensor* data_out);

  /*!
   * \brief return output of index-th node.
   *
   * This method will do a partial run of the graph
   * from begining up to the index-th node and return output of index-th node.
   * This is costly operation and suggest to use only for debug porpose.
   *
   * \param index: The  index of the node.
   *
   */
  NDArray DebugGetNodeOutput(int index);

  /*!
   * \brief Profile execution time of the module.
   *
   * We run the entire module while recording overall and per-op timing
   * information. The module may be run multiple times to ensure everything is
   * warmed up. This function is a more correct reflection of actual runtime of
   * the module compared to GraphRuntimeDebug::RunIndividual as it runs the
   * entire graph in order.
   *
   * \param collectors Optional user defined `MetricCollector`s to use with this profiling run.
   *
   * \returns A table of per-op runtimes and total times.
   */
  profiling::Report Profile(Array<profiling::MetricCollector> collectors);

 private:
  int last_executed_node_ = -1;
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_GRAPH_EXECUTOR_DEBUG_GRAPH_EXECUTOR_DEBUG_H_
