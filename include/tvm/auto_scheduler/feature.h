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
 * \file auto_scheduler/feature.h
 * \brief Feature extraction for the cost model.
 * We extract one feature vector per BufferStoreNode statement in a TIR Stmt,
 * so we call this feature as "per-store" feature.
 * The cost model also does prediction for each BufferStoreNode statement and aggregates
 * the predictions as the whole score for a TVM IR (Stmt).
 *
 * The feature specification is defined by `src/auto_scheduler/feature.cc:: FeatureSet`
 */

#ifndef TVM_AUTO_SCHEDULER_FEATURE_H_
#define TVM_AUTO_SCHEDULER_FEATURE_H_

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/auto_scheduler/measure.h>
#include <tvm/tir/function.h>

#include <string>
#include <vector>

namespace tvm {
namespace auto_scheduler {

/*!
 * \brief Get per-store features from a TIR PrimFunc
 * \param func The input lowered TIR PrimFunc
 * \param cache_line_size The size of cache line in bytes
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param ret The returned feature vector
 * \param log_scale Should the outputs be scaled by log2(1+x).
 */
void GetPerStoreFeature(const PrimFunc& func, int cache_line_size, int max_n_bufs,
                        std::vector<float>* ret, bool log_scale = true);

/*
 * \brief Get the names of elements in the feature vector. Use this for debug and inspection.
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param ret The returned names.
 */
void GetPerStoreFeatureName(int max_n_bufs, std::vector<std::string>* ret);

/*!
 * \brief Get per-store feature from states of the same task
 * \param states The input states
 * \param task The same search task for all states
 * \param skip_first_n_feature_extraction Skip feature extraction for the first n states
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param features The returned feature vector. The innermost vector contains the
 * feature vectors for all BufferStoreNode statements
 */
void GetPerStoreFeaturesFromStates(const Array<State>& states, const SearchTask& task,
                                   int skip_first_n_feature_extraction, int max_n_bufs,
                                   std::vector<std::vector<float>>* features);

/*!
 * \brief Get per-store feature from states of different tasks
 * \param states The input states
 * \param tasks The search tasks corresponding to the input states
 * \param skip_first_n_feature_extraction Skip feature extraction for the first n states
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param features The returned feature vector. The innermost vector contains the
 * feature vectors for all BufferStoreNode statements
 */
void GetPerStoreFeaturesFromStates(const Array<State>& states, const std::vector<SearchTask>& tasks,
                                   int skip_first_n_feature_extraction, int max_n_bufs,
                                   std::vector<std::vector<float>>* features);

/*!
 * \brief Get per-store features from a log file
 * \param filename The name of log file
 * \param max_lines Only read the first n lines of the file
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param features The returned feature vector. The innermost vector contains the
 * feature vectors for all BufferStoreNode statements
 * \param normalized_throughputs The normalized throughputs for all states
 * \param task_ids The task ids for all states
 */
void GetPerStoreFeaturesFromFile(const std::string& filename, int max_lines, int max_n_bufs,
                                 std::vector<std::vector<float>>* features,
                                 std::vector<float>* normalized_throughputs,
                                 std::vector<int>* task_ids);

/*!
 * \brief Get per-store features from measurement input/result pairs
 * \param inputs The measurement inputs
 * \param results The measurement results
 * \param skip_first_n_feature_extraction Skip feature extraction for the first n measurement pairs
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param features The returned feature vector. The innermost vector contains the
 * feature vectors for all BufferStoreNode statements
 * \param normalized_throughputs The normalized throughputs for all states
 * \param task_ids The task ids for all states
 */
void GetPerStoreFeaturesFromMeasurePairs(const Array<MeasureInput>& inputs,
                                         const Array<MeasureResult>& results,
                                         int skip_first_n_feature_extraction, int max_n_bufs,
                                         std::vector<std::vector<float>>* features,
                                         std::vector<float>* normalized_throughputs,
                                         std::vector<int>* task_ids);

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_FEATURE_H_
