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
 * \file ansor/feature.h
 * \brief Feature extraction for the cost model
 */

#ifndef TVM_ANSOR_FEATURE_H_
#define TVM_ANSOR_FEATURE_H_

#include <string>
#include <vector>
#include "compute_dag.h"
#include "measure.h"

namespace tvm {
namespace ansor {

/*! \brief Get PerStmt feature from a tvm IR stmt */
void GetPerStmtFeature(const Stmt& stmt,
                       int cache_line_size,
                       int max_n_bufs,
                       std::vector<float>* ret);

/* \brief Get the name of every element in the feature vector. Use this for debug and inspection */
void GetPerStmtFeatureName(int max_n_bufs, std::vector<std::string> *ret);


/*! \brief Get PerStmt feature from states and the same task */
void GetPerStmtFeaturesFromStates(const Array<State>& states,
                                  const SearchTask& task,
                                  int skip_first_n_feature_extraction,
                                  int max_n_bufs,
                                  std::vector<std::vector<float> >* features);

/*! \brief Get PerStmt feature from states and different tasks */
void GetPerStmtFeaturesFromStates(const Array<State>& states,
                                  const std::vector<SearchTask>& tasks,
                                  int skip_first_n_feature_extraction,
                                  int max_n_bufs,
                                  std::vector<std::vector<float> >* features);

/*! \brief Get PerStmt feature from a log file */
void GetPerStmtFeaturesFromFile(const std::string& filename,
                                int n_lines,
                                int max_n_bufs,
                                std::vector<std::vector<float> >* features,
                                std::vector<float>* normalized_throughputs,
                                std::vector<int>* task_ids);

/*! \brief Get PerStmt feature from measure pairs */
void GetPerStmtFeaturesFromMeasurePairs(const Array<MeasureInput>& inputs,
                                        const Array<MeasureResult>& results,
                                        int skip_first_n_feature_extraction,
                                        int max_n_bufs,
                                        std::vector<std::vector<float> >* features,
                                        std::vector<float>* normalized_throughputs,
                                        std::vector<int>* task_ids);

}   // namespace ansor
}   // namespace tvm

#endif  // TVM_ANSOR_FEATURE_H_
