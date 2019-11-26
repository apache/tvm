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

#ifndef NNVM_C_API_SUBGRAPH_H_
#define NNVM_C_API_SUBGRAPH_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <nnvm/c_api.h>

/*!
 * \brief This API partitions a graph only by the operator names
 * provided by users. This will attach a DefaultSubgraphProperty
 * to the input graph for partitioning. This function should be
 * used only for the testing purpose.
 */
NNVM_DLL int NNPartitionGraph(GraphHandle graph_handle,
                              const char* prop_name,
                              const nn_uint num_ops,
                              const char** op_names,
                              GraphHandle* ret_graph_handle);

/*!
 * \brief Given a subgraph property name, use the provided op names
 * as the op_names attribute for that subgraph property, instead of
 * the predefined one. This is only for the purpose of testing.
 */
NNVM_DLL int NNSetSubgraphPropertyOpNames(const char* prop_name,
                                          const nn_uint num_ops,
                                          const char** op_names);

/*!
 * \brief Given a subgraph property name, delete the op name set
 * in the SubgraphPropertyOpNameSet.
 */
NNVM_DLL int NNRemoveSubgraphPropertyOpNames(const char* prop_name);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // NNVM_C_API_SUBGRAPH_H_
