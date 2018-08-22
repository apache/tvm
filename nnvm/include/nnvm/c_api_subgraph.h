/*!
 *  Copyright (c) 2018 by Contributors
 * \file c_api_subgraph.h
 * \brief C API of nnvm for ease of testing backend in Python
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
