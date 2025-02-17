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
 * \file graph.h
 * \brief Utilities to get information about schedule graph.
 */
#ifndef TVM_TE_OPERATION_GRAPH_H_
#define TVM_TE_OPERATION_GRAPH_H_

#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>

namespace tvm {
namespace te {

/*!
 * \brief data structure of Operation->Tensors it reads
 */
using ReadGraph = Map<Operation, Array<Tensor>>;

/*!
 * \brief Get read graph of each operation to all the
 *  Tensors that it directly depends on.
 *
 *  The result map contains Operations needed to finish root Operation.
 * \param roots The root operation.
 * \return The result map.
 */
ReadGraph CreateReadGraph(const Array<Operation>& roots);

/*!
 * \brief Get a post DFS ordered of operations in the graph.
 * \param roots The root of the graph.
 * \param g The read graph.
 * \return vector order of Operations in PostDFS order.
 *
 * \note PostDFSOrder is a special case of Topoligical order,
 *   and can be used when topoligical order is needed.
 */
Array<Operation> PostDFSOrder(const Array<Operation>& roots, const ReadGraph& g);

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_OPERATION_GRAPH_H_
