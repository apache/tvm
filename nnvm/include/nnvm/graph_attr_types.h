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
 *  Copyright (c) 2016 by Contributors
 * \file nnvm/graph_attr_types.h
 * \brief Data structures that can appear in graph attributes.
 */
#ifndef NNVM_GRAPH_ATTR_TYPES_H_
#define NNVM_GRAPH_ATTR_TYPES_H_

#include <vector>
#include <string>
#include <unordered_map>
#include "tuple.h"
#include "layout.h"

namespace nnvm {

/*!
 * \brief The result holder of JSON serializer
 *
 * \note Stored under ret.attrs["json"], provided by Pass "SaveJSON"

 * \code
 *  Graph ret = ApplyPass(src_graph, "SaveJSON");
 *  const JSONString& json = ret.GetAttr<JSONString>("shape");
 * \endcode
 */
using JSONString = std::string;

/*!
 * \brief The result holder of shape of each NodeEntry in the graph.
 * \note Stored under graph.attrs["shape"], provided by Pass "InferShape"
 *
 * \code
 *  Graph g = ApplyPass(src_graph, "InferShape");
 *  const ShapeVector& shapes = g.GetAttr<ShapeVector>("shape");
 *  // get shape by entry id
 *  TShape entry_shape = shapes[g.indexed_graph().entry_id(my_entry)];
 * \endcode
 *
 * \sa FInferShape
 */
using ShapeVector = std::vector<TShape>;

/*!
 * \brief The result holder of type of each NodeEntry in the graph.
 * \note Stored under graph.attrs["dtype"], provided by Pass "InferType"
 *
 * \code
 *  Graph g = ApplyPass(src_graph, "InferType");
 *  const DTypeVector& types = g.GetAttr<DTypeVector>("dtype");
 *  // get type by entry id
 *  int entry_type = dtypes[g.indexed_graph().entry_id(my_entry)];
 * \endcode
 *
 * \sa FInferType
 */
using DTypeVector = std::vector<int>;

/*!
 * \brief The result holder of layout of each NodeEntry in the graph.
 * \note Stored under graph.attrs["layout"], provided by Pass "InferType"
 *
 * \code
 *  Graph g = ApplyPass(src_graph, "LayoutTransform");
 *  const LayoutVector& layouts = g.GetAttr<LayoutVector>("layout");
 *  // get layout by entry id
 *  int entry_layout = layouts[g.indexed_graph().entry_id(my_entry)];
 * \endcode
 *
 * \sa FCorrectLayout
 */
using LayoutVector = std::vector<Layout>;

/*!
 * \brief The result holder of device of each operator in the graph.
 * \note Stored under graph.attrs["device"], provided by Pass "PlaceDevice"
 *
 * \code
 *  Graph g = ApplyPass(src_graph, "PlaceDevice");
 *  const &device = g.GetAttr<DeviceVector>("device");
 *  // get device by node_id
 *  int device_type = device[g.indexed_graph().node_id(my_node)];
 * \endcode
 */
using DeviceVector = std::vector<int>;

/*!
 * \brief The result holder of device of each operator in the graph.
 *
 * \note Stored under graph.attrs["device_assign_map"], needed by Pass "PlaceDevice"
 * -1 means unknown device
 */
using DeviceAssignMap = std::unordered_map<std::string, int>;

/*!
 * \brief The result holder of storage id of each NodeEntry in the graph.
 *
 * \note Stored under graph.attrs["storage"], provided by Pass "PlanMemory"
 *  Storage id is a continuous integer.
 *  If the storage id is -1 then the storage is not assigned.
 *
 * \code
 *  Graph g = ApplyPass(src_graph, "PlanMemory");
 *  const &storage = g.GetAttr<StorageVector>("storage");
 *  // get storage id by entry
 *  int storage_id = storage[g.indexed_graph().entry_id(my_entry)];
 * \endcode
 */
using StorageVector = std::vector<int>;

}  // namespace nnvm

#endif  // NNVM_GRAPH_ATTR_TYPES_H_
