/*!
 *  Copyright (c) 2016 by Contributors
 * \file pass_functions.h
 * \brief Pass functions that simply redirect the calls to ApplyPass
 *
 *  This file serves as documentation on how to use functions implemented in "src/pass".
 *  It is totally optional to add these functions when you add a new pass, since
 *  ApplyPass can be directly called.
 */
#ifndef NNVM_PASS_FUNCTIONS_H_
#define NNVM_PASS_FUNCTIONS_H_

#include <string>
#include <memory>
#include "./base.h"
#include "./pass.h"
#include "./graph_attr_types.h"

namespace nnvm {
namespace pass {

/*!
 * \brief Load a graph from JSON string, redirects to "LoadJSON" pass.
 * \param json_str The json string.
 * \return Loaded graph.
 */
inline Graph LoadJSON(const std::string& json_str) {
  Graph ret;
  ret.attrs["json"] = std::make_shared<any>(json_str);
  return ApplyPass(ret, {"LoadJSON"});
}

/*!
 * \brief Save a graph to json, redirects to "SaveJSON" pass.
 * \param graph The to be saved.
 * \return The json string.
 */
inline std::string SaveJSON(Graph graph) {
  Graph ret = ApplyPass(std::move(graph), {"SaveJSON"});
  return ret.GetAttr<std::string>("json");
}

/*!
 * \brief Add control flow dependencies between nodes
 *  To correctly order mutation and read to resolve
 *  write after read problem and read after write problems.
 * \param src source graph
 * \return A graph that added control flow dependencies.
 */
inline Graph OrderMutation(Graph src) {
  return ApplyPass(std::move(src), {"OrderMutation"});
}

/*!
 * \brief Infer shapes in the graph given the information.
 * \param graph source graph
 * \param shape_args The shapes of aruguments to the graph.
 * \param shape_attr_key The key to the node attribute that can indicate shape.
 * \return A graph with new attribute "shape" containing inferred shape of each NodeEntry.
 *  The index of ShapeVector is given by graph.indexed_graph().entry_id
 */
inline Graph InferShape(Graph graph,
                        ShapeVector shape_args = {},
                        std::string shape_attr_key = "") {
  if (shape_args.size() != 0) {
    graph.attrs["shape_args"] = std::make_shared<any>(std::move(shape_args));
  }
  if (shape_attr_key.length() != 0) {
    graph.attrs["shape_attr_key"] = std::make_shared<any>(std::move(shape_attr_key));
  }
  return ApplyPass(std::move(graph), {"InferShape"});
}

/*!
 * \brief Infer types in the graph given the information.
 * \param graph source graph
 * \param shape_args The shapes of aruguments to the graph.
 * \param shape_attr_key The key to the node attribute that can indicate shape.
 * \return A graph with new attribute "shape" containing inferred shape of each NodeEntry.
 *  The index of ShapeVector is given by graph.indexed_graph().entry_id
 */
inline Graph InferType(Graph graph,
                       DTypeVector type_args = {},
                       std::string type_attr_key = "") {
  if (type_args.size() != 0) {
    graph.attrs["type_args"] = std::make_shared<any>(std::move(type_args));
  }
  if (type_attr_key.length() != 0) {
    graph.attrs["type_attr_key"] = std::make_shared<any>(std::move(type_attr_key));
  }
  return ApplyPass(std::move(graph), {"InferType"});
}

/*!
 * \brief Place the devices
 * \param graph source graph
 * \param device_group_attr_key The attribute name for hinting the device group.
 * \param device_assign_map The assignment map of device
 * \param device_copy_op The name of copy op to be inserted when cross device copy happened.
 * \return A graph with new attribute "device", cotaining device information of each node.
 */
inline Graph PlaceDevice(Graph graph,
                         std::string device_group_attr_key,
                         DeviceAssignMap device_assign_map,
                         std::string device_copy_op) {
  graph.attrs["device_group_attr_key"] = std::make_shared<any>(std::move(device_group_attr_key));
  graph.attrs["device_assign_map"] = std::make_shared<any>(std::move(device_assign_map));
  graph.attrs["device_copy_op"] = std::make_shared<any>(std::move(device_copy_op));
  return ApplyPass(std::move(graph), {"PlaceDevice"});
}

}  // namespace pass
}  // namespace nnvm
#endif  // NNVM_PASS_FUNCTIONS_H_
