/*!
 *  Copyright (c) 2016 by Contributors
 * \file pass.h
 * \brief Pass that can be applied to a graph.
 */
#ifndef NNVM_PASS_H_
#define NNVM_PASS_H_

#include <vector>
#include <functional>
#include "./base.h"
#include "./graph.h"

namespace nnvm {

/*!
 * \brief A PassFunction is a basic "Operator on Graph"
 *  It takes a source graph
 *
 *  A pass function can either change the graph structure of g,
 *  generating a new Graph, or add new attributes to the graph.
 *
 * \param src The graph to be transformed.
 * \return The generated graph.
 */
using PassFunction = std::function<Graph (Graph src)>;

/*!
 * \brief Apply a series of pass transformations on g.
 * \param src The graph to be transformed.
 * \param pass The name of pass to be applied.
 * \return The transformed graph
 */
Graph ApplyPass(Graph src,
                const std::vector<std::string>& pass);

/*!
 * \brief Registry entry for DataIterator factory functions.
 */
struct PassFunctionReg
    : public dmlc::FunctionRegEntryBase<PassFunctionReg,
                                        PassFunction> {
  /*!
   * \brief Whether the pass will change graph structure
   *  If this is false, the pass will only change attributes.
   */
  bool change_graph{false};
  /*! \brief dependencies on operator attributes */
  std::vector<std::string> op_attr_dependency;
  /*! \brief dependencies on attributes in the graph */
  std::vector<std::string> graph_attr_dependency;
  /*! \brief generated targets of graph attributes */
  std::vector<std::string> graph_attr_targets;
  /*!
   * \brief set whether this pass will change graph structure.
   * \param v the value to set
   * \return reference to self.
   */
  PassFunctionReg& set_change_graph(bool v) {  // NOLINT(*)
    change_graph = v;
    return *this;
  }
  /*!
   * \brief Declare this pass require operator attribute attr_name to be available.
   * \param attr_name Name of the attribute.
   * \return reference to self.
   */
  PassFunctionReg& provide_graph_attr(const std::string& attr_name) {  // NOLINT(*)
    graph_attr_targets.push_back(attr_name);
    return *this;
  }
  /*!
   * \brief declare this pass require operator attribute attr_name to be available.
   * \param attr_name Name of the attribute.
   * \return reference to self.
   */
  PassFunctionReg& depend_op_attr(const std::string& attr_name) {  // NOLINT(*)
    op_attr_dependency.push_back(attr_name);
    return *this;
  }
  /*!
   * \brief declare this pass require graph attribute attr_name to be available.
   * \param attr_name Name of the attribute.
   * \return reference to self.
   */
  PassFunctionReg& depend_graph_attr(const std::string& attr_name) {  // NOLINT(*)
    graph_attr_dependency.push_back(attr_name);
    return *this;
  }
};

/*!
 * \def NNVM_REGISTER_PASS
 * \brief Macro to register pass fuctions.
 *
 * \code
 * // example of registering a shape inference pass
 * NNVM_REGISTER_PASS(InferShape)
 * .describe("Shape Inference function, generate graph attributes")
 * .provide_graph_attr("data_shape")
 * .depend_graph_attr("indexed_graph")
 * .depend_op_attr("infer_shape")
 * .set_body([](const Graph& g) {
 *     // shape inference logic
 *   });
 * \endcode
 */
#define NNVM_REGISTER_PASS(name)                                     \
  DMLC_REGISTRY_REGISTER(::nnvm::PassFunctionReg, PassFunctionReg, name)

}  // namespace nnvm

#endif  // NNVM_PASS_H_
