/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph.h
 * \brief Data structure about computational graph.
 */
#ifndef TVM_GRAPH_HANDLE_H_
#define TVM_GRAPH_HANDLE_H_

#include <string>
#include <tvm/base.h>

namespace tvm {

/*!
 * \brief Computational graph handle.
 *  Use GraphHandle as its container type
 */
struct GraphHandleNode : public Node {
  void *graph_handle;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("graph_handle", &graph_handle);
  }

  static constexpr const char* _type_key = "GraphHandle";
  TVM_DECLARE_NODE_TYPE_INFO(GraphHandleNode, Node);
};

/*! \brief Defines graph handle */
TVM_DEFINE_NODE_REF(GraphHandle, GraphHandleNode);

}  // namespace tvm
#endif  // TVM_GRAPH_HANDLE_H_
