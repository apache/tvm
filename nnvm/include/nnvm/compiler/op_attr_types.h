/*!
 *  Copyright (c) 2017 by Contributors
 * \file op_attr_types.h
 * \brief The Expr and related elements in DataFlow construction.
 */
#ifndef NNVM_COMPILER_OP_ATTR_TYPES_H_
#define NNVM_COMPILER_OP_ATTR_TYPES_H_

#include <tvm/expr.h>
#include <tvm/tensor.h>
#include <tvm/schedule.h>
#include <tvm/packed_func_ext.h>
#include <tvm/runtime/registry.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/graph.h>
#include <vector>
#include <string>

namespace nnvm {
namespace compiler {

using ::tvm::Array;
using ::tvm::Tensor;
using ::tvm::Schedule;

/*! \brief operator pattern used in graph fusion */
enum OpPatternKind : int {
  // Elementwise operation
  kElemWise = 0,
  // Broadcast operation
  kBroadcast = 1,
  // Complex operation, can fuse bcast in input/outputs
  // but cannot chain another complex op
  kComplex = 2,
  // Extern operation, cannot fuse anything.
  kExtern = 3
};

/*! \brief the operator pattern */
using TOpPattern = int;

/*!
 * \brief Computation description interface
 * \param attrs The attribute of the node.
 * \param inputs The input tensors(placeholders)
 * \param out_info Tensors holding shape/type information about output,
 &                 these are always placeholders.
 * \return The output description of the tensor.
 */
using FTVMCompute = std::function<
  Array<Tensor>(const NodeAttrs& attrs,
                const Array<Tensor>& inputs,
                const Array<Tensor>& out_info)>;

/*!
 * \brief Build the computation schedule for
 *  op whose root is at current op.
 * \param attrs The attribute of the node.
 * \param outs The output tensors.
 * \param target The build target.
 * \return schedule The computation schedule.
 */
using FTVMSchedule = std::function<
  Schedule(const NodeAttrs& attrs,
           const Array<Tensor>& outs,
           const std::string& target)>;

/*! \brief Layout Information about an entry */
using TLayoutInfo = std::string;

/*!
 * \brief The producer consumer function of node layout
 * \param attrs The attribute of the node.
 * \param ilayouts The input layouts that the node request.
 * \param olayouts The output layouts that the node produce.
 * \return bool The success flag.
 */
using FTVMLayoutRequest = std::function<bool (const NodeAttrs& attrs,
                                              std::vector<TLayoutInfo> *ilayouts,
                                              std::vector<TLayoutInfo> *olayouts)>;

/*!
 * \brief Transform from normal operator to vectorized operator
 * \param node The source node.
 * \return Transformed vectorized op.
 */
using FTVMVectorizedOp = std::function<nnvm::NodePtr (const nnvm::Node* node)>;

}  // namespace compiler
}  // namespace nnvm
#endif  // NNVM_COMPILER_OP_ATTR_TYPES_H_
