/*!
 *  Copyright (c) 2016 by Contributors
 * \file op_attr_types.h
 * \brief The Expr and related elements in DataFlow construction.
 */
#ifndef TVM_OP_ATTR_TYPES_H_
#define TVM_OP_ATTR_TYPES_H_

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

namespace tvm {
namespace contrib {

using runtime::PackedFunc;
using nnvm::StorageVector;
using nnvm::ShapeVector;
using nnvm::DTypeVector;
using nnvm::TShape;
using nnvm::NodeAttrs;

/*! \brief DLPack compatible data types */
using DLTypeVector = std::vector<DLDataType>;
/*!
 * \brief Computation description interface
 * \param attrs The attribute of the node.
 * \param inputs The input tensors(placeholders)
 * \return The output description of the tensor.
 */
using FTVMCompute = std::function<
  Array<Tensor>
  (const NodeAttrs& attrs, const Array<Tensor>& inputs)>;

/*!
 * \brief Build the computation schedule for
 *  op whose  root is at current op.
 * \param attrs The attribute of the node.
 * \param outs The output tensors.
 * \param target The build target.
 * \return schedule The computation schedule.
 */
using FTVMSchedule = std::function<
  Schedule(const NodeAttrs& attrs,
           const Array<Tensor>& outs,
           const std::string& target)>;

/*! \brief Layout Information. */
using TLayoutInfo = std::string;

/*!
 * \brief The producer consumer function of node layout
 * \param attrs The attribute of the node.
 * \param consumer The layouts that the node consume.
 * \param producer The layouts that the node produce.
 * \return bool The success flag.
 */
using FTVMLayoutProducerConsumer = std::function<bool (const NodeAttrs& attrs,
                                                       std::vector<TLayoutInfo> *consumer,
                                                       std::vector<TLayoutInfo> *producer)>;

/*! \brief The default layout. */
const TLayoutInfo& GetDefaultLayout();

/*! \brief Parameters of layout transform operator */
struct LayoutTransformParam : public dmlc::Parameter<LayoutTransformParam> {
  std::string src_layout;
  std::string dst_layout;
  DMLC_DECLARE_PARAMETER(LayoutTransformParam) {
    DMLC_DECLARE_FIELD(src_layout);
    DMLC_DECLARE_FIELD(dst_layout);
  }
};

/*! \brief Transform from normal operator to vectorized operator */
using FTVMVectorizedOp = std::function<nnvm::NodePtr (const nnvm::Node*)>;

// The storage result of op
enum OpPatternKind : int {
  // Elementwise operation
  kElemWise,
  // Broadcast operation
  kBroadcast,
  // Complex operation, can fuse bcast in input/outputs
  // but cannot chain another complex op
  kComplex,
  // Extern operation, cannot fuse anything.
  kExtern
};

using TOpPattern = int;

/*!
 * \brief Get PackedFunction from global registry and
 *  report error if it does not exist
 * \param name The name of the function.
 * \return The created PackedFunc.
 */
inline const PackedFunc& GetPackedFunc(const std::string& name) {
  const PackedFunc* pf = tvm::runtime::Registry::Get(name);
  CHECK(pf != nullptr) << "Cannot find function " << name << " in registry";
  return *pf;
}

/*!
 * \brief Create a Graph execution module by a given graph and the code module.
 * \param g The graph to be executed.
 * \param m The tvm module containing the functions.
 * \return The created executor module.
 */
tvm::runtime::Module CreateExecutor(nnvm::Graph g);
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_OP_ATTR_TYPES_H_
