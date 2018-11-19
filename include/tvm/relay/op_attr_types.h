/*!
 *  Copyright (c) 2017 by Contributors
 * \file nnvm/compiler/op_attr_types.h
 * \brief The Expr and related elements in DataFlow construction.
 */
#ifndef TVM_RELAY_OP_ATTR_TYPES_H_
#define TVM_RELAY_OP_ATTR_TYPES_H_

#include <tvm/tensor.h>
#include <tvm/schedule.h>
#include <tvm/build_module.h>
#include <tvm/relay/type.h>
#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {

/*! \brief operator pattern used in graph fusion */
enum OpPatternKind {
  // Elementwise operation
  kElemWise = 0,
  // Broadcasting operator, can always map output axis to the input in order.
  // for example :code:`out[i, ax1, j, ax2] = input[i, j]`.
  // Note that the axis need to be in order so transpose is not a bcast operator.
  kBroadcast = 1,
  // Injective operator, can always injectively map output axis to a single input axis.
  // All injective operator can still be safely fused to injective and reduction.
  kInjective = 2,
  // Communicative reduction operator.
  kCommReduce = 3,
  // Complex operation, can still fuse elemwise operations into its output.
  // but cannot chain another complex op
  kOutEWiseFusable = 4,
  // Opaque operation, cannot fuse anything.
  kOpaque = 8
};

/*! \brief the operator pattern */
using TOpPattern = int;

/*!
 * \brief Whether operator is stateful or contain internal state.
 *
 * All the primitive ops we registered so far are pure.
 * This attribute is left for potential future compatible reasons.
 * We can always work around the stateful ops by adding an additional
 * handle argument and return it.
 */
using TOpIsStateful = bool;

/*!
 * \brief Computation description interface.
 *
 * \note This function have a special convention
 *  for functions with tuple input/output.
 *
 *  So far we restrict tuple support to the following case:
 *  - Function which takes a single tuple as input.
 *  - Function which outputs a single tuple.
 *
 *  In both cases, the tuple is flattened as array.
 *
 * \param attrs The attribute of the primitive
 * \param inputs The input tensors.
 * \param out_type The output type information
 &                 these are always placeholders.
 * \return The output compute description of the operator.
 */
using FTVMCompute = runtime::TypedPackedFunc<
  Array<Tensor>(const Attrs& attrs,
                const Array<Tensor>& inputs,
                const Type& out_type,
                const Target& target)>;

/*!
 * \brief Build the computation schedule for
 *  op whose root is at current op.
 *
 * \param attrs The attribute of the node.
 * \param outs The output tensors.
 * \param target The build target.
 * \return schedule The computation schedule.
 */
using FTVMSchedule = runtime::TypedPackedFunc<
  Schedule(const Attrs& attrs,
           const Array<Tensor>& outs,
           const Target& target)>;

/*!
 * \brief Alternate the layout of operators or replace the
 *  operator with other expressions.
 *
 * \param attrs The attribute of the node.
 * \param inputs The arguments of this operator.
 * \return new_expr The modified expression.
 */
using FTVMAlterOpLayout = runtime::TypedPackedFunc<
  Expr(const Attrs& attrs,
       const Array<Expr>& args)>;

/*!
 * \brief Forward rewriting rule for a specific op.
 *
 * \param ref_call The reference old call type to be rewritten.
 *                 We can make use of the op and type information.
 * \param new_args The new arguments (some of them could be TempExpr).
 * \param ctx  Optional context information about ref_call.
 * \return The rewriten result call, can also return nullptr,
 *         which indicate the rewriter should use the default fallback
 *         rule that realizes all its input and compose the call.
 *
 * \note When we register the function, we can register
 *       a different signature with ctx to be a specific node type.
 */
using FForwardRewrite = runtime::TypedPackedFunc<
  Expr(const Call& ref_call,
       const Array<Expr>& new_args,
       const NodeRef& ctx)>;
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_ATTR_TYPES_H_
