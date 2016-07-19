/*!
 *  Copyright (c) 2016 by Contributors
 * \file op_attr_types.h
 * \brief Data structures that can appear in operator attributes.
 */
#ifndef NNVM_OP_ATTR_TYPES_H_
#define NNVM_OP_ATTR_TYPES_H_

#include <vector>
#include <string>
#include <functional>
#include "./base.h"
#include "./tuple.h"

namespace nnvm {

// These types are optional attributes in each operator.
// Each attribute can be required by some passes.

/*!
 * \brief Return list of input arguments names of each operator.
 *
 * \param attrs The attributes of the node.
 * \return list of inputs
 * \note Register under "FListInputNames", default return {"data"}.
 *
 *  FListInputNames enables automatic variable creation for missing arguments.
 */
using FListInputNames = std::function<std::vector<std::string> (const NodeAttrs& attrs)>;

/*!
 * \brief Return list of output arguments names of each operator.
 *
 * \param attrs The attributes of the node.
 * \return list of inputs
 * \note Register under "FListOutputNames", default return {"outputs"}.
 *
 *  FListOutputNames customized naming for operator outputs.
 */
using FListOutputNames = std::function<std::vector<std::string> (const NodeAttrs& attrs)>;

/*!
 * \brief Check whether operator will mutate k-th input.
 * \param attrs The attributes of the node.
 * \param index The input index
 * \return Whether this operator will mutate index-th input.
 *
 * \note Register under "FMutateInput", default return false
 * FMutateInputs enables mutation order handling correctly.
 */
using FMutateInput = std::function<bool (const NodeAttrs& attrs, uint32_t index)>;

/*!
 * \brief Shape inference function.
 *  Update the shapes given the input shape information.
 *  TShape.ndim() == 0 means the shape is still unknown.
 *
 * \param attrs The attributes of the node.
 * \param in_shapes Array of shapes from the inputs.
 * \param out_shapes Array of shapes from the outputs.
 *
 * \return Whether all the shapes are known.
 *
 * \note Register under "FInferShape",
 *  by default do not update any shapes.
 *
 *  FInferShape is needed by shape inference
 */
using FInferShape = std::function<bool (const NodeAttrs& attrs,
                                        array_view<TShape*> in_shapes,
                                        array_view<TShape*> out_shapes)>;

}  // namespace nnvm

#endif  // NNVM_OP_ATTR_TYPES_H_
