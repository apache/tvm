/*!
 *  Copyright (c) 2016 by Contributors
 * \file op_attr_types.h
 * \brief Data structures that can appear in operator attributes.
 */
#ifndef NNVM_OP_ATTR_TYPES_H_
#define NNVM_OP_ATTR_TYPES_H_

#include <vector>
#include <string>
#include <utility>
#include <functional>
#include "./base.h"
#include "./node.h"
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
 * \brief Return number of visible outputs by the user.
 *
 * \param attrs The attributes of the node.
 *
 * \note Register under "FNumVisibleOutputs", default not registered.
 *  This can be used to hide certain output from the user,
 *  but the additional outputs can be used to pass information from
 *  forward to gradient pass.
 */
using FNumVisibleOutputs = std::function<uint32_t (const NodeAttrs& attrs)>;

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
 * \return list of input indices it mutates.
 *
 * \note Register under "FMutateInputs", default return false
 * FMutateInputs enables mutation order handling correctly.
 */
using FMutateInputs = std::function<std::vector<uint32_t> (const NodeAttrs& attrs)>;

/*!
 * \brief Inference function of certain type.
 * \tparam AttrType The type of the attribute to be infered.
 * \return whether all attributes are inferred.
 */
template<typename AttrType>
using FInferNodeEntryAttr = std::function<bool (const NodeAttrs& attrs,
                                                std::vector<AttrType> *in_attrs,
                                                std::vector<AttrType> *out_attrs)>;
/*!
 * \brief Shape inference function.
 *  Update the shapes given the input shape information.
 *  TShape.ndim() == 0 means the shape is still unknown.
 *
 * \note Register under "FInferShape",
 *  by default do not update any shapes.
 *
 *  FInferShape is needed by shape inference
 */
using FInferShape = FInferNodeEntryAttr<TShape>;

/*!
 * \brief Type inference function.
 *  Update the type given the known type information.
 *
 * \note Register under "FInferType",
 *  by default set all the output types to 0.
 */
using FInferType = FInferNodeEntryAttr<int>;

/*!
 * \brief Whether this op is an explicit backward operator,
 * If TIsBackward is true:
 *   - The first control_deps of the node points to the corresponding forward operator.
 *
 * \note Register under "TIsBackward"
 * This enables easier shape/type inference for backward operators.
 */
using TIsBackward = bool;

/*!
 * \brief Get possible inplace options.
 *  This function enables optimization to reuse memory of inputs in output.
 * \param attrs The attributes of the node
 * \return list of pair of that maps input->output,
 *   indicating possible in place operations.
 *
 * \note Register under "FInplaceOption", by default no inplace can happen.
 */
using FInplaceOption = std::function<
  std::vector<std::pair<int, int> > (const NodeAttrs& attrs)>;

/*!
 * \brief Get if the inplace option is an identity
 *  This function enables inplace optimization even when input reference count
 *  is greater than one.
 * \param attrs The attributes of the node
 * \return list of bool indicating whether corresponding pair from FInplaceOption
 *         is an identity
 *
 * \note Register under "FInplaceIdentity", by default no identities.
 */
using FInplaceIdentity = std::function<std::vector<bool> (const NodeAttrs& attrs)>;

/*!
 * \brief Get list of inputs in the op whose content are actually not used by the operator
 *  These are dummy input that can be used for example in zeros_like, ones_like.
 *
 * \param attrs The attributes of the node
 * \return list input index that are not used by the operator.
 *
 * \note Register under "FIgnoreInputs".
 */
using FIgnoreInputs = std::function<
  std::vector<uint32_t> (const NodeAttrs& attrs)>;

/*!
 * \brief Get the gradient node of the op node
 *  This function generates the backward graph of the node
 * \param nodeptr The node to take gradient
 * \param out_grads Gradient of current node's outputs
 * \return gradients of the inputs
 *
 * \note Register under "FGradient"
 */
using FGradient = std::function<std::vector<NodeEntry>(
    const NodePtr& nodeptr,
    const std::vector<NodeEntry>& out_grads)>;

/*!
 * \brief Set the attributes of input variable.
 *  Usually used for setting initialization or weight decay.
 *  \param attrs The attributes of this node.
 *  \param var the input variable
 *  \param index index of var in all inputs
 */
using FSetInputVarAttrOnCompose = std::function<void(
    const NodeAttrs& attrs,
    NodePtr var,
    const int index)>;

}  // namespace nnvm

#endif  // NNVM_OP_ATTR_TYPES_H_
