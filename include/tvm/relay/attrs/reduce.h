/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/relay/attrs/reduce.h
 * \brief Auxiliary attributes for reduce operators.
 */
#ifndef TVM_RELAY_ATTRS_REDUCE_H_
#define TVM_RELAY_ATTRS_REDUCE_H_

#include <tvm/attrs.h>
#include <string>

namespace tvm {
namespace relay {

/*! \brief Attributes for Reduce operators */
struct ReduceAttrs : public tvm::AttrsNode<ReduceAttrs> {
  Array<Integer> axis;
  bool keepdims;
  bool exclude;

  TVM_DECLARE_ATTRS(ReduceAttrs, "relay.attrs.ReduceAttrs") {
    TVM_ATTR_FIELD(axis).set_default(NullValue<Array<Integer>>())
        .describe(R"code(The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.)code");

    TVM_ATTR_FIELD(keepdims).set_default(false)
      .describe("If this is set to `True`, the reduced axes are left "
                "in the result as dimension with size one.");
    TVM_ATTR_FIELD(exclude).set_default(false)
      .describe("Whether to perform reduction on axis that are NOT in axis instead.");
  }
};
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_REDUCE_H_
