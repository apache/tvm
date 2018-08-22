/*!
 *  Copyright (c) 2018 by Contributors
 * \file elemwise.cc
 * \brief Elementwise operators.
 */
#include <tvm/relay/op.h>

namespace tvm {
namespace relay {

RELAY_REGISTER_OP("log")
.describe(R"code(Returns the log input array, computed element-wise.

.. math::
   log(x)

)code" TVM_ADD_FILELINE)
.set_support_level(1)
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.");

}  // namespace relay
}  // namespace tvm
