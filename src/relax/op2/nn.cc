#include "../op/arg2relax.h"
#include "../op/op_common.h"
#include "../op/relax2te.h"

namespace tvm {
namespace relax {
namespace {
#undef TVM_RELAX_REGISTER_OP
#undef TVM_REGISTER_GLOBAL
#define TVM_REGISTER_GLOBAL(OpName)                   \
  TVM_STR_CONCAT(TVM_FUNC_REG_VAR_DEF, __COUNTER__) = \
      ::tvm::runtime::Registry::Register("__" OpName)
#define TVM_RELAX_REGISTER_OP(OpName) TVM_REGISTER_OP("__" OpName)

// (TVM-TOOL) cc_op begin decl/nn/*
/*!
 * TBD
 * \param data TODO(tvm-unity-team): add doc
 * \param output_size TODO(tvm-unity-team): add doc
 * \param layout TODO(tvm-unity-team): add doc
 * \param out_layout TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call adaptive_max_pool2d(relax::Expr data, Optional<Array<IntImm>> output_size,
                                String layout, String out_layout);
/*!
 * TBD
 * \param query TODO(tvm-unity-team): add doc
 * \param key TODO(tvm-unity-team): add doc
 * \param value TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call attention(relax::Expr query, relax::Expr key, relax::Expr value);
/*!
 * TBD
 * \param data TODO(tvm-unity-team): add doc
 * \param pool_size TODO(tvm-unity-team): add doc
 * \param strides TODO(tvm-unity-team): add doc
 * \param padding TODO(tvm-unity-team): add doc
 * \param dilation TODO(tvm-unity-team): add doc
 * \param ceil_mode TODO(tvm-unity-team): add doc
 * \param layout TODO(tvm-unity-team): add doc
 * \param out_layout TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call avg_pool2d(relax::Expr data, Array<IntImm> pool_size, Array<IntImm> strides,
                       Array<IntImm> padding, Array<IntImm> dilation, bool ceil_mode, String layout,
                       String out_layout);
/*!
 * TBD
 * \param data TODO(tvm-unity-team): add doc
 * \param gamma TODO(tvm-unity-team): add doc
 * \param beta TODO(tvm-unity-team): add doc
 * \param moving_mean TODO(tvm-unity-team): add doc
 * \param moving_var TODO(tvm-unity-team): add doc
 * \param axes TODO(tvm-unity-team): add doc
 * \param epsilon TODO(tvm-unity-team): add doc
 * \param center TODO(tvm-unity-team): add doc
 * \param scale TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call batch_norm(relax::Expr data, relax::Expr gamma, relax::Expr beta,
                       relax::Expr moving_mean, relax::Expr moving_var, Array<IntImm> axes,
                       double epsilon, bool center, bool scale);
/*!
 * TBD
 * \param data TODO(tvm-unity-team): add doc
 * \param weight TODO(tvm-unity-team): add doc
 * \param strides TODO(tvm-unity-team): add doc
 * \param padding TODO(tvm-unity-team): add doc
 * \param dilation TODO(tvm-unity-team): add doc
 * \param groups TODO(tvm-unity-team): add doc
 * \param data_layout TODO(tvm-unity-team): add doc
 * \param kernel_layout TODO(tvm-unity-team): add doc
 * \param out_layout TODO(tvm-unity-team): add doc
 * \param out_dtype TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call conv2d(relax::Expr data, relax::Expr weight, Array<IntImm> strides,
                   Array<IntImm> padding, Array<IntImm> dilation, int64_t groups,
                   String data_layout, String kernel_layout, String out_layout,
                   runtime::DataType out_dtype);
/*!
 * TBD
 * \param data TODO(tvm-unity-team): add doc
 * \param weight TODO(tvm-unity-team): add doc
 * \param strides TODO(tvm-unity-team): add doc
 * \param padding TODO(tvm-unity-team): add doc
 * \param dilation TODO(tvm-unity-team): add doc
 * \param groups TODO(tvm-unity-team): add doc
 * \param data_layout TODO(tvm-unity-team): add doc
 * \param kernel_layout TODO(tvm-unity-team): add doc
 * \param out_layout TODO(tvm-unity-team): add doc
 * \param out_dtype TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call conv2d_transpose(relax::Expr data, relax::Expr weight, Array<IntImm> strides,
                             Array<IntImm> padding, Array<IntImm> dilation, int64_t groups,
                             String data_layout, String kernel_layout, String out_layout,
                             runtime::DataType out_dtype);
/*!
 * TBD
 * \param predictions TODO(tvm-unity-team): add doc
 * \param labels TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call cross_entropy_with_logits(relax::Expr predictions, relax::Expr labels);
/*!
 * TBD
 * \param data TODO(tvm-unity-team): add doc
 * \param rate TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call dropout(relax::Expr data, double rate);
/*!
 * TBD
 * \param data TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call gelu(relax::Expr data);
/*!
 * TBD
 * \param data TODO(tvm-unity-team): add doc
 * \param gamma TODO(tvm-unity-team): add doc
 * \param beta TODO(tvm-unity-team): add doc
 * \param num_groups TODO(tvm-unity-team): add doc
 * \param channel_axis TODO(tvm-unity-team): add doc
 * \param epsilon TODO(tvm-unity-team): add doc
 * \param center TODO(tvm-unity-team): add doc
 * \param scale TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call group_norm(relax::Expr data, relax::Expr gamma, relax::Expr beta, int64_t num_groups,
                       int64_t channel_axis, double epsilon, bool center, bool scale);
/*!
 * TBD
 * \param data TODO(tvm-unity-team): add doc
 * \param gamma TODO(tvm-unity-team): add doc
 * \param beta TODO(tvm-unity-team): add doc
 * \param axes TODO(tvm-unity-team): add doc
 * \param epsilon TODO(tvm-unity-team): add doc
 * \param center TODO(tvm-unity-team): add doc
 * \param scale TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call layer_norm(relax::Expr data, relax::Expr gamma, relax::Expr beta, Array<IntImm> axes,
                       double epsilon, bool center, bool scale);
/*!
 * TBD
 * \param data TODO(tvm-unity-team): add doc
 * \param axis TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call log_softmax(relax::Expr data, int64_t axis);
/*!
 * TBD
 * \param data TODO(tvm-unity-team): add doc
 * \param pool_size TODO(tvm-unity-team): add doc
 * \param strides TODO(tvm-unity-team): add doc
 * \param padding TODO(tvm-unity-team): add doc
 * \param dilation TODO(tvm-unity-team): add doc
 * \param ceil_mode TODO(tvm-unity-team): add doc
 * \param layout TODO(tvm-unity-team): add doc
 * \param out_layout TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call max_pool2d(relax::Expr data, Array<IntImm> pool_size, Array<IntImm> strides,
                       Array<IntImm> padding, Array<IntImm> dilation, bool ceil_mode, String layout,
                       String out_layout);
/*!
 * TBD
 * \param data TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call relu(relax::Expr data);
/*!
 * TBD
 * \param data TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call silu(relax::Expr data);
/*!
 * TBD
 * \param data TODO(tvm-unity-team): add doc
 * \param axis TODO(tvm-unity-team): add doc
 * \return TODO(tvm-unity-team): add doc
 */
relax::Call softmax(relax::Expr data, int64_t axis);
// (TVM-TOOL) cc_op end decl/nn/*

// (TVM-TOOL) cc_op begin def/nn/*
relax::Call adaptive_max_pool2d(relax::Expr data, Optional<Array<IntImm>> output_size,
                                String layout, String out_layout) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.nn.adaptive_max_pool2d");
  Array<relax::Expr> _args;
  _args.reserve(4);
  TVM_RELAX_OP_ARG_CHECK(data, data, _args);
  TVM_RELAX_OP_ARG_CHECK(AttrExpr(output_size), output_size, _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(layout), layout, _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(out_layout), out_layout, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.adaptive_max_pool2d").set_body_typed(adaptive_max_pool2d);
TVM_RELAX_REGISTER_OP("nn.adaptive_max_pool2d");
relax::Call attention(relax::Expr query, relax::Expr key, relax::Expr value) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.nn.attention");
  Array<relax::Expr> _args;
  _args.reserve(3);
  TVM_RELAX_OP_ARG_CHECK(query, query, _args);
  TVM_RELAX_OP_ARG_CHECK(key, key, _args);
  TVM_RELAX_OP_ARG_CHECK(value, value, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.attention").set_body_typed(attention);
TVM_RELAX_REGISTER_OP("nn.attention");
relax::Call avg_pool2d(relax::Expr data, Array<IntImm> pool_size, Array<IntImm> strides,
                       Array<IntImm> padding, Array<IntImm> dilation, bool ceil_mode, String layout,
                       String out_layout) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.nn.avg_pool2d");
  Array<relax::Expr> _args;
  _args.reserve(8);
  TVM_RELAX_OP_ARG_CHECK(data, data, _args);
  TVM_RELAX_OP_ARG_CHECK(ArrayToOpaque(ScalarToPrimValue(false), {2})(pool_size), pool_size, _args);
  TVM_RELAX_OP_ARG_CHECK(ArrayToOpaque(ScalarToPrimValue(false), {2})(strides), strides, _args);
  TVM_RELAX_OP_ARG_CHECK(ArrayToOpaque(ScalarToPrimValue(false), {2})(padding), padding, _args);
  TVM_RELAX_OP_ARG_CHECK(ArrayToOpaque(ScalarToPrimValue(false), {2})(dilation), dilation, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(ceil_mode), ceil_mode, _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(layout), layout, _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(out_layout), out_layout, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.avg_pool2d").set_body_typed(avg_pool2d);
TVM_RELAX_REGISTER_OP("nn.avg_pool2d");
relax::Call batch_norm(relax::Expr data, relax::Expr gamma, relax::Expr beta,
                       relax::Expr moving_mean, relax::Expr moving_var, Array<IntImm> axes,
                       double epsilon, bool center, bool scale) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.nn.batch_norm");
  Array<relax::Expr> _args;
  _args.reserve(9);
  TVM_RELAX_OP_ARG_CHECK(data, data, _args);
  TVM_RELAX_OP_ARG_CHECK(gamma, gamma, _args);
  TVM_RELAX_OP_ARG_CHECK(beta, beta, _args);
  TVM_RELAX_OP_ARG_CHECK(moving_mean, moving_mean, _args);
  TVM_RELAX_OP_ARG_CHECK(moving_var, moving_var, _args);
  TVM_RELAX_OP_ARG_CHECK(Axes()(axes), axes, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(false)(epsilon), epsilon, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(center), center, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(scale), scale, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.batch_norm").set_body_typed(batch_norm);
TVM_RELAX_REGISTER_OP("nn.batch_norm");
relax::Call conv2d(relax::Expr data, relax::Expr weight, Array<IntImm> strides,
                   Array<IntImm> padding, Array<IntImm> dilation, int64_t groups,
                   String data_layout, String kernel_layout, String out_layout,
                   runtime::DataType out_dtype) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.nn.conv2d");
  Array<relax::Expr> _args;
  _args.reserve(10);
  TVM_RELAX_OP_ARG_CHECK(data, data, _args);
  TVM_RELAX_OP_ARG_CHECK(weight, weight, _args);
  TVM_RELAX_OP_ARG_CHECK(ArrayToOpaque(ScalarToPrimValue(false), {2})(strides), strides, _args);
  TVM_RELAX_OP_ARG_CHECK(ArrayToOpaque(ScalarToPrimValue(false), {2})(padding), padding, _args);
  TVM_RELAX_OP_ARG_CHECK(ArrayToOpaque(ScalarToPrimValue(false), {2})(dilation), dilation, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(false)(groups), groups, _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(data_layout), data_layout, _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(kernel_layout), kernel_layout, _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(out_layout), out_layout, _args);
  TVM_RELAX_OP_ARG_CHECK(DType()(out_dtype), out_dtype, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.conv2d").set_body_typed(conv2d);
TVM_RELAX_REGISTER_OP("nn.conv2d");
relax::Call conv2d_transpose(relax::Expr data, relax::Expr weight, Array<IntImm> strides,
                             Array<IntImm> padding, Array<IntImm> dilation, int64_t groups,
                             String data_layout, String kernel_layout, String out_layout,
                             runtime::DataType out_dtype) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.nn.conv2d_transpose");
  Array<relax::Expr> _args;
  _args.reserve(10);
  TVM_RELAX_OP_ARG_CHECK(data, data, _args);
  TVM_RELAX_OP_ARG_CHECK(weight, weight, _args);
  TVM_RELAX_OP_ARG_CHECK(ArrayToOpaque(ScalarToPrimValue(false), {2})(strides), strides, _args);
  TVM_RELAX_OP_ARG_CHECK(ArrayToOpaque(ScalarToPrimValue(false), {2})(padding), padding, _args);
  TVM_RELAX_OP_ARG_CHECK(ArrayToOpaque(ScalarToPrimValue(false), {2})(dilation), dilation, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(false)(groups), groups, _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(data_layout), data_layout, _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(kernel_layout), kernel_layout, _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(out_layout), out_layout, _args);
  TVM_RELAX_OP_ARG_CHECK(DType()(out_dtype), out_dtype, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.conv2d_transpose").set_body_typed(conv2d_transpose);
TVM_RELAX_REGISTER_OP("nn.conv2d_transpose");
relax::Call cross_entropy_with_logits(relax::Expr predictions, relax::Expr labels) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.nn.cross_entropy_with_logits");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(predictions, predictions, _args);
  TVM_RELAX_OP_ARG_CHECK(labels, labels, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.cross_entropy_with_logits")
    .set_body_typed(cross_entropy_with_logits);
TVM_RELAX_REGISTER_OP("nn.cross_entropy_with_logits");
relax::Call dropout(relax::Expr data, double rate) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.nn.dropout");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(data, data, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(false)(rate), rate, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.dropout").set_body_typed(dropout);
TVM_RELAX_REGISTER_OP("nn.dropout");
relax::Call gelu(relax::Expr data) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.nn.gelu");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(data, data, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.gelu").set_body_typed(gelu);
TVM_RELAX_REGISTER_OP("nn.gelu");
relax::Call group_norm(relax::Expr data, relax::Expr gamma, relax::Expr beta, int64_t num_groups,
                       int64_t channel_axis, double epsilon, bool center, bool scale) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.nn.group_norm");
  Array<relax::Expr> _args;
  _args.reserve(8);
  TVM_RELAX_OP_ARG_CHECK(data, data, _args);
  TVM_RELAX_OP_ARG_CHECK(gamma, gamma, _args);
  TVM_RELAX_OP_ARG_CHECK(beta, beta, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(false)(num_groups), num_groups, _args);
  TVM_RELAX_OP_ARG_CHECK(Axis()(channel_axis), channel_axis, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(false)(epsilon), epsilon, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(center), center, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(scale), scale, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.group_norm").set_body_typed(group_norm);
TVM_RELAX_REGISTER_OP("nn.group_norm");
relax::Call layer_norm(relax::Expr data, relax::Expr gamma, relax::Expr beta, Array<IntImm> axes,
                       double epsilon, bool center, bool scale) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.nn.layer_norm");
  Array<relax::Expr> _args;
  _args.reserve(7);
  TVM_RELAX_OP_ARG_CHECK(data, data, _args);
  TVM_RELAX_OP_ARG_CHECK(gamma, gamma, _args);
  TVM_RELAX_OP_ARG_CHECK(beta, beta, _args);
  TVM_RELAX_OP_ARG_CHECK(Axes()(axes), axes, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(false)(epsilon), epsilon, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(center), center, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(scale), scale, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.layer_norm").set_body_typed(layer_norm);
TVM_RELAX_REGISTER_OP("nn.layer_norm");
relax::Call log_softmax(relax::Expr data, int64_t axis) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.nn.log_softmax");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(data, data, _args);
  TVM_RELAX_OP_ARG_CHECK(Axis()(axis), axis, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.log_softmax").set_body_typed(log_softmax);
TVM_RELAX_REGISTER_OP("nn.log_softmax");
relax::Call max_pool2d(relax::Expr data, Array<IntImm> pool_size, Array<IntImm> strides,
                       Array<IntImm> padding, Array<IntImm> dilation, bool ceil_mode, String layout,
                       String out_layout) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.nn.max_pool2d");
  Array<relax::Expr> _args;
  _args.reserve(8);
  TVM_RELAX_OP_ARG_CHECK(data, data, _args);
  TVM_RELAX_OP_ARG_CHECK(ArrayToOpaque(ScalarToPrimValue(false), {2})(pool_size), pool_size, _args);
  TVM_RELAX_OP_ARG_CHECK(ArrayToOpaque(ScalarToPrimValue(false), {2})(strides), strides, _args);
  TVM_RELAX_OP_ARG_CHECK(ArrayToOpaque(ScalarToPrimValue(false), {2})(padding), padding, _args);
  TVM_RELAX_OP_ARG_CHECK(ArrayToOpaque(ScalarToPrimValue(false), {2})(dilation), dilation, _args);
  TVM_RELAX_OP_ARG_CHECK(ScalarToPrimValue(true)(ceil_mode), ceil_mode, _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(layout), layout, _args);
  TVM_RELAX_OP_ARG_CHECK(Str()(out_layout), out_layout, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.max_pool2d").set_body_typed(max_pool2d);
TVM_RELAX_REGISTER_OP("nn.max_pool2d");
relax::Call relu(relax::Expr data) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.nn.relu");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(data, data, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.relu").set_body_typed(relu);
TVM_RELAX_REGISTER_OP("nn.relu");
relax::Call silu(relax::Expr data) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.nn.silu");
  Array<relax::Expr> _args;
  _args.reserve(1);
  TVM_RELAX_OP_ARG_CHECK(data, data, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.silu").set_body_typed(silu);
TVM_RELAX_REGISTER_OP("nn.silu");
relax::Call softmax(relax::Expr data, int64_t axis) {
  using namespace relax::arg2relax;
  static const Op& op = Op::Get("relax.nn.softmax");
  Array<relax::Expr> _args;
  _args.reserve(2);
  TVM_RELAX_OP_ARG_CHECK(data, data, _args);
  TVM_RELAX_OP_ARG_CHECK(Axis()(axis), axis, _args);
  return Call(op, _args, Attrs(nullptr), {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.softmax").set_body_typed(softmax);
TVM_RELAX_REGISTER_OP("nn.softmax");
// (TVM-TOOL) cc_op end def/nn/*

}  // namespace
}  // namespace relax
}  // namespace tvm
