/*!
 *  Copyright (c) 2018 by Contributors
 * \file pad.cc
 * \brief Implementation of operator pad
 */
#include <tvm/ir_operator.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <topi/nn.h>
#include <vector>
#include "../layout.h"
#include "../op_common.h"

namespace tvm {
namespace relay {

// relay.nn.pad
TVM_REGISTER_NODE_TYPE(PadAttrs);

bool PadRel(const Array<Type>& types,
            int num_inputs,
            const Attrs& attrs,
            const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const PadAttrs* param = attrs.as<PadAttrs>();
  CHECK(param != nullptr);

  // check that pad widths match lengths
  CHECK(data->shape.size() == param->pad_width.size())
    << "There should be as many pad width pairs as shape dimensions "
    << "but the shape has " << data->shape.size() << " dimensions "
    << "and there are " << param->pad_width.size() << " pad width pairs.";

  // each pad width element should be a pair of positive integers
  std::vector<IndexExpr> oshape;
  for (size_t i = 0; i < param->pad_width.size(); i++) {
    CHECK(param->pad_width[i].size() == 2)
      << "Each pad width element should be a pair but at index " << i
      << " there are " << param->pad_width[i].size() << " elements.";

    auto width1 = as_const_int(param->pad_width[i][0]);
    auto width2 = as_const_int(param->pad_width[i][1]);
    CHECK(width1 != nullptr);
    CHECK(width2 != nullptr);

    CHECK(*width1 >= 0)
      << "Param width elements should be positive but first pad width at "
      << "index " << i << " is " << *width1 << ".";
    CHECK(*width2 >= 0)
      << "Param width elements should be positive but first pad width at "
      << "index " << i << " is " << *width2 << ".";

    auto padding = make_const(data->shape[i].type(), *width1 + *width2);
    oshape.push_back(data->shape[i] + padding);
  }

  reporter->Assign(types[1], TensorTypeNode::make(Array<IndexExpr>(oshape),
                                                  data->dtype));
  return true;
}

Array<Tensor> PadCompute(const Attrs& attrs,
                         const Array<Tensor>& inputs,
                         const Type& out_type,
                         const Target& target) {
  const auto* param = attrs.as<PadAttrs>();
  CHECK(param != nullptr);

  auto pad_width = param->pad_width;
  CHECK(pad_width.size() == inputs[0].ndim() &&
    pad_width[0].size() == 2)
    << "Illegal pad_width";
  Array<IndexExpr> pad_before;
  for (size_t i = 0; i < pad_width.size(); ++i) {
    pad_before.push_back(pad_width[i][0]);
  }
  Array<IndexExpr> pad_after;
  for (size_t i = 0; i < pad_width.size(); ++i) {
    pad_after.push_back(pad_width[i][1]);
  }
  const auto* out_ttype = out_type.as<TensorTypeNode>();
  return Array<Tensor>{ topi::pad(inputs[0], pad_before, pad_after,
                                  tvm::make_const(out_ttype->dtype, param->pad_value)) };
}

// Handler to create a call to the padding op used by front-end FFI
Expr MakePad(Expr data, Array<Array<IndexExpr> > pad_width, double pad_value) {
  auto attrs = make_node<PadAttrs>();
  attrs->pad_value = pad_value;
  attrs->pad_width = std::move(pad_width);
  static const Op& op = Op::Get("nn.pad");
  return CallNode::make(op, {data}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.pad")
.set_body([](const TVMArgs& args, TVMRetValue* rv) {
    runtime::detail::unpack_call<Expr, 3>(MakePad, args, rv);
  });

RELAY_REGISTER_OP("nn.pad")
.describe(R"code(Pad for n-D tensor.

)code" TVM_ADD_FILELINE)
.set_attrs_type_key("relay.attrs.PadAttrs")
.set_num_inputs(1)
.add_argument("data", "Tensor", "The input tensor.")
.set_support_level(2)
.add_type_rel("Pad", PadRel)
.set_attr<TOpPattern>("TOpPattern", kInjective)
.set_attr<FTVMCompute>("FTVMCompute", PadCompute);

}  // namespace relay
}  // namespace tvm
