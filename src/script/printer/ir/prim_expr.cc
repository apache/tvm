/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/ir/prim/op.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<prim::BitwiseNot>(
      "", [](prim::BitwiseNot node, AccessPath p, IRDocsifier d) -> Doc {
        ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));
        if (a->IsInstance<LiteralDocNode>()) {
          return TIR(d, "BitwiseNot")->Call({a});
        }
        return OperationDoc(OperationDocNode::Kind::kInvert, {a});
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<prim::Not>(
      "", [](prim::Not node, AccessPath p, IRDocsifier d) -> Doc {
        ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));
        if (a->IsInstance<LiteralDocNode>()) {
          return TIR(d, "Not")->Call({a});
        }
        return OperationDoc(OperationDocNode::Kind::kNot, {a});
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<StringImm>(
      "", [](StringImm s, AccessPath p, IRDocsifier d) -> Doc {
        if (HasMultipleLines(s->value)) {
          return d->AddMetadata(s);
        } else {
          return d->AsDoc<ExprDoc>(s->value, p->Attr("value"));
        }
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<prim::Cast>(
      "", [](prim::Cast cast, AccessPath p, IRDocsifier d) -> Doc {
        ExprDoc dtype = LiteralDoc::DataType(cast.ty()->dtype, p->Attr("dtype"));
        ExprDoc value = d->AsDoc<ExprDoc>(cast->value, p->Attr("value"));
        return TIR(d, "Cast")->Call({dtype, value});
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<prim::Select>(
      "", [](prim::Select select, AccessPath p, IRDocsifier d) -> Doc {
        return TIR(d, "Select")
            ->Call({
                d->AsDoc<ExprDoc>(select->condition, p->Attr("condition")),
                d->AsDoc<ExprDoc>(select->true_value, p->Attr("true_value")),
                d->AsDoc<ExprDoc>(select->false_value, p->Attr("false_value")),
            });
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<prim::Ramp>(
      "", [](prim::Ramp ramp, AccessPath ramp_p, IRDocsifier d) -> Doc {
        return TIR(d, "Ramp")->Call({
            d->AsDoc<ExprDoc>(ramp->base, ramp_p->Attr("base")),
            d->AsDoc<ExprDoc>(ramp->stride, ramp_p->Attr("stride")),
            d->AsDoc<ExprDoc>(ramp->lanes, ramp_p->Attr("lanes")),
        });
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<prim::Broadcast>(
      "", [](prim::Broadcast bc, AccessPath bc_p, IRDocsifier d) -> Doc {
        return TIR(d, "Broadcast")
            ->Call({
                d->AsDoc<ExprDoc>(bc->value, bc_p->Attr("value")),
                d->AsDoc<ExprDoc>(bc->lanes, bc_p->Attr("lanes")),
            });
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<prim::Shuffle>(  //
      "", [](prim::Shuffle shuffle, AccessPath p, IRDocsifier d) -> Doc {
        return TIR(d, "Shuffle")
            ->Call({
                d->AsDoc<ExprDoc>(shuffle->vectors, p->Attr("vectors")),
                d->AsDoc<ExprDoc>(shuffle->indices, p->Attr("indices")),
            });
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<prim::Let>(
      "", [](prim::Let let, AccessPath p, IRDocsifier d) -> Doc {
        DictDoc where({d->AsDoc<ExprDoc>(let->var, p->Attr("var"))},
                      {d->AsDoc<ExprDoc>(let->value, p->Attr("value"))});
        return TIR(d, "Let")->Call({d->AsDoc<ExprDoc>(let->body, p->Attr("body"))},  //
                                   {"where"}, {where});
      });
}

#define TVM_SCRIPT_PRINTER_DEF_BINARY(NodeType, OpString)               \
  IRDocsifier::vtable().set_dispatch<prim::NodeType>(                   \
      "", [](prim::NodeType node, AccessPath p, IRDocsifier d) -> Doc { \
        ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));           \
        ExprDoc b = d->AsDoc<ExprDoc>(node->b, p->Attr("b"));           \
        return TIR(d, OpString)->Call({a, b});                          \
      });

TVM_FFI_STATIC_INIT_BLOCK() {
  IRDocsifier::vtable().set_dispatch<prim::Div>(
      "", [](prim::Div node, AccessPath p, IRDocsifier d) -> Doc {
        ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));
        ExprDoc b = d->AsDoc<ExprDoc>(node->b, p->Attr("b"));
        PrimExpr ret = tvm::div(node->a, node->b);
        if (!ret->IsInstance<prim::DivNode>()) {
          return TIR(d, "Div")->Call({a, b});
        }
        PrimType a_ty = node->a.ty();
        PrimType b_ty = node->b.ty();
        if (a_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) &&
            b_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
          return TIR(d, "Div")->Call({a, b});
        }
        return OperationDoc(OperationDocNode::Kind::kDiv, {a, b});
      });
}

#define TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(NodeType, NodeObj, NodeFunc, OpString, OpKind) \
  IRDocsifier::vtable().set_dispatch<prim::NodeType>(                                           \
      "", [](prim::NodeType node, AccessPath p, IRDocsifier d) -> Doc {                         \
        ExprDoc a = d->AsDoc<ExprDoc>(node->a, p->Attr("a"));                                   \
        ExprDoc b = d->AsDoc<ExprDoc>(node->b, p->Attr("b"));                                   \
        PrimExpr ret = tvm::NodeFunc(node->a, node->b);                                         \
        if (const auto* ret_node = ret.as<tvm::NodeObj>()) {                                    \
          if (ret_node->a.same_as(node->a) && ret_node->b.same_as(node->b)) {                   \
            return OperationDoc(OperationDocNode::Kind::OpKind, {a, b});                        \
          }                                                                                     \
        }                                                                                       \
        return TIR(d, OpString)->Call({a, b});                                                  \
      });

TVM_FFI_STATIC_INIT_BLOCK() {
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Add, prim::AddNode, add, "Add", kAdd);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Sub, prim::SubNode, sub, "Sub", kSub);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Mul, prim::MulNode, mul, "Mul", kMult);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(FloorDiv, prim::FloorDivNode, floordiv, "FloorDiv",
                                           kFloorDiv);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(FloorMod, prim::FloorModNode, floormod, "FloorMod",
                                           kMod);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(LShift, prim::LShiftNode, left_shift, "LShift", kLShift);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(RShift, prim::RShiftNode, right_shift, "RShift",
                                           kRShift);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(BitwiseAnd, prim::BitwiseAndNode, bitwise_and,
                                           "BitwiseAnd", kBitAnd);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(BitwiseOr, prim::BitwiseOrNode, bitwise_or, "BitwiseOr",
                                           kBitOr);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(BitwiseXor, prim::BitwiseXorNode, bitwise_xor,
                                           "BitwiseXor", kBitXor);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(LT, prim::LTNode, less, "LT", kLt);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(LE, prim::LENode, less_equal, "LE", kLtE);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(EQ, prim::EQNode, equal, "EQ", kEq);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(NE, prim::NENode, not_equal, "NE", kNotEq);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(GT, prim::GTNode, greater, "GT", kGt);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(GE, prim::GENode, greater_equal, "GE", kGtE);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(And, prim::AndNode, logical_and, "And", kAnd);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Or, prim::OrNode, logical_or, "Or", kOr);

  TVM_SCRIPT_PRINTER_DEF_BINARY(Mod, "truncmod");
  TVM_SCRIPT_PRINTER_DEF_BINARY(Min, "min");
  TVM_SCRIPT_PRINTER_DEF_BINARY(Max, "max");
}

#undef TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR
#undef TVM_SCRIPT_PRINTER_DEF_BINARY

}  // namespace printer
}  // namespace script
}  // namespace tvm
