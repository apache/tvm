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

namespace {
// Both operands must remain IR values when Python evaluates an overloaded operator.
ExprDoc BinaryOperationDoc(OperationDocNode::Kind kind, PrimExpr lhs, PrimExpr rhs, AccessPath p,
                           IRDocsifier d) {
  ExprDoc a = d->AsDoc<ExprDoc>(lhs, p->Attr("a"));
  ExprDoc b = d->AsDoc<ExprDoc>(rhs, p->Attr("b"));
  if (a->IsInstance<LiteralDocNode>() && b->IsInstance<LiteralDocNode>()) {
    a = TIR(d, DType2Str(lhs.ty()->dtype))->Call({a});
    b = TIR(d, DType2Str(rhs.ty()->dtype))->Call({b});
  }
  return OperationDoc(kind, {a, b});
}
}  // namespace

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
        PrimType a_ty = node->a.ty();
        PrimType b_ty = node->b.ty();
        if (a_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) &&
            b_ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
          return TIR(d, "Div")->Call(
              {d->AsDoc<ExprDoc>(node->a, p->Attr("a")), d->AsDoc<ExprDoc>(node->b, p->Attr("b"))});
        }
        return BinaryOperationDoc(OperationDocNode::Kind::kDiv, node->a, node->b, p, d);
      });
}

#define TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(NodeType, OpKind)                         \
  IRDocsifier::vtable().set_dispatch<prim::NodeType>(                                      \
      "", [](prim::NodeType node, AccessPath p, IRDocsifier d) -> Doc {                    \
        return BinaryOperationDoc(OperationDocNode::Kind::OpKind, node->a, node->b, p, d); \
      });

TVM_FFI_STATIC_INIT_BLOCK() {
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Add, kAdd);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Sub, kSub);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Mul, kMult);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(FloorDiv, kFloorDiv);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(FloorMod, kMod);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(LShift, kLShift);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(RShift, kRShift);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(BitwiseAnd, kBitAnd);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(BitwiseOr, kBitOr);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(BitwiseXor, kBitXor);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(LT, kLt);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(LE, kLtE);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(EQ, kEq);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(NE, kNotEq);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(GT, kGt);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(GE, kGtE);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(And, kAnd);
  TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR(Or, kOr);

  TVM_SCRIPT_PRINTER_DEF_BINARY(Mod, "truncmod");
  TVM_SCRIPT_PRINTER_DEF_BINARY(Min, "min");
  TVM_SCRIPT_PRINTER_DEF_BINARY(Max, "max");
}

#undef TVM_SCRIPT_PRINTER_DEF_BINARY_WITH_SUGAR
#undef TVM_SCRIPT_PRINTER_DEF_BINARY

}  // namespace printer
}  // namespace script
}  // namespace tvm
