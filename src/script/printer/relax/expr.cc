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

#include <tvm/relax/distributed/struct_info.h>

#include <limits>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::PrimValue>(  //
        "", [](relax::PrimValue n, ObjectPath n_p, IRDocsifier d) -> Doc {
          // TODO(@junrushao): float numbers
          return Relax(d, "prim_value")->Call({d->AsDoc<ExprDoc>(n->value, n_p->Attr("value"))});
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::StringImm>(  //
        "", [](relax::StringImm n, ObjectPath n_p, IRDocsifier d) -> Doc {
          return Relax(d, "str")->Call({LiteralDoc::Str(n->value, n_p->Attr("value"))});
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::DataTypeImm>(  //
        "", [](relax::DataTypeImm n, ObjectPath n_p, IRDocsifier d) -> Doc {
          return Relax(d, "dtype")->Call({LiteralDoc::DataType(n->value, n_p->Attr("value"))});
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::Tuple>(  //
        "", [](relax::Tuple n, ObjectPath n_p, IRDocsifier d) -> Doc {
          // TODO(@junrushao): revisit tuple printing
          if (n->fields.empty()) {
            return Relax(d, "tuple")->Call({});
          }
          Array<ExprDoc> fields_doc;
          ObjectPath fields_p = n_p->Attr("fields");
          for (int i = 0, l = n->fields.size(); i < l; ++i) {
            fields_doc.push_back(d->AsDoc<ExprDoc>(n->fields[i], fields_p->ArrayIndex(i)));
          }
          return TupleDoc(fields_doc);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::TupleGetItem>(  //
        "", [](relax::TupleGetItem n, ObjectPath n_p, IRDocsifier d) -> Doc {
          ExprDoc idx = LiteralDoc::Int(n->index, n_p->Attr("index"));
          return d->AsDoc<ExprDoc>(n->tuple, n_p->Attr("tuple"))[{idx}];
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::ShapeExpr>(  //
        "", [](relax::ShapeExpr n, ObjectPath n_p, IRDocsifier d) -> Doc {
          Array<ExprDoc> values_doc;
          ObjectPath values_p = n_p->Attr("values");
          for (int i = 0, l = n->values.size(); i < l; ++i) {
            values_doc.push_back(PrintShapeVar(n->values[i], values_p->ArrayIndex(i), d));
          }
          return Relax(d, "shape")->Call({ListDoc(values_doc)});
        });

Optional<ExprDoc> SpecialScalar(const runtime::NDArray& n, const ObjectPath& p) {
  DataType dtype = n.DataType();
  const void* data = n->data;
  if (n->ndim != 0 || n->device.device_type != kDLCPU) {
    return NullOpt;
  }

  if (dtype == DataType::Int(8)) {
    return LiteralDoc::Int(*reinterpret_cast<const int8_t*>(data), p);
  } else if (dtype == DataType::Int(16)) {
    return LiteralDoc::Int(*reinterpret_cast<const int16_t*>(data), p);
  } else if (dtype == DataType::Int(32)) {
    return LiteralDoc::Int(*reinterpret_cast<const int32_t*>(data), p);
  } else if (dtype == DataType::Int(64)) {
    return LiteralDoc::Int(*reinterpret_cast<const int64_t*>(data), p);
  } else if (dtype == DataType::Float(16)) {
    // From IEEE-754 float16 definition
    //
    // Ref: https://en.wikipedia.org/wiki/Half-precision_floating-point_format
    uint16_t bits = *reinterpret_cast<const uint16_t*>(data);
    uint16_t sign_bit = (bits & 0b1000'0000'0000'0000) >> 15;
    uint16_t exponent = (bits & 0b0111'1100'0000'0000) >> 10;
    uint16_t fraction = (bits & 0b0000'0011'1111'1111) >> 0;

    double value;
    if (exponent == 0b1'1111 && fraction == 0) {
      value = std::numeric_limits<double>::infinity();
    } else if (exponent == 0b1'1111) {
      value = std::numeric_limits<double>::quiet_NaN();
    } else if (exponent == 0 && fraction == 0) {
      value = 0.0;
    } else if (exponent == 0) {
      value = std::pow(2.0, -24) * static_cast<double>(fraction);
    } else {
      value = std::pow(2.0, static_cast<double>(exponent) - 25) *
              static_cast<double>(fraction | (1 << 10));
    }
    if (sign_bit) {
      value *= -1.0;
    }

    return LiteralDoc::Float(value, p);
  } else if (dtype == DataType::Float(32)) {
    return LiteralDoc::Float(*reinterpret_cast<const float*>(data), p);
  } else if (dtype == DataType::Float(64)) {
    return LiteralDoc::Float(*reinterpret_cast<const double*>(data), p);
  } else if (dtype == DataType::Bool()) {
    return LiteralDoc::Boolean(*reinterpret_cast<const uint8_t*>(data), p);
  } else {
    return NullOpt;
  }
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::Constant>(  //
        "", [](relax::Constant n, ObjectPath n_p, IRDocsifier d) -> Doc {
          if (Optional<ExprDoc> s = SpecialScalar(n->data, n_p->Attr("data"))) {
            if (n->struct_info_.as<relax::distributed::DTensorStructInfoNode>()) {
              ExprDoc ann = d->AsDoc<ExprDoc>(n->struct_info_, n_p->Attr("struct_info_"));
              return Relax(d, "dist.const")->Call({s.value(), ann});
            }
            return Relax(d, "const")
                ->Call({
                    s.value(),
                    LiteralDoc::DataType(n->data.DataType(), n_p->Attr("data")->Attr("dtype")),
                });
          }
          return d->AddMetadata(n);
        });

Doc PrintRelaxVar(relax::Var n, ObjectPath p, IRDocsifier d) {
  if (!d->IsVarDefined(n)) {
    ExprDoc ann = d->AsDoc<ExprDoc>(n->struct_info_, p->Attr("struct_info_"));
    Frame f = d->frames.back();
    ExprDoc var = DefineVar(n, f, d);
    f->stmts.push_back(AssignDoc(var, NullOpt, ann));
  }
  return d->GetVarDoc(n).value();
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<relax::Var>("", PrintRelaxVar);
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable).set_dispatch<relax::DataflowVar>("", PrintRelaxVar);

TVM_SCRIPT_REPR(relax::PrimValueNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::StringImmNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::DataTypeImmNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::TupleNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::TupleGetItemNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::ShapeExprNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::VarNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::DataflowVarNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::ConstantNode, ReprPrintRelax);

}  // namespace printer
}  // namespace script
}  // namespace tvm
