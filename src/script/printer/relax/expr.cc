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

Optional<ExprDoc> InlineConstant(const runtime::NDArray& array, const ObjectPath& path) {
  if (array->device.device_type != kDLCPU) {
    return NullOpt;
  }

  DataType dtype = array.DataType();
  std::function<ExprDoc(void*, const ObjectPath&)> element_printer;
  if (dtype == DataType::Int(8)) {
    element_printer = [](void* ptr, const ObjectPath& elem_path) {
      return LiteralDoc::Int(*reinterpret_cast<const int8_t*>(ptr), elem_path);
    };
  } else if (dtype == DataType::Int(16)) {
    element_printer = [](void* ptr, const ObjectPath& elem_path) {
      return LiteralDoc::Int(*reinterpret_cast<const int16_t*>(ptr), elem_path);
    };
  } else if (dtype == DataType::Int(32)) {
    element_printer = [](void* ptr, const ObjectPath& elem_path) {
      return LiteralDoc::Int(*reinterpret_cast<const int32_t*>(ptr), elem_path);
    };
  } else if (dtype == DataType::Int(64)) {
    element_printer = [](void* ptr, const ObjectPath& elem_path) {
      return LiteralDoc::Int(*reinterpret_cast<const int64_t*>(ptr), elem_path);
    };
  } else if (dtype == DataType::Float(16)) {
    element_printer = [](void* ptr, const ObjectPath& elem_path) {
      // From IEEE-754 float16 definition
      //
      // Ref: https://en.wikipedia.org/wiki/Half-precision_floating-point_format
      uint16_t bits = *reinterpret_cast<const uint16_t*>(ptr);
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

      return LiteralDoc::Float(value, elem_path);
    };
  } else if (dtype == DataType::Float(32)) {
    element_printer = [](void* ptr, const ObjectPath& elem_path) {
      return LiteralDoc::Float(*reinterpret_cast<const float*>(ptr), elem_path);
    };
  } else if (dtype == DataType::Float(64)) {
    element_printer = [](void* ptr, const ObjectPath& elem_path) {
      return LiteralDoc::Float(*reinterpret_cast<const double*>(ptr), elem_path);
    };
  } else if (dtype == DataType::Bool()) {
    element_printer = [](void* ptr, const ObjectPath& elem_path) {
      return LiteralDoc::Boolean(*reinterpret_cast<const uint8_t*>(ptr), elem_path);
    };

  } else {
    return NullOpt;
  }

  size_t elem_nbytes = (array->dtype.bits * array->dtype.lanes + 7) / 8;
  void* base_ptr = static_cast<char*>(array->data) + array->byte_offset;
  auto get_ptr_to_element = [&](std::vector<size_t> indices) -> void* {
    ICHECK_EQ(indices.size(), array->ndim);

    size_t elem_offset = 0;
    if (array->strides) {
      for (int i = 0; i < array->ndim; i++) {
        elem_offset += indices[i] * array->strides[i];
      }
    } else {
      for (int i = 0; i < array->ndim; i++) {
        elem_offset *= array->shape[i];
        elem_offset += indices[i];
      }
    }

    return static_cast<char*>(base_ptr) + elem_offset * elem_nbytes;
  };

  if (array->ndim == 0) {
    return element_printer(get_ptr_to_element({}), path);
  } else if (array->ndim == 1) {
    Array<ExprDoc> elements;
    for (size_t i = 0; i < array->shape[0]; i++) {
      elements.push_back(element_printer(get_ptr_to_element({i}), path->ArrayIndex(i)));
    }
    return ListDoc(elements);
  } else if (array->ndim == 2) {
    Array<ExprDoc> elements;
    for (size_t i = 0; i < array->shape[0]; i++) {
      Array<ExprDoc> row;
      for (size_t j = 0; j < array->shape[1]; j++) {
        row.push_back(
            element_printer(get_ptr_to_element({i, j}), path->ArrayIndex(i)->ArrayIndex(j)));
      }
      elements.push_back(ListDoc(row));
    }
    return ListDoc(elements);
  } else {
    // For now, only supporting inline constants with low
    // dimensionality.  Can generalize later if necessary.
    return NullOpt;
  }
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::Constant>(  //
        "", [](relax::Constant node, ObjectPath path, IRDocsifier d) -> Doc {
          ExprDoc value = [&]() {
            if (node->data.Shape()->Product() <= 16) {
              if (Optional<ExprDoc> opt = InlineConstant(node->data, path->Attr("data"))) {
                return opt.value();
              }
            }

            return d->AddMetadata(node->data);
          }();

          ExprDoc ann = [&]() -> ExprDoc {
            if (auto tensor_sinfo = node->struct_info_.as<relax::TensorStructInfoNode>()) {
              if (tensor_sinfo->ndim == 0) {
                return LiteralDoc::DataType(tensor_sinfo->dtype,
                                            path->Attr("struct_info_")->Attr("dtype"));
              }
            }
            return d->AsDoc<ExprDoc>(node->struct_info_, path->Attr("struct_info_"));
          }();

          auto type_name = (node->struct_info_.as<relax::distributed::DTensorStructInfoNode>())
                               ? "dist.const"
                               : "const";

          return Relax(d, type_name)->Call({value, ann});
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
