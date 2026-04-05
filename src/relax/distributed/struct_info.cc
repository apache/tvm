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

/*!
 * \file src/relax/distributed/struct_info.cc
 * \brief Relax dtensor struct info.
 */

#include <tvm/ffi/ir/traits.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/distributed/struct_info.h>

#include "../ir/script_print_utils.h"

namespace tvm {
namespace relax {
namespace distributed {

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = ::tvm::ffi::reflection;
  DTensorStructInfoNode::RegisterReflection();
  PlacementNode::RegisterReflection();
  PlacementSpecNode::RegisterReflection();
  refl::GlobalDef().def("relax._placement_str", [](Placement node) -> ffi::String {
    return node->ToString();
  });
}

PlacementSpec PlacementSpec::Sharding(int axis) {
  ObjectPtr<PlacementSpecNode> n = ffi::make_object<PlacementSpecNode>();
  n->axis = axis;
  n->kind = PlacementSpecKind::kSharding;
  return PlacementSpec(n);
}

PlacementSpec PlacementSpec::Replica() {
  ObjectPtr<PlacementSpecNode> n = ffi::make_object<PlacementSpecNode>();
  n->axis = -1;
  n->kind = PlacementSpecKind::kReplica;
  return PlacementSpec(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.distributed.Sharding", [](int axis) { return PlacementSpec::Sharding(axis); })
      .def("relax.distributed.Replica", []() { return PlacementSpec::Replica(); });
}

ffi::String PlacementNode::ToString() const {
  std::stringstream ss;
  for (size_t i = 0; i < dim_specs.size(); ++i) {
    if (i != 0) {
      ss << ", ";
    }
    if (dim_specs[i]->kind == PlacementSpecKind::kReplica) {
      ss << "R";
    } else {
      ss << "S[" << dim_specs[i]->axis << "]";
    }
  }
  return ss.str();
}

Placement::Placement(ffi::Array<PlacementSpec> dim_specs) {
  ObjectPtr<PlacementNode> n = ffi::make_object<PlacementNode>();
  n->dim_specs = std::move(dim_specs);
  data_ = std::move(n);
}

Placement Placement::FromText(ffi::String text_repr) {
  ffi::Array<PlacementSpec> dim_specs;
  std::stringstream ss(text_repr);
  while (true) {
    char indicator = 0;
    ss >> indicator;
    if (ss.eof()) {
      break;
    }
    if (indicator == 'R') {
      dim_specs.push_back(PlacementSpec::Replica());
    } else if (indicator == 'S') {
      char lbracket;
      ss >> lbracket;
      TVM_FFI_ICHECK_EQ(lbracket, '[');
      std::string substr;
      getline(ss, substr, ']');
      std::stringstream ss2(substr);
      int dim;
      ss2 >> dim;
      dim_specs.push_back(PlacementSpec::Sharding(dim));
      TVM_FFI_ICHECK(ss2.eof()) << "Invalid placement text repr";
    } else if (indicator == ',') {
      continue;
    } else if (indicator == ' ') {
      continue;
    } else {
      TVM_FFI_THROW(InternalError) << "Invalid placement text repr";
    }
  }
  return Placement(dim_specs);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.distributed.PlacementFromText", Placement::FromText)
      .def("relax.distributed.Placement",
           [](ffi::Array<PlacementSpec> dim_specs) { return Placement(dim_specs); });
}

// DTensor
DTensorStructInfo::DTensorStructInfo(TensorStructInfo tensor_sinfo, DeviceMesh device_mesh,
                                     Placement placement, Span span) {
  TVM_FFI_CHECK_EQ(device_mesh->shape.size(), placement->dim_specs.size(), ValueError)
      << "The device mesh and placement must have the same dimension size";
  for (auto spec : placement->dim_specs) {
    if (spec->kind == PlacementSpecKind::kReplica) continue;
    TVM_FFI_CHECK_LT(spec->axis, tensor_sinfo->ndim, ValueError)
        << "Sharding dimension should be smaller than tensor ndim";
  }
  ObjectPtr<DTensorStructInfoNode> n = ffi::make_object<DTensorStructInfoNode>();
  n->device_mesh = std::move(device_mesh);
  n->placement = std::move(placement);
  n->tensor_sinfo = std::move(tensor_sinfo);
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.distributed.DTensorStructInfo",
      [](TensorStructInfo tensor_sinfo, DeviceMesh device_mesh, Placement placement, Span span) {
        return DTensorStructInfo(tensor_sinfo, device_mesh, placement, span);
      });
}

// ---- __ffi_text_print__ overrides ----

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace printer;
  namespace refl = ::tvm::ffi::reflection;
  namespace text = ::tvm::ffi::ir::text;
  // DeviceMesh: R.device_mesh((dim0, dim1, ...), I.Range(start, end))
  refl::TypeAttrDef<DeviceMeshNode>().def(
      "__ffi_text_print__",
      [](DeviceMesh node, text::IRPrinter printer, text::AccessPath path) -> text::NodeAST {
        // Build shape tuple from ffi::Shape
        ffi::List<text::ExprAST> shape_elts;
        for (size_t i = 0; i < node->shape.size(); ++i) {
          shape_elts.push_back(text::LiteralAST::Int(node->shape[i]));
        }
        text::ExprAST shape_doc = text::TupleAST({}, std::move(shape_elts));
        // Build second arg: I.Range(start, end) or device_ids list
        ffi::List<text::ExprAST> args;
        args.push_back(shape_doc);
        if (node->device_range.defined()) {
          Range r = node->device_range.value();
          text::ExprAST range_begin = Print(printer, r->min, path->Attr("device_range")->Attr("min"));
          // Range stores (min, extent); V1 prints I.Range(min, min + extent)
          PrimExpr end_expr = r->min + r->extent;
          text::ExprAST range_end = Print(printer, end_expr, path->Attr("device_range")->Attr("extent"));
          args.push_back(text::ExprCall(IR("Range"), {range_begin, range_end}));
        } else {
          args.push_back(Print(printer, node->device_ids, path->Attr("device_ids")));
        }
        return text::ExprCall(Relax("device_mesh"), std::move(args));
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  using namespace printer;
  namespace refl = ::tvm::ffi::reflection;
  namespace text = ::tvm::ffi::ir::text;
  // DTensorStructInfo: R.DTensor(shape, dtype, "mesh[i]", "placement_str")
  refl::TypeAttrDef<DTensorStructInfoNode>().def(
      "__ffi_text_print__",
      [](DTensorStructInfo n, text::IRPrinter printer,
         text::AccessPath path) -> text::NodeAST {
        ffi::List<text::ExprAST> args;
        ffi::List<ffi::String> kwargs_keys;
        ffi::List<text::ExprAST> kwargs_values;
        bool require_kwargs = false;
        // Shape from tensor_sinfo
        if (n->tensor_sinfo->shape.defined()) {
          if (const auto* shape = n->tensor_sinfo->shape.value().as<relax::ShapeExprNode>()) {
            auto shape_expr = ffi::GetRef<relax::ShapeExpr>(shape);
            text::AccessPath shape_p = path->Attr("tensor_sinfo")->Attr("shape")->Attr("values");
            ffi::List<text::ExprAST> shape_docs;
            for (int i = 0, ndim = shape_expr->values.size(); i < ndim; ++i) {
              shape_docs.push_back(PrintShapeValue(shape_expr->values[i],
                                                    shape_p->ArrayItem(i), printer,
                                                    /*stringify_vars=*/false));
            }
            args.push_back(text::TupleAST({}, std::move(shape_docs)));
          } else {
            args.push_back(Print(printer, n->tensor_sinfo->shape.value(),
                                  path->Attr("tensor_sinfo")->Attr("shape")));
          }
        } else {
          require_kwargs = true;
        }
        // dtype
        if (!n->tensor_sinfo->IsUnknownDtype()) {
          if (!require_kwargs) {
            args.push_back(text::LiteralAST::Str(DType2Str(n->tensor_sinfo->dtype)));
          } else {
            kwargs_keys.push_back(ffi::String("dtype"));
            kwargs_values.push_back(text::LiteralAST::Str(DType2Str(n->tensor_sinfo->dtype)));
          }
        } else {
          require_kwargs = true;
        }
        // device_mesh: print as string reference "mesh[i]" or inline
        if (!require_kwargs) {
          args.push_back(Print(printer, n->device_mesh, path->Attr("device_mesh")));
        } else {
          kwargs_keys.push_back(ffi::String("device_mesh"));
          kwargs_values.push_back(Print(printer, n->device_mesh, path->Attr("device_mesh")));
        }
        // placement
        if (!require_kwargs) {
          args.push_back(Print(printer, n->placement, path->Attr("placement")));
        } else {
          kwargs_keys.push_back(ffi::String("placement"));
          kwargs_values.push_back(Print(printer, n->placement, path->Attr("placement")));
        }
        // ndim when shape is not defined
        if (!n->tensor_sinfo->shape.defined() && !n->tensor_sinfo->IsUnknownNdim()) {
          kwargs_keys.push_back(ffi::String("ndim"));
          kwargs_values.push_back(text::LiteralAST::Int(n->tensor_sinfo->ndim));
        }
        return text::ExprCallKw(Relax("DTensor"), std::move(args),
                          std::move(kwargs_keys), std::move(kwargs_values));
      });
}

}  // namespace distributed
}  // namespace relax
}  // namespace tvm
