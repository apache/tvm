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
#include <tvm/ir/expr.h>
#include <tvm/relax/distributed/struct_info.h>

#include "../ir/utils.h"
#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

// distributed::Placement
TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::distributed::Placement>("",
                                                 [](relax::distributed::Placement n, ObjectPath n_p,
                                                    IRDocsifier d) -> Doc {
                                                   return d->AsDoc<Doc>(n->ToString(), n_p);
                                                 });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::distributed::DTensorStructInfo>(
        "", [](relax::distributed::DTensorStructInfo n, ObjectPath n_p, IRDocsifier d) -> Doc {
          Array<ExprDoc> args;
          Array<String> kwargs_keys;
          Array<ExprDoc> kwargs_values;
          bool require_kwargs = false;
          if (n->tensor_sinfo->shape.defined()) {
            // Need to dig into ShapeExpr to preserve the `R.shape` prefix
            if (const auto* shape = n->tensor_sinfo->shape.value().as<relax::ShapeExprNode>()) {
              auto shape_expr = GetRef<relax::ShapeExpr>(shape);
              ObjectPath shape_p = n_p->Attr("shape")->Attr("values");
              Array<ExprDoc> shape_docs;
              for (int i = 0, ndim = shape_expr->values.size(); i < ndim; ++i) {
                shape_docs.push_back(
                    PrintShapeVar(shape_expr->values[i], shape_p->ArrayIndex(i), d));
              }
              args.push_back(TupleDoc(shape_docs));
            } else {
              args.push_back(d->AsDoc<ExprDoc>(n->tensor_sinfo->shape.value(), n_p->Attr("shape")));
            }
          } else {
            require_kwargs = true;
          }
          if (!n->tensor_sinfo->IsUnknownDtype()) {
            if (!require_kwargs) {
              args.push_back(LiteralDoc::DataType(n->tensor_sinfo->dtype, n_p->Attr("dtype")));
            } else {
              kwargs_keys.push_back("dtype");
              kwargs_values.push_back(
                  LiteralDoc::DataType(n->tensor_sinfo->dtype, n_p->Attr("dtype")));
            }
          } else {
            require_kwargs = true;
          }
          if (!require_kwargs) {
            args.push_back(d->AsDoc<ExprDoc>(n->device_mesh, n_p->Attr("device_mesh")));
          } else {
            kwargs_keys.push_back("device_mesh");
            kwargs_values.push_back(d->AsDoc<ExprDoc>(n->device_mesh, n_p->Attr("device_mesh")));
          }
          if (!require_kwargs) {
            args.push_back(d->AsDoc<ExprDoc>(n->placement, n_p->Attr("placement")));
          } else {
            kwargs_keys.push_back("placement");
            kwargs_values.push_back(d->AsDoc<ExprDoc>(n->placement, n_p->Attr("placement")));
          }
          if (!n->tensor_sinfo->shape.defined() && !n->tensor_sinfo->IsUnknownNdim()) {
            kwargs_keys.push_back("ndim");
            kwargs_values.push_back(LiteralDoc::Int(n->tensor_sinfo->ndim, n_p->Attr("ndim")));
          }
          return Relax(d, "DTensor")->Call(args, kwargs_keys, kwargs_values);
        });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::distributed::DeviceMesh>(
        "", [](relax::distributed::DeviceMesh n, ObjectPath n_p, IRDocsifier d) -> Doc {
          bool has_relax_frame = false;
          const IRFrameNode* f = nullptr;
          for (const Frame& frame : d->frames) {
            if (frame.as<RelaxFrameNode>()) {
              has_relax_frame = true;
              break;
            } else if (const auto* ir_frame = frame.as<IRFrameNode>()) {
              f = ir_frame;
            }
          }
          if (!has_relax_frame || !f) {
            Array<ExprDoc> args;
            args.push_back(d->AsDoc<ExprDoc>(n->shape, n_p->Attr("shape")));
            if (n->device_range.defined()) {
              args.push_back(d->AsDoc<ExprDoc>(n->device_range, n_p->Attr("device_range")));
            } else {
              args.push_back(d->AsDoc<ExprDoc>(n->device_ids, n_p->Attr("device_ids")));
            }
            return Relax(d, "device_mesh")->Call(args);
          } else {
            for (const auto& kv : *f->global_infos) {
              for (int i = 0; i < static_cast<int>(kv.second.size()); i++) {
                if (kv.second[i].same_as(n)) {
                  std::stringstream ss;
                  ss << kv.first << "[" << i << "]";
                  return d->AsDoc<Doc>(String(ss.str()), n_p);
                }
              }
            }
            LOG(FATAL) << "Cannot find device mesh in global infos";
          }
        });

TVM_SCRIPT_REPR(relax::distributed::DeviceMeshNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::distributed::PlacementNode, ReprPrintRelax);
TVM_SCRIPT_REPR(relax::distributed::DTensorStructInfoNode, ReprPrintRelax);
}  // namespace printer
}  // namespace script
}  // namespace tvm
