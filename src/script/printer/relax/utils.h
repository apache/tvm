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
#ifndef TVM_SCRIPT_PRINTER_RELAX_UTILS_H_
#define TVM_SCRIPT_PRINTER_RELAX_UTILS_H_

#include <tvm/relax/analysis.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/type.h>
#include <tvm/relax/utils.h>
#include <tvm/script/printer/ir_docsifier.h>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace script {
namespace printer {

class RelaxFrameNode : public FrameNode {
 public:
  bool is_func = false;
  bool module_alias_printed = false;
  std::unordered_set<const tir::VarNode*>* func_vars = nullptr;

  void VisitAttrs(AttrVisitor* v) {
    FrameNode::VisitAttrs(v);
    v->Visit("is_global_func", &is_func);
    // `func_var_to_define` is not visited
  }

  static constexpr const char* _type_key = "script.printer.RelaxFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(RelaxFrameNode, FrameNode);
};

class RelaxFrame : public Frame {
 public:
  explicit RelaxFrame(const IRDocsifier& d) {
    ObjectPtr<RelaxFrameNode> n = make_object<RelaxFrameNode>();
    n->stmts.clear();
    n->d = d.get();
    n->is_func = false;
    n->func_vars = nullptr;
    data_ = std::move(n);
  }

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(RelaxFrame, Frame, RelaxFrameNode);
};

/*! \brief Redirected method for the ReprPrinter */
inline std::string ReprPrintRelax(const ObjectRef& obj, const PrinterConfig& cfg) {
  IRDocsifier d(cfg);
  With<RelaxFrame> f(d);
  (*f)->AddDispatchToken(d, "relax");
  return Docsify(obj, d, *f, cfg);
}

inline IdDoc DefineVar(const relax::Var& var, const Frame& frame, const IRDocsifier& d) {
  return d->Define(var, frame, var->name_hint().empty() ? "v" : var->name_hint());
}

inline Optional<ExprDoc> StructInfoAsAnn(const relax::Var& v, const ObjectPath& v_p,
                                         const IRDocsifier& d, const Optional<relax::Expr>& rhs) {
  if (!v->struct_info_.defined()) {
    return NullOpt;
  }
  bool attempt_to_hide_struct_info = !d->cfg->show_all_struct_info;

  if (const auto* call = rhs.as<relax::CallNode>()) {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    static const Op& call_dps_packed_op = Op::Get("relax.call_dps_packed");
    if (call->op.same_as(call_tir_op) || call->op.same_as(call_dps_packed_op)) {
      attempt_to_hide_struct_info = true;
    }
  }
  if (attempt_to_hide_struct_info) {
    Optional<relax::StructInfo> inferred_sinfo = NullOpt;
    if (auto opt = rhs.as<relax::Call>()) {
      auto call = opt.value();
      if (auto opt = call->op.as<Op>()) {
        auto op = opt.value();

        static auto op_map_infer_struct_info =
            Op::GetAttrMap<relax::FInferStructInfo>("FInferStructInfo");

        auto temp_builder = relax::BlockBuilder::Create(NullOpt);
        inferred_sinfo = op_map_infer_struct_info[op](call, temp_builder);
      } else if (auto opt = call->op.as<relax::FuncStructInfo>()) {
        auto temp_builder = relax::BlockBuilder::Create(NullOpt);
        inferred_sinfo =
            DeriveCallRetStructInfo(opt.value(), call, temp_builder, temp_builder->GetAnalyzer());
      }

    } else if (const auto* tuple = rhs.as<relax::TupleNode>()) {
      inferred_sinfo = relax::TupleStructInfo(tuple->fields.Map(relax::GetStructInfo));

    } else if (const auto* get_item = rhs.as<relax::TupleGetItemNode>()) {
      if (auto ptr = get_item->tuple->struct_info_.as<relax::TupleStructInfoNode>();
          ptr && get_item->index < static_cast<int>(ptr->fields.size())) {
        inferred_sinfo = ptr->fields[get_item->index];
      }

    } else if (const auto* trivial_binding = rhs.as<relax::VarNode>()) {
      inferred_sinfo = trivial_binding->struct_info_.as<relax::StructInfo>();
    }

    if (inferred_sinfo && StructuralEqual()(inferred_sinfo, v->struct_info_)) {
      return NullOpt;
    }
  }
  return d->AsDoc<ExprDoc>(v->struct_info_, v_p->Attr("struct_info_"));
}

Array<StmtDoc> PrintSeqExpr(const relax::SeqExpr& n, const ObjectPath& n_p, const IRDocsifier& d,
                            bool use_ret);

ExprDoc PrintShapeVar(const PrimExpr& e, const ObjectPath& e_p, const IRDocsifier& d);

inline int FindVDeviceIndexByTargetKind(const VDevice& vdevice, const IRDocsifier& d) {
  Array<GlobalInfo> vdevices = d->global_infos["vdevice"];
  int kind_index = 0;
  for (size_t i = 0; i < vdevices.size(); ++i) {
    auto vdev = Downcast<VDevice>(vdevices[i]);
    if (vdev.same_as(vdevice)) {
      return kind_index;
    }
    if (vdev->target->kind->name == vdevice->target->kind->name) {
      kind_index++;
    }
  }
  return -1;
}

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_RELAX_UTILS_H_
