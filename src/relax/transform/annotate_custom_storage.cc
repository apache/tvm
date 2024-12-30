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
 * \file src/relax/transform/annotate_texture_storage.cc
 * \brief Texture Storage Annotation Pass.
 */

#include <tvm/node/serialization.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/nested_msg.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/index_map.h>

#include <tuple>

#include "../op/tensor/manipulate.h"
#include "infer_layout_utils.h"
#include "utils.h"

namespace tvm {
namespace relax {

using tvm::tir::Buffer;

static Array<PrimExpr> GetShapeFromTensorStructInfo(const TensorStructInfo& tensor_sinfo) {
  auto shape = tensor_sinfo->GetShape();
  ICHECK(shape.defined());
  return shape.value();
}

class CollectProduserScopeInfo : public ExprVisitor {
 public:
  using ExprVisitor::VisitExpr_;

  Map<Expr, StructInfo> Collect(const IRModule& mod, Function func,
                                const Map<Expr, Map<Expr, Array<String>>>& scope_info,
                                const Target& target) {
    mod_ = mod;
    scope_info_ = scope_info;
    target_ = target;
    VisitExpr(func->body);

    return producer_sinfo;
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call) final {
    ExprVisitor::VisitBinding_(binding, call);

    static const Op& call_tir_op = Op::Get("relax.call_tir");
    StructInfo out_sinfo;

    if (call->op == call_tir_op) {
      out_sinfo = call->sinfo_args[0];
    } else {
      return;
    }

    std::unordered_map<String, int> scope_count;

    auto arg_var = binding->var.as<VarNode>();
    if (scope_info_.find(GetRef<Expr>(arg_var)) != scope_info_.end()) {
      for (const auto& val : scope_info_[GetRef<Expr>(arg_var)]) {
        auto call_node = Downcast<Call>(val.first);
        if (scope_count.find(val.second[0]) == scope_count.end()) {
          scope_count.insert({val.second[0], 1});
        } else {
          auto curr_count = scope_count[val.second[0]];
          scope_count.emplace(val.second[0], curr_count + 1);
        }
      }
    }
    String final_scope = "global";
    int count = 0;
    for (const auto& sval : scope_count) {
      if (sval.second > count) {
        final_scope = sval.first;
        count = sval.second;
      }
    }
    StructInfo updated_ret_sinfo = UpdateStructInfo(out_sinfo, {final_scope});
    producer_sinfo.Set(GetRef<Expr>(call), updated_ret_sinfo);
  }

 private:
  StructInfo UpdateStructInfo(const StructInfo& out_sinfo, Array<String> scope) {
    if (out_sinfo->IsInstance<TensorStructInfoNode>()) {
      auto tensor_sinfo = Downcast<TensorStructInfo>(out_sinfo);
      auto shape_arr = GetShapeFromTensorStructInfo(tensor_sinfo);
      return TensorStructInfo(ShapeExpr(shape_arr), tensor_sinfo->dtype,
                              VDevice(target_, 0, scope[0]));
    }

    ICHECK(out_sinfo->IsInstance<TupleStructInfoNode>())
        << "Expect output struct info of call_tir to be either TupleStructInfo or "
           "TensorStructInfo, but got "
        << out_sinfo;

    const auto& tuple_sinfo = Downcast<TupleStructInfo>(out_sinfo);
    Array<StructInfo> sinfo_fields;
    for (const auto& si : tuple_sinfo->fields) {
      ICHECK(si->IsInstance<TensorStructInfoNode>())
          << "Fields of TupleStructInfo must be TensorStructInfo for call_tir "
             "output structinfo, but got "
          << si;
      auto sinfo = Downcast<TensorStructInfo>(si);
      auto shape_arr = GetShapeFromTensorStructInfo(sinfo);
      sinfo_fields.push_back(
          TensorStructInfo(ShapeExpr(shape_arr), sinfo->dtype, VDevice(target_, 0, scope[0])));
    }
    return TupleStructInfo(sinfo_fields);
  }

  Map<Expr, Map<Expr, Array<String>>> scope_info_;
  Map<Expr, StructInfo> producer_sinfo;
  IRModule mod_;
  Target target_;
};

class CollectConsumerScopeInfo : public ExprVisitor {
 public:
  using ExprVisitor::VisitExpr_;

  std::pair<Map<Expr, Array<String>>, Map<Expr, Map<Expr, Array<String>>>> Collect(
      const IRModule& mod, Function func, const Target& target) {
    mod_ = mod;
    target_ = target;
    VisitExpr(func->body);
    for (const auto& val : arg_to_binding) {
      if (scope_info.find(val.first) != scope_info.end()) {
        if (scope_info.find(val.second) == scope_info.end()) {
          scope_info.Set(val.second, scope_info[val.first]);
        } else {
          auto ent = scope_info[val.second];
          for (auto ent_val : scope_info[val.first]) {
            ent.Set(ent_val.first, ent_val.second);
          }
          scope_info.Set(val.second, ent);
        }
      }
    }

    return std::make_pair(call_scope_info, scope_info);
  }

  void VisitBinding_(const VarBindingNode* binding,
                     const TupleGetItemNode* tuple_get_item_node) final {
    if (arg_to_binding.find(GetRef<Expr>(binding->var.get())) == arg_to_binding.end()) {
      arg_to_binding.Set(GetRef<Expr>(binding->var.get()),
                         GetRef<Expr>(tuple_get_item_node->tuple.get()));
    }
  }

  void VisitExpr_(const CallNode* call) final {
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    GlobalVar gv;
    Array<Attrs> op_attrs;
    Optional<Integer> op_pattern = Integer(static_cast<int>(relay::kOpaque));
    Tuple func_args;

    StructInfo out_sinfo;

    if (call->op == call_tir_op) {
      gv = Downcast<GlobalVar>(call->args[0]);
      tir::PrimFunc pfunc = Downcast<tir::PrimFunc>(mod_->Lookup(gv));
      op_attrs = ExtractAttrs<tir::PrimFunc>(pfunc);
      op_pattern = ExtractPattern<tir::PrimFunc>(pfunc);
      out_sinfo = call->sinfo_args[0];
      func_args = Downcast<Tuple>(call->args[1]);
    } else {
      return;
    }

    bool is_texture_supported = SupportsTexture(op_attrs, op_pattern.value());

    Array<String> arg_scope;
    for (auto arg : func_args->fields) {
      auto sinfo = GetStructInfo(arg);
      if (auto tensor_sinfo = sinfo.as<TensorStructInfo>()) {
        auto scope = is_texture_supported
                         ? Scope(GetShapeFromTensorStructInfo(tensor_sinfo.value()))
                         : "global";
        Map<Expr, Array<String>> ent_call;
        const VarNode* arg_var = arg.as<VarNode>();
        if (scope_info.find(GetRef<Expr>(arg_var)) != scope_info.end()) {
          ent_call = scope_info[GetRef<Expr>(arg_var)];
        }
        ent_call.Set(GetRef<Expr>(call), {scope});
        scope_info.Set(GetRef<Expr>(arg_var), ent_call);
        arg_scope.push_back(scope);
      }
    }
    call_scope_info.Set(GetRef<Expr>(call), arg_scope);
  }

 private:
  template <typename T>
  Array<Attrs> ExtractAttrs(const T& func) {
    Array<Attrs> op_attrs;
    Optional<ObjectRef> attrs = func->template GetAttr<ObjectRef>("op_attrs");
    if (attrs) {
      if (auto val = attrs.value().as<Attrs>()) {
        op_attrs.push_back(val.value());
      } else if (auto val = attrs.value().as<Array<Attrs>>()) {
        op_attrs = val.value();
      }
    }
    return std::move(op_attrs);
  }

  template <typename T>
  Optional<Integer> ExtractPattern(const T& func) {
    Optional<Integer> op_pat = func->template GetAttr<Integer>("op_pattern");
    return std::move(op_pat);
  }

  bool SupportsTexture(const Array<Attrs>& op_attrs, Integer op_pattern) {
    if (op_pattern.IntValue() < relay::kCommReduce) return true;

    for (auto attr : op_attrs) {
      if (auto conv_attr = attr.as<Conv2DAttrs>()) {
        if (conv_attr->data_layout == "NCHW4c" && conv_attr->kernel_layout == "OIHW4o") {
          return true;
        }
      } else if (auto pool_attrs = attr.as<Pool2DAttrs>()) {
        if (pool_attrs->layout == "NCHW4c") {
          return true;
        }
      } else if (auto avg_attrs = attr.as<AdaptivePool2DAttrs>()) {
        if (avg_attrs->layout == "NCHW4c") {
          return true;
        }
      } else if (attr.as<LayerNormAttrs>()) {
        return true;
      }
    }

    return false;
  }

  std::string Scope(Array<PrimExpr> shape) {
    // currently we support only textures been made from 5d tensors
    // 5d requirement is not limitation of textures in general, it is limitation how
    // we are representing memory scopes/layout and flattening of textures in tir
    if (shape.size() == 5 && shape[4].as<IntImmNode>()->value == 4) {
      std::map<int, std::string> diffs;
      int spatial_limit =
          target_->GetAttr<Integer>("texture_spatial_limit").value_or(Integer(16384))->value;
      int depth_limit =
          target_->GetAttr<Integer>("texture_depth_limit").value_or(Integer(2048))->value;
      int a0 = shape[0].as<IntImmNode>()->value;
      int a1 = shape[1].as<IntImmNode>()->value;
      int a2 = shape[2].as<IntImmNode>()->value;
      int a3 = shape[3].as<IntImmNode>()->value;

      int d1r = a0 * a1;
      int d2r = a2 * a3;
      int d3r = a1 * a2 * a3;
      std::string scope = "global";
      if (a0 < spatial_limit && d3r < spatial_limit)
        scope += ".texture-weight";
      else if (a0 < depth_limit && a1 < spatial_limit && d2r < spatial_limit)
        scope += ".texture-nhwc";
      else if (d1r < depth_limit && a2 < spatial_limit && a3 < spatial_limit)
        scope += ".texture";
      return scope;
    }
    return "global";
  }

  Map<Expr, Map<Expr, Array<String>>> scope_info;
  Map<Expr, Array<String>> call_scope_info;
  Map<Expr, Expr> arg_to_binding;
  IRModule mod_;
  Target target_;
};

class DefineVDevice : ExprMutator {
 public:
  explicit DefineVDevice(const Target& target) : target_(target) {}

  IRModule Run(IRModule& mod) {
    mod_ = mod;
    for (const auto& [gv, func] : mod_->functions) {
      if (func->IsInstance<relax::FunctionNode>()) {
        const auto& base_func = mod_->Lookup(gv);
        // Only non primitive relax functions
        if (base_func->HasNonzeroAttr(attr::kPrimitive)) {
          continue;
        }
        auto info = CollectConsumerScopeInfo().Collect(mod_, Downcast<Function>(func), target_);
        call_scope_info_ = info.first;
        scope_info_ = info.second;
        producer_sinfo_ = CollectProduserScopeInfo().Collect(mod_, Downcast<Function>(func),
                                                             scope_info_, target_);
        relax::Function update_func = Downcast<Function>(VisitExpr(func));
        updates_->Add(gv, update_func);
      }
    }
    mod_.CopyOnWrite()->Update(updates_);

    Array<GlobalInfo> global_vdevices_;
    for (auto vdev : vdevices_) {
      global_vdevices_.push_back(vdev.as<GlobalInfo>().value());
    }
    mod_.CopyOnWrite()->global_infos.Set("vdevice", global_vdevices_);

    mod_ = relax::transform::SpecializeTIRParams()(mod_);
    mod_ = relax::transform::DeadCodeElimination()(mod_);
    mod_ = relax::transform::RealizeVDevice()(mod_);
    mod_ = relax::transform::SpecializeTIRParams()(mod_);

    return mod_;
  }

  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const CallNode* call_node) override {
    auto call = Downcast<Call>(ExprMutator::VisitExpr_(call_node));
    static const Op& call_tir_op = Op::Get("relax.call_tir");

    GlobalVar gv;
    Tuple func_args;

    StructInfo out_sinfo;

    if (call->op == call_tir_op) {
      gv = Downcast<GlobalVar>(call->args[0]);
      tir::PrimFunc pfunc = Downcast<tir::PrimFunc>(mod_->Lookup(gv));
      out_sinfo = call->sinfo_args[0];
      func_args = Downcast<Tuple>(call->args[1]);
    } else {
      return call;
    }

    Array<Expr> new_args;
    StructInfo updated_ret_sinfo = producer_sinfo_[GetRef<Expr>(call_node)];

    int arg_idx = 0;
    for (auto arg : func_args->fields) {
      auto sinfo = GetStructInfo(arg);
      if (auto tensor_sinfo = sinfo.as<TensorStructInfo>()) {
        String scope = "global";
        if (call_scope_info_.find(GetRef<Expr>(call_node)) != call_scope_info_.end()) {
          scope = call_scope_info_[GetRef<Expr>(call_node)][arg_idx];
        }
        new_args.push_back(HintArg(arg, scope));
        arg_idx++;
      } else {
        new_args.push_back(arg);
      }
    }

    if (call->op == call_tir_op) {
      auto updated_call =
          Call(call_tir_op, {gv, Tuple(new_args)}, call->attrs, {updated_ret_sinfo});
      return builder_->Normalize(updated_call);
    } else {
      auto updated_call = Call(call->op, new_args, call->attrs, {updated_ret_sinfo});
      return builder_->Normalize(updated_call);
    }
  }

 private:
  void AppendToVDevices(VDevice vdev) {
    int device_type = vdev->target->GetTargetDeviceType();
    for (auto vdevice : vdevices_) {
      int dev_type = vdevice->target->GetTargetDeviceType();
      if (dev_type == device_type && vdevice->vdevice_id == vdev->vdevice_id &&
          vdevice->memory_scope == vdev->memory_scope) {
        return;
      }
    }
    vdevices_.push_back(vdev);
    return;
  }

  Expr HintArg(const Expr& arg, const String& scope) {
    if (arg->IsInstance<ConstantNode>()) {
      if (auto tsinfo = arg->struct_info_.as<TensorStructInfoNode>()) {
        if (!tsinfo->vdevice.defined()) {
          VDevice vdev = VDevice(target_, 0, scope);
          CHECK(tsinfo->shape.defined()) << "Shape not defined for a constant tensor ..!";
          arg->struct_info_ =
              TensorStructInfo(tsinfo->shape.value(), tsinfo->dtype, vdev, tsinfo->span);
          return arg;
        }
      }
    }
    ObjectPtr<HintOnDeviceAttrs> attrs = make_object<HintOnDeviceAttrs>();
    attrs->dev_type = target_->GetTargetDeviceType();
    attrs->dev_id = 0;
    attrs->memory_scope = scope;

    Expr new_arg = Call(hint_on_device_op_, {arg}, Attrs{std::move(attrs)}, {});

    AppendToVDevices(VDevice(target_, 0, scope));
    return std::move(new_arg);
  }

  Optional<Target> GetTarget(const StructInfo& sinfo) {
    auto tinfo = sinfo.as<TensorStructInfoNode>();
    if (tinfo->vdevice.defined()) {
      auto vdevice = tinfo->vdevice.value();
      if (vdevice->target.defined()) {
        return vdevice->target;
      }
    }
    return NullOpt;
  }

  const Op& hint_on_device_op_ = Op::Get("relax.hint_on_device");
  IRModule mod_;
  IRModule updates_;
  Target target_;
  Array<VDevice> vdevices_;
  Map<Expr, Map<Expr, Array<String>>> scope_info_;
  Map<Expr, StructInfo> producer_sinfo_;
  Map<Expr, Array<String>> call_scope_info_;
};

namespace transform {

Pass AnnotateCustomMemoryScope(Target target) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return relax::DefineVDevice(target).Run(mod); };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"AnnotateCustomMemoryScope",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.AnnotateCustomMemoryScope")
    .set_body_typed(AnnotateCustomMemoryScope);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
