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
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/distributed/struct_info.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace printer {

class AttrPrinter : public tvm::AttrVisitor {
 public:
  explicit AttrPrinter(ObjectPath p, const IRDocsifier& d, Array<String>* keys,
                       Array<ExprDoc>* values)
      : p(std::move(p)), d(d), keys(keys), values(values) {}

  void Visit(const char* key, double* value) final {
    keys->push_back(key);
    values->push_back(LiteralDoc::Float(*value, p->Attr(key)));
  }

  void Visit(const char* key, int64_t* value) final {
    keys->push_back(key);
    values->push_back(LiteralDoc::Int(*value, p->Attr(key)));
  }

  void Visit(const char* key, uint64_t* value) final {
    keys->push_back(key);
    values->push_back(LiteralDoc::Int(*value, p->Attr(key)));
  }

  void Visit(const char* key, int* value) final {
    keys->push_back(key);
    values->push_back(LiteralDoc::Int(*value, p->Attr(key)));
  }

  void Visit(const char* key, bool* value) final {
    keys->push_back(key);
    values->push_back(LiteralDoc::Boolean(*value, p->Attr(key)));
  }

  void Visit(const char* key, std::string* value) final {
    keys->push_back(key);
    values->push_back(LiteralDoc::Str(*value, p->Attr(key)));
  }

  void Visit(const char* key, DataType* value) final {
    keys->push_back(key);
    values->push_back(LiteralDoc::DataType(*value, p->Attr(key)));
  }

  void Visit(const char* key, runtime::ObjectRef* value) final {
    keys->push_back(key);
    values->push_back(d->AsDoc<ExprDoc>(*value, p->Attr(key)));
  }

  void Visit(const char* key, void** value) final {
    LOG(FATAL) << "TypeError: void is not allowed in Attrs";
  }

  void Visit(const char* key, runtime::NDArray* value) final {
    LOG(FATAL) << "TypeError: NDArray is not allowed in Attrs";
  }

  ObjectPath p;
  const IRDocsifier& d;
  Array<String>* keys;
  Array<ExprDoc>* values;
};

ExprDoc PrintCallee(const relax::Expr& n, const ObjectPath& n_p, const IRDocsifier& d) {
  // TODO(@junrushao): handle callee better
  if (const auto* ext = n.as<relax::ExternFuncNode>()) {
    return LiteralDoc::Str(ext->global_symbol, n_p);
  } else {
    return d->AsDoc<ExprDoc>(n, n_p);
  }
}

Optional<ExprDoc> PrintCallTIRDPSPacked(const relax::Call& n, const ObjectPath& n_p,
                                        const IRDocsifier& d) {
  static const Op& call_tir_op = Op::Get("relax.call_tir");
  static const Op& call_tir_inplace_op = Op::Get("relax.call_tir_inplace");
  static const Op& call_dps_packed_op = Op::Get("relax.call_dps_packed");
  static const Op& call_tir_with_grad_op = Op::Get("relax.call_tir_with_grad");
  static const Op& call_tir_local_view = Op::Get("relax.dist.call_tir_local_view");
  if (!n->op.same_as(call_tir_op) && !n->op.same_as(call_dps_packed_op) &&
      !n->op.same_as(call_tir_with_grad_op) && !n->op.same_as(call_tir_local_view) &&
      !n->op.same_as(call_tir_inplace_op)) {
    return NullOpt;
  }
  ICHECK(n->args.size() == 2 || n->args.size() == 3);
  ICHECK(n->sinfo_args.size() == 1);
  Array<ExprDoc> args;
  Array<String> kwargs_keys;
  Array<ExprDoc> kwargs_values;
  // Step 1. Print n->args[0], the callee
  args.push_back(PrintCallee(n->args[0], n_p->Attr("args")->ArrayIndex(0), d));
  // Step 2. Print n->args[1], the input arguments
  args.push_back(d->AsDoc<ExprDoc>(n->args[1], n_p->Attr("args")->ArrayIndex(1)));
  // Step 3. Print n->sinfo_args, the output struct info
  relax::StructInfo o_sinfo = n->sinfo_args[0];
  ObjectPath o_sinfo_p = n_p->Attr("sinfo_args")->ArrayIndex(0);
  bool is_dtensor = false;
  kwargs_keys.push_back("out_sinfo");
  if (const auto* o = o_sinfo.as<relax::TupleStructInfoNode>()) {
    Array<ExprDoc> fields;
    ObjectPath fields_p = o_sinfo_p->Attr("fields");
    for (int i = 0, l = o->fields.size(); i < l; ++i) {
      if (o->fields[i].as<relax::distributed::DTensorStructInfoNode>()) {
        is_dtensor = true;
      }
      fields.push_back(d->AsDoc<ExprDoc>(o->fields[i], fields_p->ArrayIndex(i)));
    }
    kwargs_values.push_back(ListDoc(fields));
  } else {
    if (o_sinfo.as<relax::distributed::DTensorStructInfoNode>()) {
      is_dtensor = true;
    }
    kwargs_values.push_back(d->AsDoc<ExprDoc>(o_sinfo, o_sinfo_p));
  }

  // for call_tir_inplace, we also need to include the inplace args
  if (n->op.same_as(call_tir_inplace_op)) {
    kwargs_keys.push_back("inplace_indices");
    Array<ExprDoc> index_fields;
    if (auto* call_tir_inplace_attrs = n->attrs.as<relax::CallTIRInplaceAttrs>()) {
      for (auto inplace_index : call_tir_inplace_attrs->inplace_indices) {
        index_fields.push_back(
            LiteralDoc::Int(inplace_index.IntValue(), n_p->Attr("attrs")->Attr("inplace_indices")));
      }
    }
    kwargs_values.push_back(ListDoc(index_fields));
  }

  // start of specially handling call_tir_with_grad
  if (const auto* call_tir_with_grad_attrs = n->attrs.as<relax::CallTIRWithGradAttrs>()) {
    kwargs_keys.push_back("te_grad_name");
    kwargs_values.push_back(LiteralDoc::Str(call_tir_with_grad_attrs->te_grad_name,
                                            n_p->Attr("attrs")->Attr("te_grad_name")));
    if (!call_tir_with_grad_attrs->te_grad_kwargs.empty()) {
      kwargs_keys.push_back("te_grad_kwargs");
      kwargs_values.push_back(d->AsDoc<ExprDoc>(call_tir_with_grad_attrs->te_grad_kwargs,
                                                n_p->Attr("attrs")->Attr("te_grad_kwargs")));
    }
  }
  if (n->op.same_as(call_tir_with_grad_op)) {
    return Relax(d, "call_tir_with_grad")->Call(args, kwargs_keys, kwargs_values);
  }
  // end of specially handling call_tir_with_grad

  if (n->op.same_as(call_dps_packed_op)) {
    return Relax(d, "call_dps_packed")->Call(args, kwargs_keys, kwargs_values);
  }
  // Step 4. Print n->args[2], the tir variables
  if (n->args.size() == 3) {
    kwargs_keys.push_back("tir_vars");
    kwargs_values.push_back(d->AsDoc<ExprDoc>(n->args[2], n_p->Attr("args")->ArrayIndex(2)));
  }
  if (n->op.same_as(call_tir_local_view)) {
    return Relax(d, "dist.call_tir_local_view")->Call(args, kwargs_keys, kwargs_values);
  } else if (is_dtensor) {
    return Relax(d, "dist.call_tir")->Call(args, kwargs_keys, kwargs_values);
  } else if (n->op.same_as(call_tir_inplace_op)) {
    return Relax(d, "call_tir_inplace")->Call(args, kwargs_keys, kwargs_values);
  } else {
    return Relax(d, "call_tir")->Call(args, kwargs_keys, kwargs_values);
  }
}

Optional<ExprDoc> PrintAssertOp(const relax::Call& n, const ObjectPath& n_p, const IRDocsifier& d) {
  static const Op& assert_op = Op::Get("relax.assert_op");
  if (!n->op.same_as(assert_op)) {
    return NullOpt;
  }
  ICHECK(n->args.size() >= 2);
  // special handling: it is important to indicate that the format string (second argument)
  // is the _format_ string, or else roundtripping will fail
  // (the format string will be interpreted as an argument and there will be a new default format
  // string given)
  Array<ExprDoc> args;
  args.push_back(d->AsDoc<ExprDoc>(n->args[0], n_p->Attr("args")->ArrayIndex(0)));
  ExprDoc second_arg = d->AsDoc<ExprDoc>(n->args[1], n_p->Attr("args")->ArrayIndex(1));
  for (size_t i = 2; i < n->args.size(); i++) {
    args.push_back(d->AsDoc<ExprDoc>(n->args[i], n_p->Attr("args")->ArrayIndex(i)));
  }
  return Relax(d, "assert_op")->Call(args, {"format"}, {second_arg});
}

Optional<ExprDoc> PrintHintOnDevice(const relax::Call& n, const ObjectPath& n_p,
                                    const IRDocsifier& d) {
  static const Op& hint_on_device_op = Op::Get("relax.hint_on_device");
  if (!n->op.same_as(hint_on_device_op)) {
    return NullOpt;
  }
  Array<ExprDoc> args;

  args.push_back(PrintCallee(n->args[0], n_p->Attr("args")->ArrayIndex(0), d));
  Array<String> kwargs_keys;
  Array<ExprDoc> kwargs_values;
  ICHECK(n->attrs.defined());
  if (n->attrs.as<relax::HintOnDeviceAttrs>()) {
    AttrPrinter printer(n_p->Attr("attrs"), d, &kwargs_keys, &kwargs_values);
    const_cast<BaseAttrsNode*>(n->attrs.get())->VisitAttrs(&printer);
    args.push_back(Relax(d, "device")->Call({}, kwargs_keys, kwargs_values));
  }
  return Relax(d, "hint_on_device")->Call(args);
}

Optional<ExprDoc> PrintToVDevice(const relax::Call& n, const ObjectPath& n_p,
                                 const IRDocsifier& d) {
  static const Op& to_vdevice_op = Op::Get("relax.to_vdevice");
  if (!n->op.same_as(to_vdevice_op)) {
    return NullOpt;
  }
  Array<ExprDoc> args;

  args.push_back(PrintCallee(n->args[0], n_p->Attr("args")->ArrayIndex(0), d));
  Array<String> kwargs_keys;
  Array<ExprDoc> kwargs_values;
  ICHECK(n->attrs.defined());
  if (const auto* attrs = n->attrs.as<relax::ToVDeviceAttrs>()) {
    VDevice vdev = attrs->dst_vdevice;
    std::string dev_kind = vdev->target->kind->name;
    int dev_index = FindVDeviceIndexByTargetKind(vdev, d);
    kwargs_keys.push_back("dst_vdevice");
    kwargs_values.push_back(
        LiteralDoc::Str(dev_kind + ":" + std::to_string(dev_index), n_p->Attr("dst_vdevice")));
  }
  return Relax(d, "to_vdevice")->Call(args, kwargs_keys, kwargs_values);
}

Optional<ExprDoc> PrintRelaxPrint(const relax::Call& n, const ObjectPath& n_p,
                                  const IRDocsifier& d) {
  static const Op& print_op = Op::Get("relax.print");
  if (!n->op.same_as(print_op)) {
    return NullOpt;
  }
  ICHECK(n->args.size() >= 1);
  // special handling: it is important to indicate that the format string (first argument)
  // is the _format_ string, or else roundtripping will fail
  // (the format string will be interpreted as an argument and there will be a new default format
  // string given)
  ExprDoc first_arg = d->AsDoc<ExprDoc>(n->args[0], n_p->Attr("args")->ArrayIndex(0));
  Array<ExprDoc> args;
  for (size_t i = 1; i < n->args.size(); i++) {
    args.push_back(d->AsDoc<ExprDoc>(n->args[i], n_p->Attr("args")->ArrayIndex(i)));
  }
  return Relax(d, "print")->Call(args, {"format"}, {first_arg});
}

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch<relax::Call>(  //
        "", [](relax::Call n, ObjectPath n_p, IRDocsifier d) -> Doc {
          // Special case: call_tir, call_dps_packed, call_tir_with_grad
          if (Optional<ExprDoc> doc = PrintCallTIRDPSPacked(n, n_p, d)) {
            return doc.value();
          }
          // Special case: assert_op
          if (Optional<ExprDoc> doc = PrintAssertOp(n, n_p, d)) {
            return doc.value();
          }
          // Special case: hint_on_device
          if (Optional<ExprDoc> doc = PrintHintOnDevice(n, n_p, d)) {
            return doc.value();
          }
          // Special case: to_vdevice
          if (Optional<ExprDoc> doc = PrintToVDevice(n, n_p, d)) {
            return doc.value();
          }
          // Special case: print
          if (Optional<ExprDoc> doc = PrintRelaxPrint(n, n_p, d)) {
            return doc.value();
          }
          ExprDoc prefix{nullptr};
          Array<ExprDoc> args;
          Array<String> kwargs_keys;
          Array<ExprDoc> kwargs_values;
          // Step 1. Print op
          if (const auto* op = n->op.as<relax::ExternFuncNode>()) {
            prefix = Relax(d, "call_packed");
            args.push_back(LiteralDoc::Str(op->global_symbol, n_p->Attr("op")));
          } else if (const auto* op = n->op.as<tvm::OpNode>()) {
            std::string name = op->name;
            if (name.rfind("relax.", 0) == 0) {
              prefix = Relax(d, name.substr(6));
            } else {
              prefix = IdDoc(name);
            }
            prefix->source_paths.push_back(n_p->Attr("op"));
          } else if (n->op->IsInstance<relax::VarNode>() ||
                     n->op->IsInstance<tvm::GlobalVarNode>()) {
            prefix = d->AsDoc<ExprDoc>(n->op, n_p->Attr("op"));
          } else {
            LOG(FATAL) << "TypeError: Unsupported op: " << n->op->GetTypeKey();
          }
          // Step 2. Print args
          if (!n->args.empty()) {
            args.push_back(PrintCallee(n->args[0], n_p->Attr("args")->ArrayIndex(0), d));
          }
          for (int i = 1, l = n->args.size(); i < l; ++i) {
            args.push_back(d->AsDoc<ExprDoc>(n->args[i], n_p->Attr("args")->ArrayIndex(i)));
          }
          // Step 3. Print attrs
          if (n->attrs.defined()) {
            if (n->op->IsInstance<relax::ExternFuncNode>()) {
              kwargs_keys.push_back("attrs_type_key");
              kwargs_values.push_back(LiteralDoc::Str(n->attrs->GetTypeKey(), n_p->Attr("attrs")));
            }
            if (const auto* attrs = n->attrs.as<tvm::DictAttrsNode>()) {
              std::vector<std::pair<String, ObjectRef>> sorted;
              for (const auto& kv : attrs->dict) {
                sorted.push_back(kv);
              }
              std::sort(sorted.begin(), sorted.end());
              for (const auto& kv : sorted) {
                kwargs_keys.push_back(kv.first);
                kwargs_values.push_back(
                    d->AsDoc<ExprDoc>(kv.second, n_p->Attr("attrs")->Attr(kv.first)));
              }
            } else {
              AttrPrinter printer(n_p->Attr("attrs"), d, &kwargs_keys, &kwargs_values);
              const_cast<BaseAttrsNode*>(n->attrs.get())->VisitAttrs(&printer);
            }
          }
          // Step 4. Print type_args
          if (n->sinfo_args.size() > 0) {
            ObjectPath sinfo_args_p = n_p->Attr("sinfo_args");
            Array<ExprDoc> sinfo_args;
            for (int i = 0, l = n->sinfo_args.size(); i < l; ++i) {
              sinfo_args.push_back(
                  d->AsDoc<ExprDoc>(n->sinfo_args[i], sinfo_args_p->ArrayIndex(i)));
            }
            kwargs_keys.push_back("sinfo_args");
            kwargs_values.push_back(TupleDoc(sinfo_args));
          }
          return prefix->Call(args, kwargs_keys, kwargs_values);
        });

TVM_SCRIPT_REPR(relax::CallNode, ReprPrintRelax);

}  // namespace printer
}  // namespace script
}  // namespace tvm
