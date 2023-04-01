#include "./te_integration.h"

#include <tvm/relax/struct_info.h>
#include <tvm/te/operation.h>
#include <tvm/tir/stmt_functor.h>

#include "../../te/operation/create_primfunc.h"

namespace tvm {
namespace relax {

static Expr call_tir(Expr func, Tuple args, StructInfo out_sinfo, Optional<Expr> packed_ints) {
  static const Op& op = Op::Get("relax.call_tir");
  Call call;
  if (!packed_ints) {
    // don't use additional optional argument
    call = Call(op, {func, args}, {}, {out_sinfo});
  } else {
    call = Call(op, {func, args, packed_ints.value()}, {}, {out_sinfo});
  }
  return call;
}

ObjectRef AsObjectRef(const runtime::TVMRetValue& rv) {
  switch (rv.type_code()) {
    case kTVMArgInt:
      return IntImm(DataType::Int(64), rv.operator int64_t());
    case kTVMArgFloat:
      return FloatImm(DataType::Float(64), rv.operator double());
    case kTVMNullptr:
      return ObjectRef(nullptr);
    case kTVMDLTensorHandle:
      return rv.operator runtime::NDArray();
    case kTVMObjectHandle:
    case kTVMModuleHandle:
    case kTVMPackedFuncHandle:
      return rv.operator ObjectRef();
    case kTVMStr:
      return String(rv.operator std::string());
    case kTVMNDArrayHandle:
      return rv.operator runtime::NDArray();
    default:
      LOG(FATAL) << "Unknown type code " << rv.type_code();
      return ObjectRef(nullptr);
  }
}

Array<PrimExpr> GetShape(const relax::Var& var);

Array<PrimExpr> GetShape(const ShapeStructInfo& shape) {
  CHECK(!shape->IsUnknownNdim() && shape->values.defined());
  return shape->values.value();
}

Array<PrimExpr> GetShape(const TensorStructInfo& tensor) {
  CHECK(!tensor->IsUnknownNdim() && !tensor->IsUnknownDtype() && tensor->shape.defined());
  if (const auto* shape = tensor->shape.as<ShapeExprNode>()) {
    return shape->values;
  } else if (const auto* shape_var = tensor->shape.as<VarNode>()) {
    return GetShape(GetRef<relax::Var>(shape_var));
  }
  LOG(FATAL) << "TypeError: Unsupported TensorStructInfo::shape: " << tensor->shape;
}

Array<PrimExpr> GetShape(const relax::Var& var) {
  if (const auto* tensor = var->struct_info_.as<TensorStructInfoNode>()) {
    return GetShape(GetRef<TensorStructInfo>(tensor));
  }
  if (const auto* shape = var->struct_info_.as<ShapeStructInfoNode>()) {
    return GetShape(GetRef<ShapeStructInfo>(shape));
  }
  LOG(FATAL) << "TypeError: Unsupported Var::struct_info_: " << var->struct_info_;
}

runtime::TVMRetValue RelaxToTE(const relax::Expr& relax_obj,
                               const std::function<void(const ObjectRef&)> f_mark_as_input,
                               const std::function<PrimExpr(const PrimExpr&)>& f_remap_tir) {
  runtime::TVMRetValue rv;
  if (const auto* prim_value = relax_obj.as<PrimValueNode>()) {
    if (const auto* int_value = prim_value->value.as<IntImmNode>()) {
      rv = int_value->value;
    } else if (const auto* float_value = prim_value->value.as<FloatImmNode>()) {
      rv = float_value->value;
    } else {
      LOG(FATAL) << "Unsupported PrimValue: " << prim_value->value;
    }
  } else if (relax_obj->IsInstance<PrimExprNode>()) {
    LOG(FATAL) << "TypeError: PrimExpr is not supported: " << relax_obj;
  } else if (const auto* string_imm = relax_obj.as<relax::StringImmNode>()) {
    rv = string_imm->value;
  } else if (const auto* dtype_imm = relax_obj.as<relax::DataTypeImmNode>()) {
    rv = dtype_imm->value;
  } else if (const auto* opaque = relax_obj.as<relax::OpaqueObjectNode>()) {
    rv = opaque->value;
  } else if (const auto* tuple = relax_obj.as<relax::TupleNode>()) {
    Array<ObjectRef> fields;
    fields.reserve(tuple->fields.size());
    for (const auto& field : tuple->fields) {
      fields.push_back(AsObjectRef(RelaxToTE(field, f_mark_as_input, f_remap_tir)));
    }
    rv = fields;
  } else if (const auto* shape_expr = relax_obj.as<relax::ShapeExprNode>()) {
    rv = shape_expr->values;
  } else if (const auto* constant = relax_obj.as<relax::ConstantNode>()) {
    rv = constant->data;
  } else if (const auto* var = relax_obj.as<relax::VarNode>()) {
    if (const auto* tensor = relax_obj.as<TensorStructInfoNode>()) {
      Array<PrimExpr> shape = GetShape(GetRef<TensorStructInfo>(tensor));
      if (f_remap_tir) {
        int ndim = shape.size();
        for (int i = 0; i < ndim; ++i) {
          shape.Set(i, f_remap_tir(shape[i]));
        }
      }
      DataType dtype = tensor->dtype;
      ICHECK(!dtype.is_void());
      te::Tensor te_tensor = te::placeholder(shape, dtype);
      f_mark_as_input(te_tensor);
      rv = te_tensor;
    } else if (const auto* shape_info = relax_obj.as<ShapeStructInfoNode>()) {
      Array<PrimExpr> shape = GetShape(GetRef<ShapeStructInfo>(shape_info));
      if (f_remap_tir) {
        int ndim = shape.size();
        for (int i = 0; i < ndim; ++i) {
          shape.Set(i, f_remap_tir(shape[i]));
        }
      }
      rv = shape;
    } else {
      LOG(FATAL) << "Unsupported Var::struct_info_: " << var->struct_info_;
    }
  }
  return rv;
}

class TEInputHandler {
 public:
  void operator()(const ObjectRef& obj) {
    std::string name;
    if (cnt < 26) {
      name = std::string(1, 'A' + cnt);
    } else {
      name = "T" + std::to_string(cnt);
    }
    ++cnt;
    if (const auto* tensor = obj.as<te::TensorNode>()) {
      const_cast<te::OperationNode*>(tensor->op.operator->())->name = name;
      inputs.push_back(GetRef<te::Tensor>(tensor));
    } else {
      LOG(FATAL) << "Unsupported object as the input: " << obj;
    }
  }

  int cnt = 0;
  std::vector<te::Tensor> inputs;
};

class TEVarRemapper {
 public:
  PrimExpr operator()(const PrimExpr& e) {
    return tir::Substitute(e, [this](const tir::Var& old_var) -> Optional<PrimExpr> {
      if (old2new.count(old_var.get())) {
        return GetRef<PrimExpr>(old2new.at(old_var.get()));
      }
      tir::Var new_var = old_var.copy_with_suffix("");
      old_vars.push_back(old_var);
      new_vars.push_back(new_var);
      old2new[old_var.get()] = new_var.get();
      new2old[new_var.get()] = old_var.get();
      return new_var;
    });
  }

  PrimExpr Inverse(const PrimExpr& e) {
    return tir::Substitute(e, [this](const tir::Var& new_var) -> Optional<PrimExpr> {
      if (new2old.count(new_var.get())) {
        return GetRef<PrimExpr>(new2old.at(new_var.get()));
      }
      return NullOpt;
    });
  }

  std::vector<tir::Var> old_vars;
  std::vector<tir::Var> new_vars;
  std::unordered_map<const tir::VarNode*, const tir::VarNode*> old2new;
  std::unordered_map<const tir::VarNode*, const tir::VarNode*> new2old;
};

StructInfo TEToStructInfo(const ObjectRef& obj) {
  if (const auto* tensor = obj.as<te::TensorNode>()) {
    return TensorStructInfo(ShapeExpr(tensor->shape), tensor->dtype);
  } else if (const auto* array = obj.as<ArrayNode>()) {
    Array<StructInfo> results;
    results.reserve(array->size());
    for (const auto& field : *array) {
      results.push_back(TEToStructInfo(field));
    }
    return TupleStructInfo(results);
  } else {
    LOG(FATAL) << "Unsupported ObjectRef: " << obj;
  }
}

FInferStructInfo InferStructInfoFromTE(std::string global_func_name) {
  const auto* te_func = runtime::Registry::Get(global_func_name);
  return [te_func](const Call& call, const BlockBuilder& ctx) -> StructInfo {
    using namespace tvm::runtime;
    int n_args = call->args.size();
    std::vector<TVMRetValue> inputs(n_args, TVMRetValue());
    std::vector<TVMValue> values(n_args, TVMValue());
    std::vector<int> codes(n_args, 0);
    TVMArgsSetter setter(values.data(), codes.data());
    TEInputHandler input_handler;
    for (int i = 0; i < n_args; ++i) {
      inputs[i] = RelaxToTE(
          /*relax_obj=*/call->args[i],
          /*f_mark_as_input=*/input_handler,
          /*f_remap_tir=*/nullptr);
      setter(i, inputs[i]);
    }
    tvm::runtime::TVMArgs args(values.data(), codes.data(), n_args);
    tvm::runtime::TVMRetValue rv;
    te_func->CallPacked(args, &rv);
    return TEToStructInfo(rv);
  };
}

std::vector<te::Tensor> FlattenNestedTETensor(const ObjectRef& obj) {
  if (const auto* tensor = obj.as<te::TensorNode>()) {
    return {GetRef<te::Tensor>(tensor)};
  } else if (const auto* array = obj.as<ArrayNode>()) {
    std::vector<te::Tensor> results;
    for (const auto& field : *array) {
      std::vector<te::Tensor> nested = FlattenNestedTETensor(field);
      results.insert(results.end(), nested.begin(), nested.end());
    }
    return results;
  } else {
    LOG(FATAL) << "Unsupported ObjectRef: " << obj;
  }
}

FLegalize LegalizeFromTE(std::string global_func_name, std::string primfunc_name_hint) {
  const auto* te_func = runtime::Registry::Get(global_func_name);
  return [te_func, primfunc_name_hint = std::move(primfunc_name_hint)](const BlockBuilder& bb,
                                                                       const Call& call) -> Expr {
    using namespace tvm::runtime;
    int n_args = call->args.size();
    std::vector<TVMRetValue> inputs(n_args, TVMRetValue());
    std::vector<TVMValue> values(n_args, TVMValue());
    std::vector<int> codes(n_args, 0);
    TVMArgsSetter setter(values.data(), codes.data());
    TEInputHandler input_handler;
    TEVarRemapper remapper;
    for (int i = 0; i < n_args; ++i) {
      inputs[i] = RelaxToTE(
          /*relax_obj=*/call->args[i],
          /*f_mark_as_input=*/input_handler,
          /*f_remap_tir=*/remapper);
      setter(i, inputs[i]);
    }
    tvm::runtime::TVMArgs args(values.data(), codes.data(), n_args);
    tvm::runtime::TVMRetValue rv;
    te_func->CallPacked(args, &rv);
    std::vector<te::Tensor> te_outputs = FlattenNestedTETensor(rv);
    Array<te::Tensor> arg_list;
    Array<tir::Var> tir_var_list;
    arg_list.insert(arg_list.end(), input_handler.inputs.begin(), input_handler.inputs.end());
    arg_list.insert(arg_list.end(), te_outputs.begin(), te_outputs.end());
    tir_var_list.insert(tir_var_list.end(), remapper.new_vars.begin(), remapper.new_vars.end());
    tir::PrimFunc prim_func = tir::CreatePrimFunc(arg_list, tir_var_list, DataType::Int(64));
    GlobalVar gv = bb->AddFunction(prim_func, primfunc_name_hint);
    return call_tir(gv, Tuple(call->args), call->sinfo_args[0], NullOpt);  // TODO: add tir_var_list
  };
}

}  // namespace relax
}  // namespace tvm
