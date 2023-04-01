#ifndef TVM_RELAX_OP_RELAX2TE_H_
#define TVM_RELAX_OP_RELAX2TE_H_

#include <tvm/ir/function.h>
#include <tvm/relax/op/basic.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/builtin_fp16.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/te/operation.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/topi/array_api/base.h>

#include "../../te/operation/create_primfunc.h"
#include "./op_common.h"

namespace tvm {
namespace relax {
namespace relax2te {

using topi::array_api::NotDerivable;

using Converter = PackedFunc;

#define TVM_RELAX_OP_TE_TRIVIAL_CONVERTER(FuncName, TypeFrom, FnConvert)      \
  inline Converter FuncName() {                                               \
    return runtime::TypedPackedFunc<runtime::TVMRetValue(const TypeFrom& e)>( \
        [](const TypeFrom& e) -> runtime::TVMRetValue {                       \
          runtime::TVMRetValue ret;                                           \
          ret = FnConvert;                                                    \
          return ret;                                                         \
        });                                                                   \
  }

struct DTypeAll {
  inline void operator()(const runtime::DataType& dtype) {}
};

struct DTypeInt {
  inline void operator()(const runtime::DataType& dtype) {
    if (!dtype.is_int() && !dtype.is_uint()) {
      LOG(FATAL) << "TypeError: Expect dtype to be int, but got " << dtype;
    }
  }
};

struct DTypeFloat {
  inline void operator()(const runtime::DataType& dtype) {
    if (!dtype.is_float()) {
      LOG(FATAL) << "TypeError: Expect dtype to be float, but got " << dtype;
    }
  }
};

struct DTypeBool {
  inline void operator()(const runtime::DataType& dtype) {
    if (!dtype.is_bool()) {
      LOG(FATAL) << "TypeError: Expect dtype to be bool, but got " << dtype;
    }
  }
};

// Array

inline Converter ArrayFromOpaque() {
  return runtime::TypedPackedFunc<ObjectRef(const relax::AttrExpr& e)>(
      [](const relax::AttrExpr& e) -> ObjectRef { return e->value; });
}

inline Converter ArrayFromShapeExpr() {
  return runtime::TypedPackedFunc<Array<PrimExpr>(const relax::ShapeExpr& e)>(
      [](const relax::ShapeExpr& e) -> Array<PrimExpr> { return e->values; });
}

inline Converter ArrayFromTupleTensor(std::function<Array<te::Tensor>(const relax::Tuple&)> elem) {
  return runtime::TypedPackedFunc<Array<te::Tensor>(const relax::Tuple& e)>(
      [elem = std::move(elem)](const relax::Tuple& e) -> Array<te::Tensor> { return elem(e); });
}

// Optional

inline Converter OptionalFromOpaque(Converter elem) {
  return runtime::TypedPackedFunc<ObjectRef(const relax::AttrExpr& e)>(
      [elem](const relax::AttrExpr& e) -> ObjectRef {
        if (e->value.defined()) {
          return elem(e->value);
        } else {
          return ObjectRef(nullptr);
        }
      });
}

// Int, Float, Bool
template <class DTypeChecker>
inline Converter ScalarFromPrimValue() {
  return runtime ::TypedPackedFunc<runtime::TVMRetValue(const relax ::PrimValue& e)>(
      [](const relax ::PrimValue& e) -> runtime ::TVMRetValue {
        runtime ::TVMRetValue ret;
        DTypeChecker()(e->value.dtype());
        if (const auto* int_imm = e->value.as<IntImmNode>()) {
          ret = int_imm->value;
        } else if (const auto* float_imm = e->value.as<FloatImmNode>()) {
          ret = float_imm->value;
        } else {
          LOG(FATAL) << "TypeError: Expect PrimValue to be scalar, but got " << e->value;
        }
        return ret;
      });
};

// PrimExpr, IntPrimExpr, FloatPrimExpr, BoolPrimExpr
// TIRVar, IntTIRVar, FloatTIRVar, BoolTIRVar
template <class DTypeChecker>
inline Converter PrimExprFromPrimValue() {
  return runtime ::TypedPackedFunc<PrimExpr(const relax ::PrimValue& e)>(
      [](const relax ::PrimValue& e) -> PrimExpr {
        DTypeChecker()(e->value.dtype());
        return e->value;
      });
};

inline Optional<PrimExpr> ScalarFromUnitNDArray(const runtime::NDArray& from) {
  runtime::NDArray value = from.CopyTo(DLDevice{kDLCPU, 0});
  runtime::DataType dtype(value->dtype);
  if (dtype == DataType::Int(8)) {
    return IntImm(DataType::Int(8), *static_cast<int8_t*>(value->data));
  } else if (dtype == DataType::Int(16)) {
    return IntImm(DataType::Int(16), *static_cast<int16_t*>(value->data));
  } else if (dtype == DataType::Int(32)) {
    return IntImm(DataType::Int(32), *static_cast<int32_t*>(value->data));
  } else if (dtype == DataType::Int(64)) {
    return IntImm(DataType::Int(64), *static_cast<int64_t*>(value->data));
  } else if (dtype == DataType::UInt(8)) {
    return IntImm(DataType::UInt(8), *static_cast<uint8_t*>(value->data));
  } else if (dtype == DataType::UInt(16)) {
    return IntImm(DataType::UInt(16), *static_cast<uint16_t*>(value->data));
  } else if (dtype == DataType::UInt(32)) {
    return IntImm(DataType::UInt(32), *static_cast<uint32_t*>(value->data));
  } else if (dtype == DataType::UInt(64)) {
    return IntImm(DataType::UInt(64), *static_cast<uint64_t*>(value->data));
  } else if (dtype == DataType::Float(16)) {
    return FloatImm(DataType::Float(16), __gnu_h2f_ieee(*static_cast<uint16_t*>(value->data)));
  } else if (dtype == DataType::Float(32)) {
    return FloatImm(DataType::Float(32), *static_cast<float*>(value->data));
  } else if (dtype == DataType::Float(64)) {
    return FloatImm(DataType::Float(64), *static_cast<double*>(value->data));
  } else if (dtype == DataType::Bool()) {
    return Bool(*static_cast<uint8_t*>(value->data));
  }
  return NullOpt;
}

inline Optional<runtime::NDArray> ScalarToUnitNDArray(const PrimExpr& e) {
  runtime::DataType dtype = e->dtype;
  runtime::NDArray value = runtime::NDArray::Empty({}, dtype, DLDevice{kDLCPU, 0});
  void* ptr = value->data;
  if (dtype == DataType::Int(8)) {
    *static_cast<int8_t*>(ptr) = static_cast<int8_t>(e.as<IntImmNode>()->value);
  } else if (dtype == DataType::Int(16)) {
    *static_cast<int16_t*>(ptr) = static_cast<int16_t>(e.as<IntImmNode>()->value);
  } else if (dtype == DataType::Int(32)) {
    *static_cast<int32_t*>(ptr) = static_cast<int32_t>(e.as<IntImmNode>()->value);
  } else if (dtype == DataType::Int(64)) {
    *static_cast<int64_t*>(ptr) = static_cast<int64_t>(e.as<IntImmNode>()->value);
  } else if (dtype == DataType::UInt(8)) {
    *static_cast<uint8_t*>(ptr) = static_cast<uint8_t>(e.as<IntImmNode>()->value);
  } else if (dtype == DataType::UInt(16)) {
    *static_cast<uint16_t*>(ptr) = static_cast<uint16_t>(e.as<IntImmNode>()->value);
  } else if (dtype == DataType::UInt(32)) {
    *static_cast<uint32_t*>(ptr) = static_cast<uint32_t>(e.as<IntImmNode>()->value);
  } else if (dtype == DataType::UInt(64)) {
    *static_cast<uint64_t*>(ptr) = static_cast<uint64_t>(e.as<IntImmNode>()->value);
  } else if (dtype == DataType::Float(16)) {
    *static_cast<uint16_t*>(ptr) = __gnu_f2h_ieee(static_cast<float>(e.as<FloatImmNode>()->value));
  } else if (dtype == DataType::Float(32)) {
    *static_cast<float*>(ptr) = static_cast<float>(e.as<FloatImmNode>()->value);
  } else if (dtype == DataType::Float(64)) {
    *static_cast<double*>(ptr) = static_cast<double>(e.as<FloatImmNode>()->value);
  } else if (dtype == DataType::Bool()) {
    *static_cast<uint8_t*>(ptr) = static_cast<uint8_t>(e.as<IntImmNode>()->value);
  } else {
    return NullOpt;
  }
  return value;
}

// Tensor, IntTensor, FloatTensor, BoolTensor
template <class DTypeChecker, bool try_convert_to_const_scalar = false>
inline Converter TETensorFromRelaxTensor(std::function<te::Tensor(relax::Expr)> elem,
                                         std::vector<int> ndim_candidates) {
  return runtime::TypedPackedFunc<ObjectRef(const relax::Expr& e)>(
      [elem = std::move(elem),
       ndim_candidates = std::move(ndim_candidates)](const relax::Expr& e) -> ObjectRef {  //
        const TensorStructInfoNode* tensor = GetStructInfoAs<TensorStructInfoNode>(e);
        if (tensor == nullptr) {
          LOG(FATAL) << "TypeError: Expect Expr to be Tensor, but got " << e->struct_info_;
        }
        if (try_convert_to_const_scalar && tensor->ndim == 0) {
          if (const auto* a = e.as<relax::ConstantNode>()) {
            if (Optional<PrimExpr> scalar = ScalarFromUnitNDArray(a->data)) {
              return scalar.value();
            }
          }
        }
        DTypeChecker()(tensor->dtype);
        CHECK(ndim_candidates.empty() || std::find(ndim_candidates.begin(), ndim_candidates.end(),
                                                   tensor->ndim) != ndim_candidates.end())
            << "ValueError: Expect ndim to be one of "
            << topi::array_api::_StringifyIntVector(ndim_candidates)
            << ", but got: " << tensor->ndim;
        return elem(e);
      });
}

// Str
TVM_RELAX_OP_TE_TRIVIAL_CONVERTER(Str, relax::StringImm, e->value);

// DType
TVM_RELAX_OP_TE_TRIVIAL_CONVERTER(DType, relax::DataTypeImm, runtime::DLDataType2String(e->value));

// Shape

inline Converter ShapeArrayFromShape() {
  return runtime::TypedPackedFunc<Array<PrimExpr>(const relax::Expr& e)>(
      [](const relax::Expr& e) -> Array<PrimExpr> {
        if (const auto* shape = e.as<relax::ShapeExprNode>()) {
          return shape->values;
        }
        if (auto info = MatchStructInfo<ShapeStructInfo>(e)) {
          if (auto result = info.value()->values) {
            return result.value();
          } else {
            throw NotDerivable("NotDerivable: Cannot deduce Relax shape to Array<PrimExpr>");
          }
        }
        LOG(FATAL) << "TypeError: Expect RelaxExpr to be Shape, but got " << e;
      });
}

// Axis & Axes

inline int _TensorNDim(const te::Tensor& tensor) { return tensor->shape.size(); }

inline int _TupleTensorNDim(const Array<te::Tensor>& tensors) {
  CHECK(!tensors.empty()) << "Expect tensors to be non-empty";
  int ndim = _TensorNDim(tensors[0]);
  for (const auto& tensor : tensors) {
    CHECK_EQ(_TensorNDim(tensor), ndim) << "Expect tensors to have the same ndim";
  }
  return ndim;
}

inline Converter Axis(int ndim, bool is_insertion, bool normalize) {
  return runtime::TypedPackedFunc<IntImm(const relax ::PrimValue& e)>(
      [ndim, is_insertion, normalize](const relax ::PrimValue& e) -> IntImm {
        int axis = Downcast<IntImm>(e->value)->value;
        if (normalize && ndim != kUnknownNDim) {
          int mod = ndim + (is_insertion ? 1 : 0);
          if (mod > 0 && (axis >= mod || axis < -mod)) {
            LOG(FATAL) << "ValueError: Expect axis to be in range [" << -mod << ", " << mod
                       << "), but got: " << axis;
          }
          if (axis < 0) {
            axis += mod;
          }
        }
        return IntImm(DataType::Int(64), axis);
      });
};

inline Converter Axes(int ndim, bool is_insertion, bool normalize) {
  return runtime ::TypedPackedFunc<Optional<Array<IntImm>>(const relax ::AttrExpr& e)>(
      [ndim, is_insertion, normalize](const relax ::AttrExpr& e) -> Optional<Array<IntImm>> {
        if (!e->value.defined()) {
          return NullOpt;
        }
        if (!normalize || ndim == kUnknownNDim) {
          return Downcast<Array<IntImm>>(e->value);
        }
        Array<IntImm> old_axes = Downcast<Array<IntImm>>(e->value);
        Array<IntImm> axes;
        std::unordered_set<int> used_axis;
        int mod = ndim + (is_insertion ? old_axes.size() : 0);
        for (const IntImm& _axis : old_axes) {
          int axis = _axis->value;
          if (axis >= mod || axis < -mod) {
            LOG(FATAL) << "ValueError: Expect axis to be in range [" << -mod << ", " << mod
                       << "), but got: " << old_axes;
          }
          if (axis < 0) {
            axis += mod;
          }
          CHECK(!used_axis.count(axis))
              << "ValueError: Expect axis to be unique, but got: " << old_axes;
          used_axis.insert(axis);
          axes.push_back(IntImm(DataType::Int(64), axis));
        }
        return axes;
      });
};

// Trivial: GlobalVar, ExternFunc

TVM_RELAX_OP_TE_TRIVIAL_CONVERTER(IndexMapFromOpaque, relax::AttrExpr, e->value);
TVM_RELAX_OP_TE_TRIVIAL_CONVERTER(ObjectFromOpaque, relax::AttrExpr, e->value);

inline runtime::TVMRetValue CallGlobalFunc(const runtime::PackedFunc* f,
                                           const std::vector<runtime::TVMRetValue>& args) {
  int num_args = static_cast<int>(args.size());
  std::vector<TVMValue> values(num_args);
  std::vector<int> codes(num_args);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  for (int i = 0; i < num_args; ++i) {
    setter(i, args[i]);
  }
  TVMRetValue rv;
  f->CallPacked(TVMArgs(values.data(), codes.data(), num_args), &rv);
  return rv;
}

class TensorHandler {
 public:
  std::function<te::Tensor(const relax::Expr&)> Tensor(std::string name,
                                                       runtime::DataType default_dtype,
                                                       bool allow_ndim_only) {
    return [this, name, default_dtype, allow_ndim_only](const relax::Expr& e) -> te::Tensor {
      return this->AddInput(e, name, default_dtype, allow_ndim_only);
    };
  }

  std::function<Array<te::Tensor>(const relax::Tuple&)> TupleTensor(std::string name,
                                                                    runtime::DataType default_dtype,
                                                                    bool allow_ndim_only) {
    return
        [this, default_dtype, name, allow_ndim_only](const relax::Tuple& e) -> Array<te::Tensor> {
          int n = e->fields.size();
          Array<te::Tensor> ret;
          ret.reserve(n);
          for (int i = 0; i < n; ++i) {
            ret.push_back(this->AddInput(e->fields[i], name + std::to_string(i), default_dtype,
                                         allow_ndim_only));
          }
          return ret;
        };
  }

  te::Tensor AddInput(const relax::Expr& e, const std::string& name,
                      runtime::DataType default_dtype, bool allow_ndim_only) {
    TensorStructInfo tensor = MatchStructInfo<TensorStructInfo>(e).value();
    // Handle dtype
    runtime::DataType dtype = tensor->dtype;
    if (dtype.is_void()) {
      has_void_dtype = true;
      dtype = default_dtype;
    }
    // Handle shape
    Array<PrimExpr> shape;
    if (Optional<Array<PrimExpr>> tensor_shape = tensor->GetShape()) {
      shape = tensor_shape.value();
    } else if (allow_ndim_only && !tensor->IsUnknownNdim() && !tensor->shape.defined()) {
      shape.reserve(tensor->ndim);
      for (int i = 0; i < tensor->ndim; ++i) {
        tir::Var var(name + "_" + std::to_string(i), DataType::Int(64));
        shape.push_back(var);
        proxy_dims.insert(var.get());
      }
    } else {
      throw NotDerivable("NotDerivable: shape is not defined");
    }
    // Construct te::Tensor
    te::Tensor te_tensor = te::placeholder(shape, dtype, name);
    te_tensors.push_back(te_tensor);
    rx_tensors.push_back(e);
    return te_tensor;
  }

  StructInfo AddOutput(const ObjectRef& obj, DataType out_dtype = DataType::Void()) {
    if (const auto* tensor = obj.as<te::TensorNode>()) {
      te_tensors.push_back(GetRef<te::Tensor>(tensor));
      DataType dtype = out_dtype.is_void()
                           ? (this->has_void_dtype ? DataType::Void() : tensor->dtype)
                           : out_dtype;
      Array<PrimExpr> shape;
      for (const PrimExpr& e : tensor->shape) {
        if (tir::UsesVar(e, [&](const tir::VarNode* v) { return proxy_dims.count(v); })) {
          return TensorStructInfo(dtype, tensor->shape.size());
        } else {
          shape.push_back(e);
        }
      }
      return TensorStructInfo(ShapeExpr(shape), dtype);
    } else if (const auto* array = obj.as<ArrayNode>()) {
      Array<StructInfo> ret;
      ret.reserve(array->size());
      for (const ObjectRef& item : *array) {
        ret.push_back(AddOutput(item, out_dtype));
      }
      return TupleStructInfo(ret);
    }
    LOG(FATAL) << "TypeError: Expect output to be te::Tensor or Array<te::Tensor>, but got "
               << obj->GetTypeKey();
  }

  static Array<tir::Var> RemoveUnused(Array<te::Tensor>* te_tensors,
                                      Array<relax::Expr>* rx_tensors) {
    std::unordered_set<const te::TensorNode*> used;
    std::vector<const te::TensorNode*> queue;
    int head = 0;
    int num_inputs = rx_tensors->size();
    int num_total = te_tensors->size();
    for (int i = num_inputs; i < num_total; ++i) {
      const te::TensorNode* tensor = (*te_tensors)[i].get();
      queue.push_back(tensor);
      used.insert(tensor);
    }
    while (head < static_cast<int>(queue.size())) {
      const te::TensorNode* tensor = queue[head++];
      for (const te::Tensor& t : tensor->op->InputTensors()) {
        if (used.insert(t.get()).second) {
          queue.push_back(t.get());
        }
      }
    }
    for (int i = num_inputs - 1; i >= 0; --i) {
      const te::TensorNode* tensor = (*te_tensors)[i].get();
      if (used.count(tensor) == 0) {
        te_tensors->erase(te_tensors->begin() + i);
        rx_tensors->erase(rx_tensors->begin() + i);
      }
    }
    std::unordered_set<const tir::VarNode*> var_set;
    Array<tir::Var> vars;
    auto f_add_var = [&var_set, &vars](const PrimExpr& expr) {
      tir::PostOrderVisit(expr, [&](const ObjectRef& obj) {
        if (const tir::VarNode* v = obj.as<tir::VarNode>()) {
          if (!var_set.count(v)) {
            var_set.insert(v);
            vars.push_back(GetRef<tir::Var>(v));
          }
        }
      });
    };
    for (const te::Tensor& tensor : *te_tensors) {
      for (const PrimExpr& e : tensor->shape) {
        if (const auto* var = e.as<tir::VarNode>()) {
          var_set.insert(var);
          vars.push_back(GetRef<tir::Var>(var));
        }
      }
    }
    for (const te::TensorNode* tensor : queue) {
      if (const auto* op = tensor->op.as<te::ComputeOpNode>()) {
        for (const tir::IterVar& v : op->axis) {
          var_set.insert(v->var.get());
        }
        for (const tir::IterVar& v : op->reduce_axis) {
          var_set.insert(v->var.get());
        }
      }
    }
    vars.clear();
    for (int i = static_cast<int>(queue.size()) - 1; i >= 0; --i) {
      const te::TensorNode* tensor = queue[i];
      if (tensor->op->IsInstance<te::PlaceholderOpNode>()) {
        continue;
      } else if (const auto* op = tensor->op.as<te::ComputeOpNode>()) {
        for (const PrimExpr& e : op->body) {
          f_add_var(e);
        }
      } else {
        LOG(FATAL) << "TypeError: Unsupported TE operator: " << tensor->op->GetTypeKey();
      }
    }
    return vars;
  }

  relax::Call EmitTE(const BlockBuilder& bb, const std::string& name_hint,
                     const StructInfo& out_sinfo) {
    if (has_void_dtype) {
      throw NotDerivable("NotDerivable: Input tensor has void dtype, which is not allowed");
    }
    if (!proxy_dims.empty()) {
      throw NotDerivable("NotDerivable: Tensor shape unknown, and proxy dims are not allowed");
    }
    Array<tir::Var> unbound_vars = RemoveUnused(&te_tensors, &rx_tensors);
    tir::PrimFunc prim_func = tir::CreatePrimFunc(te_tensors,    //
                                                  unbound_vars,  //
                                                  DataType::Int(64));
    prim_func = WithoutAttr(std::move(prim_func), tvm::attr::kGlobalSymbol);
    prim_func = tir::RenewDefs(std::move(prim_func));
    GlobalVar g_var = bb->AddFunction(std::move(prim_func), name_hint);
    if (!unbound_vars.empty()) {
      ShapeExpr unbound(Array<PrimExpr>{unbound_vars.begin(), unbound_vars.end()});
      return call_tir(g_var, Tuple(rx_tensors), out_sinfo, unbound);
    }
    return call_tir(g_var, Tuple(rx_tensors), out_sinfo, NullOpt);
  }

  void Clear() {
    rx_tensors.clear();
    te_tensors.clear();
    proxy_dims.clear();
    has_void_dtype = false;
  }

  Array<relax::Expr> rx_tensors = {};
  Array<te::Tensor> te_tensors = {};
  std::unordered_set<const tir::VarNode*> proxy_dims = {};
  bool has_void_dtype = false;
};

}  // namespace relax2te
}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_RELAX_TO_TE_H_
