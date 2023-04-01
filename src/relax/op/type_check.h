#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tir/index_map.h>

#include <functional>

namespace tvm {
namespace relax {

using OpChecker = PackedFunc;

inline bool DTypeAll(runtime::DataType dtype) { return true; }
inline bool DTypeInt(runtime::DataType dtype) { return dtype.is_int(); }
inline bool DTypeFloat(runtime::DataType dtype) { return dtype.is_float(); }
inline bool DTypeBool(runtime::DataType dtype) { return dtype.is_bool(); }

inline Array<relax::Expr> VariadicArgs(const Array<Optional<relax::Expr>>& args, int min_args) {
  int num_args = args.size();
  while (num_args > min_args && !args[num_args - 1].defined()) {
    --num_args;
  }
  Array<relax::Expr> ret;
  ret.reserve(num_args);
  for (int i = 0; i < num_args; ++i) {
    CHECK(args[i].defined()) << "ValueError: args[" << i << "] is nullptr";
    ret.push_back(args[i].value());
  }
  return ret;
}

template <class ElemType>
inline Array<ElemType> CheckArrayLength(const Array<ElemType>& obj,
                                        const std::vector<int>& length) {
  if (length.empty()) {
    return obj;
  }
  if (length.size() == 1 && obj.size() == 1) {
    return Array<ElemType>(length[0], obj[0]);
  }
  if (std::find(length.begin(), length.end(), obj.size()) == length.end()) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < length.size(); ++i) {
      if (i != 0) {
        os << ", ";
      }
      os << length[i];
    }
    os << "]";
    LOG(FATAL) << "TypeError: Expect length to be one of " << os.str() << ", but got "
               << obj.size();
  }
  return obj;
}

inline OpChecker FromArrayToOpaque(OpChecker elem, std::vector<int> length) {
  return runtime::TypedPackedFunc<relax::Expr(Array<ObjectRef>)>(
      [elem = std::move(elem),
       length = std::move(length)](Array<ObjectRef> args) -> relax::OpaqueObject {
        Array<ObjectRef> ret;
        for (const ObjectRef& i : CheckArrayLength<ObjectRef>(args, length)) {
          ret.push_back(elem(i));
        }
        return OpaqueObject(ret);
      });
}

inline OpChecker FromArrayToShapeExpr(OpChecker elem, std::vector<int> length) {
  return runtime::TypedPackedFunc<relax::Expr(Array<PrimExpr>)>(
      [elem = std::move(elem),
       length = std::move(length)](Array<PrimExpr> args) -> relax::ShapeExpr {
        Array<PrimExpr> arg = CheckArrayLength<PrimExpr>(args, length);
        for (const ObjectRef& obj : arg) {
          elem(obj);
        }
        return ShapeExpr(arg);
      });
}

inline OpChecker FromArrayToTuple(OpChecker elem, std::vector<int> length) {
  return runtime::TypedPackedFunc<relax::Expr(Array<ObjectRef>)>(
      [elem = std::move(elem), length = std::move(length)](Array<ObjectRef> args) -> relax::Tuple {
        Array<relax::Expr> ret;
        for (const ObjectRef& i : CheckArrayLength<ObjectRef>(args, length)) {
          ret.push_back(elem(i));
        }
        return Tuple(ret);
      });
}

inline OpChecker FromOptionalToOpaque(OpChecker elem) {
  return runtime::TypedPackedFunc<Optional<relax::Expr>(Optional<ObjectRef>)>(
      [elem = std::move(elem)](Optional<ObjectRef> obj) -> Optional<relax::Expr> {
        if (obj.defined()) {
          return OpaqueObject(elem(obj.value()));
        } else {
          return Optional<relax::Expr>(nullptr);
        }
      });
}

inline OpChecker FromScalarConstant(const std::function<bool(runtime::DataType)>& dtype_checker) {
  return runtime::TypedPackedFunc<relax::Expr(const PrimExpr& e)>(
      [dtype_checker = std::move(dtype_checker)](const PrimExpr& e) -> relax::PrimValue {
        if (!dtype_checker(e->dtype)) {
          LOG(FATAL) << "TypeError: Invalid dtype: " << e->dtype;
        }
        if (const auto* int_imm = e.as<IntImmNode>()) {
          return relax::PrimValue::Int64(int_imm->value);
        } else if (const auto* float_imm = e.as<FloatImmNode>()) {
          return relax::PrimValue::Float64(float_imm->value);
        } else {
          LOG(FATAL) << "TypeError: Expect scalar constants, but got " << e;
        }
      });
}

inline OpChecker FromPrimExpr(const std::function<bool(runtime::DataType)>& dtype_checker) {
  return runtime::TypedPackedFunc<relax::Expr(const PrimExpr& e)>(
      [dtype_checker = std::move(dtype_checker)](const PrimExpr& e) -> relax::PrimValue {
        if (!dtype_checker(e->dtype)) {
          LOG(FATAL) << "TypeError: Invalid dtype: " << e->dtype;
        }
        return relax::PrimValue(e);
      });
}

inline OpChecker FromTIRVar(const std::function<bool(runtime::DataType)>& dtype_checker) {
  return runtime::TypedPackedFunc<relax::Expr(const tir::Var& e)>(
      [dtype_checker = std::move(dtype_checker)](const tir::Var& e) -> relax::Expr {
        if (!dtype_checker(e->dtype)) {
          LOG(FATAL) << "TypeError: Invalid dtype: " << e->dtype;
        }
        return relax::PrimValue(e);
      });
}

inline OpChecker FromTensor(const std::function<bool(runtime::DataType)>& dtype_checker,
                            std::vector<int> ndim) {
  return runtime::TypedPackedFunc<relax::Expr(const relax::Expr& e)>(
      [dtype_checker = std::move(dtype_checker),
       ndim = std::move(ndim)](const relax::Expr& e) -> relax::Expr {
        const auto* tensor_info = e->struct_info_.as<relax::TensorStructInfoNode>();
        CHECK(tensor_info) << "TypeError: Expect Tensor, but got " << e->struct_info_;
        if (!tensor_info->IsUnknownDtype() && !dtype_checker(tensor_info->dtype)) {
          LOG(FATAL) << "TypeError: Invalid dtype: " << tensor_info->dtype;
        }
        if (!tensor_info->IsUnknownNdim() && !ndim.empty() &&
            std::find(ndim.begin(), ndim.end(), tensor_info->ndim) == ndim.end()) {
          std::ostringstream os;
          os << "[";
          for (size_t i = 0; i < ndim.size(); ++i) {
            if (i != 0) {
              os << ", ";
            }
            os << ndim[i];
          }
          os << "]";
          LOG(FATAL) << "TypeError: Expect ndim to be one of " << os.str() << ", but got "
                     << tensor_info->ndim;
        }
        return e;
      });
}

inline OpChecker FromAnyRelaxExpr() {
  return runtime::TypedPackedFunc<relax::Expr(const relax::Expr& e)>(
      [](const relax::Expr& e) -> relax::Expr { return e; });
}

inline OpChecker FromTupleExpr() {
  return runtime::TypedPackedFunc<relax::Expr(const relax::Tuple& e)>(
      [](const relax::Tuple& e) -> relax::Expr { return e; });
}

inline OpChecker FromStr() {
  return runtime::TypedPackedFunc<relax::Expr(const String& s)>(
      [](const String& s) -> relax::StringImm { return relax::StringImm(s); });
}

inline OpChecker FromDType() {
  return runtime::TypedPackedFunc<relax::Expr(const runtime::DataType& dtype)>(
      [](const runtime::DataType& dtype) -> DataTypeImm { return DataTypeImm(dtype); });
}

inline OpChecker FromShape() {
  return runtime::TypedPackedFunc<relax::Expr(const relax::Expr& e)>(
      [](const relax::Expr& e) -> relax::Expr {
        const auto* shape_info = e->struct_info_.as<relax::ShapeStructInfoNode>();
        CHECK(shape_info) << "TypeError: Expect Shape, but got " << e->struct_info_;
        return e;
      });
}

inline OpChecker _FromAxis(int ndim, bool is_insertion, bool normalize) {
  return runtime::TypedPackedFunc<relax::Expr(int64_t)>(
      [ndim, is_insertion, normalize](int64_t axis) -> relax::PrimValue {
        int mod = ndim + (is_insertion ? 1 : 0);
        if (ndim != -1 && normalize) {
          axis %= mod;
          axis = (axis < 0) ? (axis + mod) : axis;
        }
        return PrimValue::Int64(axis);
      });
}

inline OpChecker FromAxis(relax::Expr of, bool is_insertion, bool normalize) {
  int ndim = MatchStructInfo<TensorStructInfo>(of).value()->ndim;
  return _FromAxis(ndim, is_insertion, normalize);
}

inline OpChecker FromAxis(const Array<relax::Expr>& of, bool is_insertion, bool normalize) {
  CHECK(!of.empty()) << "ValueError: Expect at least one tensor";
  int ndim = -1;
  for (const relax::Expr& obj : of) {
    int cur = MatchStructInfo<TensorStructInfo>(obj).value()->ndim;
    if (cur != -1) {
      if (ndim == -1) {
        ndim = cur;
      } else if (ndim != cur) {
        LOG(FATAL) << "ValueError: Expect all tensors to have the same ndim";
      }
    }
  }
  return _FromAxis(ndim, is_insertion, normalize);
}

inline OpChecker _FromAxes(int ndim, bool is_insertion, bool normalize) {
  return runtime::TypedPackedFunc<relax::Expr(Array<IntImm>)>(
      [ndim, is_insertion, normalize](Array<IntImm> axes) -> relax::OpaqueObject {
        int mod = ndim + (is_insertion ? 1 : 0);
        std::vector<int64_t> new_axes;
        for (const IntImm& ax : axes) {
          new_axes.push_back(ax->value);
        }
        if (ndim != -1 && normalize) {
          for (int64_t& axis : new_axes) {
            axis %= mod;
            axis = (axis < 0) ? (axis + mod) : axis;
          }
          std::sort(new_axes.begin(), new_axes.end());
          new_axes.erase(std::unique(new_axes.begin(), new_axes.end()), new_axes.end());
        }
        Array<IntImm> ret;
        for (int64_t ax : new_axes) {
          ret.push_back(IntImm(runtime::DataType::Int(64), ax));
        }
        return OpaqueObject(ret);
      });
}

inline OpChecker FromAxes(relax::Expr of, bool is_insertion, bool normalize) {
  int ndim = MatchStructInfo<TensorStructInfo>(of).value()->ndim;
  return _FromAxes(ndim, is_insertion, normalize);
}

inline OpChecker FromAxes(const Array<relax::Expr>& of, bool is_insertion, bool normalize) {
  CHECK(!of.empty()) << "ValueError: Expect at least one tensor";
  int ndim = -1;
  for (const relax::Expr& obj : of) {
    int cur = MatchStructInfo<TensorStructInfo>(obj).value()->ndim;
    if (cur != -1) {
      if (ndim == -1) {
        ndim = cur;
      } else if (ndim != cur) {
        LOG(FATAL) << "ValueError: Expect all tensors to have the same ndim";
      }
    }
  }
  return _FromAxes(ndim, is_insertion, normalize);
}

inline OpChecker FromGlobalVar() {
  return runtime::TypedPackedFunc<relax::Expr(const GlobalVar& e)>(
      [](const GlobalVar& e) -> GlobalVar { return e; });
}

inline OpChecker FromExternFunc() {
  return runtime::TypedPackedFunc<relax::Expr(const ExternFunc& e)>(
      [](const ExternFunc& e) -> ExternFunc { return e; });
}

inline OpChecker FromIndexMap() {
  return runtime::TypedPackedFunc<relax::Expr(const tir::IndexMap& e)>(
      [](const tir::IndexMap& e) -> relax::OpaqueObject { return relax::OpaqueObject(e); });
}

inline OpChecker FromStructInfo() {
  return runtime::TypedPackedFunc<relax::StructInfo(const StructInfo& e)>(
      [](const StructInfo& e) -> relax::StructInfo { return e; });
}

}  // namespace relax
}  // namespace tvm
