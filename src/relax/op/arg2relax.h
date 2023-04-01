#ifndef TVM_RELAX_OP_ARG2RELAX_H_
#define TVM_RELAX_OP_ARG2RELAX_H_

#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/struct_info.h>
#include <tvm/topi/array_api/base.h>

#include "./op_common.h"

namespace tvm {
namespace relax {
namespace arg2relax {

using Converter = PackedFunc;

#define TVM_RELAX_OP_ARG_TRIVIAL_CONVERTER(FuncName, TypeFrom, FnConvert) \
  inline Converter FuncName() {                                           \
    return runtime::TypedPackedFunc<relax::Expr(const TypeFrom& e)>(      \
        [](const TypeFrom& e) -> relax::Expr { return FnConvert(e); });   \
  }

#define TVM_RELAX_OP_ARG_CHECK(Checker, Arg, Result)                                     \
  try {                                                                                  \
    (Result).push_back(Checker);                                                         \
  } catch (const ::tvm::Error& e) {                                                      \
    LOG(FATAL) << "ValueError: Error creating operator " << __func__ << " in argument #" \
               << (Result).size() << " (" << #Arg << "): " << e.what();                  \
  }

inline void VariadicArgs(Array<relax::Expr>* args, uint32_t min_args) {
  while (args->size() > min_args && !args->back().defined()) {
    args->pop_back();
  }
}

// Array

template <class ElemType>
inline Array<ElemType> _ArrayLen(const Array<ElemType>& obj, const std::vector<int>& length) {
  if (!length.empty()) {
    if (length.size() == 1 && obj.size() == 1) {
      return Array<ElemType>(length[0], obj[0]);
    } else if (std::find(length.begin(), length.end(), obj.size()) == length.end()) {
      LOG(FATAL) << "TypeError: Expect length to be one of "
                 << topi::array_api::_StringifyIntVector(length) << ", but got " << obj.size();
    }
  }
  return obj;
}

template <class TElem>
inline Array<TElem> _ArrayApply(const Array<ObjectRef>& obj, const Converter& elem) {
  Array<TElem> ret;
  ret.reserve(obj.size());
  if (elem != nullptr) {
    for (const ObjectRef& o : obj) {
      ret.push_back(elem(o));
    }
  } else {
    for (const ObjectRef& o : obj) {
      ret.push_back(Downcast<TElem>(o));
    }
  }
  return ret;
}

inline Converter ArrayToTuple(Converter elem, std::vector<int> length) {
  return runtime::TypedPackedFunc<relax::Expr(Array<ObjectRef>)>(
      [elem, length](Array<ObjectRef> args) -> relax::Tuple {
        return Tuple(_ArrayApply<relax::Expr>(_ArrayLen(args, length), elem));
      });
}

inline Converter ArrayToShapeExpr(Converter elem, std::vector<int> length) {
  return runtime::TypedPackedFunc<relax::Expr(Array<ObjectRef>)>(
      [elem, length](Array<ObjectRef> args) -> relax::ShapeExpr {
        return ShapeExpr(_ArrayApply<PrimExpr>(_ArrayLen(args, length), elem));
      });
}

inline Converter ArrayToOpaque(Converter elem, std::vector<int> length) {
  return runtime::TypedPackedFunc<relax::Expr(Array<ObjectRef>)>(
      [elem, length](Array<ObjectRef> args) -> relax::AttrExpr {
        return AttrExpr(_ArrayApply<ObjectRef>(_ArrayLen(args, length), elem));
      });
}

// Optional

inline Converter OptionalToOpaque(Converter elem) {
  return runtime::TypedPackedFunc<relax::Expr(ObjectRef)>([elem](ObjectRef arg) -> relax::AttrExpr {
    if (arg.defined()) {
      if (elem != nullptr) {
        ObjectRef ret = elem(arg);
        return AttrExpr(ret);
      } else {
        return AttrExpr(Downcast<relax::Expr>(arg));
      }
    } else {
      return AttrExpr(ObjectRef(nullptr));
    }
  });
}

// Int, Float, Bool
inline Converter ScalarToPrimValue(bool is_bool) {
  return runtime ::TypedPackedFunc<relax::Expr(const PrimExpr& e)>(
      [is_bool](const PrimExpr& e) -> relax::Expr {
        if (const auto* int_imm = e.as<IntImmNode>()) {
          if (is_bool) {
            return relax::PrimValue::Bool(int_imm->value);
          } else {
            return relax::PrimValue::Int64(int_imm->value);
          }
        } else if (const auto* float_imm = e.as<FloatImmNode>()) {
          return relax::PrimValue::Float32(float_imm->value);
        } else {
          LOG(FATAL) << "TypeError: Expect scalar constants, but got " << e;
        }
      });
};

// PrimExpr, IntPrimExpr, FloatPrimExpr, BoolPrimExpr
// TIRVar, IntTIRVar, FloatTIRVar, BoolTIRVar
TVM_RELAX_OP_ARG_TRIVIAL_CONVERTER(PrimExprToPrimValue, PrimExpr, relax::PrimValue);

// Trivial: Tensor, IntTensor, FloatTensor, BoolTensor
// Trivial: AnyRelaxExpr
// Trivial: TupleExpr

// Str
TVM_RELAX_OP_ARG_TRIVIAL_CONVERTER(Str, String, relax::StringImm);

// DType
TVM_RELAX_OP_ARG_TRIVIAL_CONVERTER(DType, DataType, relax::DataTypeImm);

// Trivial: Shape

// Axis
inline Converter Axis() {
  return runtime ::TypedPackedFunc<relax ::Expr(const Integer& e)>(
      [](const Integer& e) -> relax ::Expr { return relax::PrimValue::Int64(e->value); });
};

inline Converter Axes() {
  return runtime::TypedPackedFunc<relax::Expr(Array<ObjectRef>)>(
      [](Optional<Array<ObjectRef>> args) -> relax::AttrExpr {
        if (!args.defined()) {
          return AttrExpr(ObjectRef(nullptr));
        }
        Array<PrimExpr> axes;
        axes.reserve(args.value().size());
        for (const ObjectRef& o : args.value()) {
          axes.push_back(Downcast<PrimExpr>(o));
        }
        return AttrExpr(axes);
      });
}

// Trivial: GlobalVar
// Trivial: ExternFunc

// IndexMap
TVM_RELAX_OP_ARG_TRIVIAL_CONVERTER(IndexMapToOpaque, tir::IndexMap, relax::AttrExpr);

}  // namespace arg2relax
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_ARG2RELAX_H_
