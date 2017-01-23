/*!
 *  Copyright (c) 2016 by Contributors
 * \file ir.h
 * \brief Additional high level nodes in the IR
 */
#ifndef TVM_IR_H_
#define TVM_IR_H_

#include <ir/Expr.h>
#include <ir/IR.h>
#include <type_traits>
#include <string>
#include "./base.h"
#include "./expr.h"

namespace tvm {
namespace ir {

using Halide::Internal::ExprNode;
using Halide::Internal::StmtNode;
using Halide::Internal::IRNodeType;
using Halide::Internal::ForType;
using Halide::DeviceAPI;

/*! \brief Reduction operator operator */
struct Reduce : public ExprNode<Reduce> {
  /*!
   * \brief The binary operator of reduction
   */
  std::string op;
  /*! \brief The source operand */
  Expr source;
  /*! \brief The reduction domains */
  Array<IterVar> rdom;

  /*! \brief construct expr from op and rdom */
  static Expr make(std::string op, Expr src, Array<IterVar> rdom);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("op", &op);
    v->Visit("source", &source);
    v->Visit("rdom", &rdom);
  }
  static const IRNodeType _type_info = IRNodeType::ExtensionExpr;
  static constexpr const char* _type_key = "Reduce";
  static constexpr const char* Add = "Add";
  static constexpr const char* Max = "Max";
  static constexpr const char* Min = "Min";
};

/*! \brief namespace of TVM Intrinsic functions */
namespace intrinsic {
// Most of the intrinsics is to enab
/*!
 * \brief See pesudo code
 *
 *  Type tvm_api_load_arg(TVMArg* args, int* args_type_id, i) {
 *     assert(arg_type_id[i] == typeid(Type));
 *     return args[i];
 *  }
 */
constexpr const char* tvm_api_load_arg = "tvm_api_load_arg";
/*!
 * \brief See pesudo code
 *
 *  Type tvm_array_get_field(TVMArray* arr, int field_id) {
 *     return arr->field;
 *  }
 * \sa TVMArrayFieldKind
 */
constexpr const char* tvm_array_get_field = "tvm_array_get_field";
/*!
 * \brief See pesudo code
 *
 *  bool tvm_handle_is_null(void* handle) {
 *     return handle == nullptr
 *  }
 */
constexpr const char* tvm_handle_is_null = "tvm_handle_is_null";
/*!
 * \brief See pesudo code
 *
 *  bool tvm_print(VType value) {
 *     LOG(INFO) << value;
 *  }
 */
constexpr const char* tvm_print = "tvm_print";

/*! \brief The field id of each field in array */
enum TVMArrayFieldKind {
  kData = 0,
  kNDim = 1,
  kShape = 2,
  kStrides = 3,
  kTypeCode = 4,
  kTypeBits = 5,
  kTypeLanes = 6
};
}   // namespace intrinsic

// Reuse IR node defintiion from HalideIR
using Halide::Internal::IntImm;
using Halide::Internal::UIntImm;
using Halide::Internal::FloatImm;
using Halide::Internal::StringImm;
using Halide::Internal::Cast;
using Halide::Internal::Add;
using Halide::Internal::Sub;
using Halide::Internal::Mul;
using Halide::Internal::Div;
using Halide::Internal::Mod;
using Halide::Internal::Min;
using Halide::Internal::Max;
using Halide::Internal::EQ;
using Halide::Internal::NE;
using Halide::Internal::LT;
using Halide::Internal::LE;
using Halide::Internal::GT;
using Halide::Internal::GE;
using Halide::Internal::And;
using Halide::Internal::Or;
using Halide::Internal::Not;
using Halide::Internal::Select;
using Halide::Internal::Load;
using Halide::Internal::Ramp;
using Halide::Internal::Broadcast;
using Halide::Internal::Call;
using Halide::Internal::Let;
using Halide::Internal::LetStmt;
using Halide::Internal::AttrStmt;
using Halide::Internal::AssertStmt;
using Halide::Internal::ProducerConsumer;
using Halide::Internal::For;
using Halide::Internal::Store;
using Halide::Internal::Provide;
using Halide::Internal::Allocate;
using Halide::Internal::Free;
using Halide::Internal::Realize;
using Halide::Internal::Block;
using Halide::Internal::IfThenElse;
using Halide::Internal::Evaluate;

}  // namespace ir
}  // namespace tvm

#endif  // TVM_IR_H_
