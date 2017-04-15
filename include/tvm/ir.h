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
  /*! \brief The reduction axis */
  Array<IterVar> axis;
  /*!
   * \brief Predicate on the reduction
   *  Only add the body to reduction if condition is true.
   */
  Expr condition;

  /*! \brief construct expr from op and rdom */
  static Expr make(std::string op, Expr src,
                   Array<IterVar> rdom,
                   Expr condition = const_true());
  /*!
   * \brief Get initial value for reduction.
   * \param op The operator
   * \param type The data type.
   * \return The initial value that can be assigned to reduction.
   */
  static Expr InitValue(const std::string& op, Type type);
  /*!
   * \brief Combine two values with given reduction.
   * \param op The operator
   * \param a The left operand.
   * \param b The left operand.
   * \return The combined reduction result.
   */
  static Expr Combine(const std::string& op, Expr a, Expr b);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("op", &op);
    v->Visit("source", &source);
    v->Visit("axis", &axis);
    v->Visit("condition", &condition);
  }
  static const IRNodeType _type_info = IRNodeType::ExtensionExpr;
  static constexpr const char* _type_key = "Reduce";
  static constexpr const char* Add = "Add";
  static constexpr const char* Max = "Max";
  static constexpr const char* Min = "Min";
};

/*!
 * \brief Auxiliary data structure used in IR Pass to indicate a tensor.
 */
struct TensorKey {
  FunctionRef f;
  int value_index;

  inline bool operator==(const TensorKey& other) const {
    return f == other.f && value_index == other.value_index;
  }
  inline std::string GetName() const {
    if (f->num_outputs() == 1) return f->func_name();
    std::ostringstream os;
    os << f->func_name() << ".v" << value_index;
    return os.str();
  }
};

/*! \brief namespace of possible attribute sin AttrStmt.type_key */
namespace attr {
// The above attr does not pass to ir stage.
/*!
 * \brief Mark launching extent of thread, used by device API.
 */
constexpr const char* thread_extent = "thread_extent";
/*!
 * \brief Mark launching of a virtual thread.
 */
constexpr const char* virtual_thread = "virtual_thread";
/*!
 * \brief Mark the scope as volatile access for certain handle.
 */
constexpr const char* volatile_scope = "volatile_scope";
/*!
 * \brief Mark storage scope of buffers
 */
constexpr const char* storage_scope = "storage_scope";
/*! \brief Mark storage scope of realization */
constexpr const char* realize_scope = "realize_scope";
/*! \brief Mark of loop scope */
constexpr const char* loop_scope = "loop_scope";
/*! \brief Mark of scan update scope */
constexpr const char* scan_update_scope = "scan_update_scope";
/*! \brief Mark of scan init scope */
constexpr const char* scan_init_scope = "scan_init_scope";
/*! \brief extern operator scope */
constexpr const char* extern_op_scope = "extern_op_scope";
// Pipeline related attributes
/*! \brief channel read scope */
constexpr const char* channel_read_scope = "channel_read_scope";
/*! \brief Advance step of channel after end of scope */
constexpr const char* channel_read_advance = "channel_read_advance";
/*! \brief channel write scope */
constexpr const char* channel_write_scope = "channel_write_scope";
/*! \brief Advance step of channel after end of scope */
constexpr const char* channel_write_advance = "channel_write_advance";
/*! \brief pipeline stage scope, implies always execution */
constexpr const char* pipeline_stage_scope = "pipeline_stage_scope";
/*! \brief pipeline execution scope, implies the scope can be pipelined. */
constexpr const char* pipeline_exec_scope = "pipeline_exec_scope";
}  // namespace attr

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
 *  int tvm_call_packed(name, TVMValue* args) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     (*f)(args, type_code_of(args), len(args));
 *     return 0;
 *  }
 */
constexpr const char* tvm_call_packed = "tvm_call_packed";
/*!
 * \brief See pesudo code
 *
 *  int tvm_storage_sync(std::string storage_scope) {
 *     __sync(storage_scope);
 *     return 0;
 *  }
 */
constexpr const char* tvm_storage_sync = "tvm_storage_sync";
/*!
 * \brief Initialize the global barrier.
 *  Call this at beginning of kernel that need global barrier.
 */
constexpr const char* tvm_global_barrier_kinit = "tvm_global_barrier_kinit";
/*!
 * \brief See pesudo code
 *
 *  Expr tvm_thread_allreduce(std::string op, Expr value, Expr cond,
 *                             Var thread_idx1, thread_idx2...) {
 *     // constraint by the other thread_idx remain the same.
 *     return reduce(op, value, cond,
 *                   over [thread_idx1, thread_idx2] passed by any caller)
 *  }
 */
constexpr const char* tvm_thread_allreduce = "tvm_thread_allreduce";

/*! \brief The field id of each field in array */
enum TVMArrayFieldKind {
  kData = 0,
  kNDim = 1,
  kShape = 2,
  kStrides = 3,
  kTypeCode = 4,
  kTypeBits = 5,
  kTypeLanes = 6,
  kByteOffset = 7
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
// ir functions
using Halide::Internal::is_const_power_of_two_integer;

}  // namespace ir
}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::ir::TensorKey> {
  std::size_t operator()(const ::tvm::ir::TensorKey& k) const {
    size_t lhs = k.f.hash();
    size_t rhs = static_cast<size_t>(k.value_index);
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
    return lhs;
  }
};
}  // namespace std

#endif  // TVM_IR_H_
