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

using HalideIR::Internal::ExprNode;
using HalideIR::Internal::StmtNode;
using HalideIR::Internal::IRNodeType;
using HalideIR::Internal::ForType;
using HalideIR::DeviceAPI;

// Node container for CommReducer
struct CommReducerNode;

struct CommReducer : public NodeRef {
  CommReducer() {}
  explicit CommReducer(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const CommReducerNode* get() const;
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const CommReducerNode* operator->() const;
  /*! \brief type indicate the container type */
  using ContainerType = CommReducerNode;
};

/*!
 * \brief A commutative reducer node to represent a commutative
 *  binary operator with identity element
 */
struct CommReducerNode : public Node {
  /*! \brief The left argument of reducer */
  Array<Var> lhs;
  /*! \brief The right argument of reducer */
  Array<Var> rhs;
  /*! \brief The result of reducer */
  Array<Expr> result;
  /*!
   * \brief The identity element of reducer, which leaves other
   *  elements unchanged when combined with it, with respect to
   *  the binary operation of this reducer uses.
   */
  Array<Expr> identity_element;
  /*! \brief Function call operator to combine a and b */
  Array<Expr> operator()(Array<Expr> a, Array<Expr> b) const;
  /*! \brief construct CommReducer from args, result and identity_element */
  TVM_DLL static CommReducer make(Array<Var> lhs, Array<Var> rhs,
                                 Array<Expr> result, Array<Expr> identity_element);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("lhs", &lhs);
    v->Visit("rhs", &rhs);
    v->Visit("result", &result);
    v->Visit("identity_element", &identity_element);
  }

  static constexpr const char* _type_key = "CommReducer";
  TVM_DECLARE_NODE_TYPE_INFO(CommReducerNode, Node);
};

inline const CommReducerNode* CommReducer::get() const {
  return static_cast<CommReducerNode*>(node_.get());
}
inline const CommReducerNode* CommReducer::operator->() const {
  return static_cast<CommReducerNode*>(node_.get());
}

/*! \brief Reduction operator operator */
struct Reduce : public ExprNode<Reduce> {
  /*! \brief The commutative combiner */
  CommReducer combiner;
  /*! \brief The source operand */
  Array<Expr> source;
  /*! \brief The reduction axis */
  Array<IterVar> axis;
  /*!
   * \brief Predicate on the reduction
   *  Only add the body to reduction if condition is true.
   */
  Expr condition;
  /*! \brief the index of this reduce node */
  int value_index;

  /*! \brief construct expr from op and rdom */
  TVM_DLL static Expr make(CommReducer combiner,
                           Array<Expr> src,
                           Array<IterVar> rdom,
                           Expr condition,
                           int value_index);

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dtype", &type);
    v->Visit("combiner", &combiner);
    v->Visit("source", &source);
    v->Visit("axis", &axis);
    v->Visit("condition", &condition);
    v->Visit("value_index", &value_index);
  }
  static const IRNodeType _type_info = IRNodeType::ExtensionExpr;
  static constexpr const char* _type_key = "Reduce";
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

/*! \brief namespace of possible attribute sin AttrStmt.attr_key */
namespace attr {
// The above attr does not pass to ir stage.
/*! \brief Mark launching extent of thread, used by device API. */
constexpr const char* thread_extent = "thread_extent";
/*! \brief Mark launching of a virtual thread. */
constexpr const char* virtual_thread = "virtual_thread";
/*! \brief Mark region is processed by a co-proccesor */
constexpr const char* coproc_scope = "coproc_scope";
/*!
 * \brief Mark region creates coprocessor micro ops,
 *  can be reused if corresponding variable is independent.
 */
constexpr const char* coproc_uop_scope = "coproc_uop_scope";
/*! \brief Mark the scope as volatile access for certain handle. */
constexpr const char* volatile_scope = "volatile_scope";
/*!
 * \brief Mark the scope as when computation start to happen
 *  This can hint some code generator to create a new function for compute.
 */
constexpr const char* compute_scope = "compute_scope";
/*! \brief Mark storage scope of buffers */
constexpr const char* storage_scope = "storage_scope";
/*! \brief Mark storage alignement requirement of buffers */
constexpr const char* storage_alignment = "storage_alignment";
/*! \brief Mark storage scope of realization */
constexpr const char* realize_scope = "realize_scope";
/*! \brief The allocation context for global malloc in host. */
constexpr const char* device_context_id = "device_context_id";
/*! \brief The device type. */
constexpr const char* device_context_type = "device_context_type";
/*! \brief Mark of loop scope */
constexpr const char* loop_scope = "loop_scope";
/*! \brief Mark of reduce scope */
constexpr const char* reduce_scope = "reduce_scope";
/*! \brief Mark region is guarded by the pragma */
constexpr const char* pragma_scope = "pragma_scope";
/*!
 * \brief Mark of prefetch scope, value=offset,
 *  run prefetch of Tensor on the current loop scope
 */
constexpr const char* prefetch_scope = "prefetch_scope";
/*!
 * \brief Marks production of double buffer data
 */
constexpr const char* double_buffer_scope = "double_buffer_scope";
/*!
 * \brief Marks region used by double buffer write
 */
constexpr const char* double_buffer_write = "double_buffer_write";
/*! \brief Mark of scan update scope */
constexpr const char* scan_update_scope = "scan_update_scope";
/*! \brief Mark of scan init scope */
constexpr const char* scan_init_scope = "scan_init_scope";
/*!
 * \brief Mark alignment of buffer dimension
 *  stmt.node is Tensor
 *  stmt.value is tvm_tuple(dim, align, offset)
 *  This gives hint to require stride of dim to be k * align + offset.
 */
constexpr const char* buffer_dim_align = "buffer_dim_align";
/*!
 * \brief Bind the buffer specification to the region of the op
 *  When this scope occurs, the stmt.node is a Array<NodeRef> = [buffer, tensor]
 *  stmt.value is a tvm_tuple(min0, extent0, min1, extent1, ...).
 *  The scope represents that we need to bind the storage region of tensor to buffer.
 *  This will affect replacement of some variables inside the scope that
 *  corresponds to field of buffer to be the actual expressions of tensor during
 *  storage flattening phase.
 */
constexpr const char* buffer_bind_scope = "buffer_bind_scope";
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
/*!
 * \brief See pesudo code
 *
 *  Handle tvm_address_of(Load *op) {
 *     return &op->buffer_var[index];
 *  }
 */
constexpr const char* tvm_address_of = "tvm_address_of";
/*!
 * \brief Same as select, used for unsafe memory access.
 *
 *  Type tvm_if_then_else(cond, a, b) {
 *    return cond ? a : b;
 *  }
 */
constexpr const char* tvm_if_then_else = "tvm_if_then_else";
/*!
 * \brief Get head access address with memory access pattern info.
 *
 *  This operator also marks range of the memory access
 *  The offset and extent are in unit of the DType(including vectorization factor).
 *  rw_mask is a bit_mask setting whether the access is a read(1) or write(2).
 *  The access is assume to happen in the current expression.
 *
 *  PtrType tvm_access_ptr(Expr dtype, DType* data,
 *                         int offset, int extent,
 *                         int rw_mask) {
 *    // DType == dtype.type();
 *    return &data[offset];
 *  }
 */
constexpr const char* tvm_access_ptr = "tvm_access_ptr";
/*!
 * \brief Create a function local static handle that iniitalizes to nullptr.
 *  can be used to cache function local static resources.
 */
constexpr const char* tvm_static_handle = "tvm_static_handle";
/*!
 * \brief Return a unique context id, used for hint of workspace separation.
 *  Different context id ganrantees not having overlapping workspace.
 */
constexpr const char* tvm_context_id = "tvm_context_id";
/*!
 * \brief tvm_tuple is not an actual function and cannot codegen.
 *  It is used to represent tuple structure in value field of AttrStmt,
 *  for the sake of giving hint to optimization.
 *
 *  Handle tvm_tuple(value0, value1, ..., value_n);
 */
constexpr const char* tvm_tuple = "tvm_tuple";
/*!
 * \brief See pesudo code
 *
 *  Type tvm_struct_get(StructType* arr, int index, int field_id) {
 *     return arr[index]->field;
 *  }
 * \sa TVMStructFieldKind
 */
constexpr const char* tvm_struct_get = "tvm_struct_get";
/*!
 * \brief See pesudo code
 *
 *  Handle tvm_struct_set(StructType* arr, int index, int field_id, value) {
 *     arr[index]->field = value;
 *  }
 * \sa TVMStructFieldKind
 */
constexpr const char* tvm_struct_set = "tvm_struct_set";
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
 *  void tvm_throw_last_error() {
 *    throw TVMGetLastError();
 *  }
 */
constexpr const char* tvm_throw_last_error = "tvm_throw_last_error";
/*!
 * \brief See pesudo code
 *
 *  dtype in {shape, array, arg_value, arg_tcode}
 *
 *  Handle tvm_stack_alloca(string dtype, int num) {
 *     return new on stack dtype[num];
 *  }
 */
constexpr const char* tvm_stack_alloca = "tvm_stack_alloca";
/*!
 * \brief Allocate a shape tuple on stack, return the handle.
 *
 *  Handle tvm_stack_make_shape(list args) {
 *     ret = alloca stack int64_t[len(args)];
 *     for i in range(len(args)):
 *        ret[i] = args[i]
 *     return &ret[0];
 *  }
 */
constexpr const char* tvm_stack_make_shape = "tvm_stack_make_shape";
/*!
 * \brief Allocate a NDArray(DLTensor) on stack, return the handle.
 *
 *  Type tvm_stack_make_array(Expr data,
 *                            Expr shape,
 *                            Expr strides,
 *                            Expr ndim,
 *                            Expr dtype,
 *                            Expr elem_offset) {
 *     ret = alloca stack DLTensor();
 *     ret->data = data;
 *     ret->shape = shape;
 *     ret->strides = strides != 0 ? strides : nullptr;
 *     ret->ndim = ndim;
 *     ret->dtype = dtype.type();
 *     ret->byte_offset = elem_offset * sizeof(dtype);
 *     return ret;
 *  }
 */
constexpr const char* tvm_stack_make_array = "tvm_stack_make_array";
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
 *  Mark the content as thread local context, can get optimized
 *  by only call the call once at thread start.
 *
 *  Do not allow nesting(getting a thread context from another).
 *
 *  Handle tvm_thread_context(Expr call) {
 *     return call;
 *  }
 */
constexpr const char* tvm_thread_context = "tvm_thread_context";
/*!
 * \brief Lowered version of call packed, the space of value and
 *  type codes are explicitly allocated.
 *
 *  int tvm_call_packed_lowered(name,
 *                              TVMValue* value_stack,
 *                              int* tcode_stack,
 *                              int begin,
 *                              int end) {
 *     ModuleNode* env = GetCurrentEnv();
 *     const PackedFunc* f = env->GetFuncFromEnv(name);
 *     f->CallPacked(TVMArgs(value_stack[begin:end],
 *                           tcode_stack[begin:end]),
 *                   TVMRetValue(value_stack + end, tcode_stack + end));
 *  }
 */
constexpr const char* tvm_call_packed_lowered = "tvm_call_packed_lowered";
/*!
 * \brief See pseudo code
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
 *  void tvm_thread_allreduce(UIntImm size, Expr source0, ..., Expr cond,
 *                            Var reduce_temp0, .., Var thread_idx1, ...) {
 *     // constraint by the other thread_idx remain the same.
 *     // reduce_temp is used to save intermediate result.
 *     reduce_temp0, ... = reduce(combiner, source0, ..., cond
 *       over [thread_idx1, thread_idx2] passed by any caller)
 *  }
 */
constexpr const char* tvm_thread_allreduce = "tvm_thread_allreduce";

/*! \brief The kind of structure field info */
enum TVMStructFieldKind : int {
  // array head address
  kArrAddr,
  kArrData,
  kArrShape,
  kArrStrides,
  kArrNDim,
  kArrTypeCode,
  kArrTypeBits,
  kArrTypeLanes,
  kArrByteOffset,
  kArrDeviceId,
  kArrDeviceType,
  kArrKindBound_,
  // TVMValue field
  kTVMValueContent,
  kTVMValueKindBound_
};
}   // namespace intrinsic

// Reuse IR node defintiion from HalideIR
using HalideIR::Internal::IntImm;
using HalideIR::Internal::UIntImm;
using HalideIR::Internal::FloatImm;
using HalideIR::Internal::StringImm;
using HalideIR::Internal::Cast;
using HalideIR::Internal::Add;
using HalideIR::Internal::Sub;
using HalideIR::Internal::Mul;
using HalideIR::Internal::Div;
using HalideIR::Internal::Mod;
using HalideIR::Internal::Min;
using HalideIR::Internal::Max;
using HalideIR::Internal::EQ;
using HalideIR::Internal::NE;
using HalideIR::Internal::LT;
using HalideIR::Internal::LE;
using HalideIR::Internal::GT;
using HalideIR::Internal::GE;
using HalideIR::Internal::And;
using HalideIR::Internal::Or;
using HalideIR::Internal::Not;
using HalideIR::Internal::Select;
using HalideIR::Internal::Load;
using HalideIR::Internal::Ramp;
using HalideIR::Internal::Broadcast;
using HalideIR::Internal::Call;
using HalideIR::Internal::Let;
using HalideIR::Internal::LetStmt;
using HalideIR::Internal::AttrStmt;
using HalideIR::Internal::AssertStmt;
using HalideIR::Internal::ProducerConsumer;
using HalideIR::Internal::For;
using HalideIR::Internal::Store;
using HalideIR::Internal::Provide;
using HalideIR::Internal::Allocate;
using HalideIR::Internal::Free;
using HalideIR::Internal::Realize;
using HalideIR::Internal::Prefetch;
using HalideIR::Internal::Block;
using HalideIR::Internal::IfThenElse;
using HalideIR::Internal::Evaluate;
using HalideIR::Internal::Shuffle;
// ir functions
using HalideIR::Internal::is_const_power_of_two_integer;

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
