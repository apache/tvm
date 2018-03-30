/*!
 *  Copyright (c) 2016 by Contributors
 * \file expr.h
 * \brief The Expr and related elements in DataFlow construction.
 */
#ifndef TVM_EXPR_H_
#define TVM_EXPR_H_

#include <ir/Expr.h>
#include <ir/IROperator.h>
#include <ir/IRPrinter.h>
#include <string>
#include <algorithm>
#include "./base.h"
#include "./runtime/c_runtime_api.h"

namespace tvm {

using HalideIR::Type;
using HalideIR::Float;
using HalideIR::Bool;
using HalideIR::Int;
using HalideIR::UInt;
using HalideIR::Handle;
using HalideIR::ExprHash;
using HalideIR::ExprEqual;

using HalideIR::Expr;
using HalideIR::VarExpr;
using HalideIR::IR::RangeNode;
using HalideIR::IR::FunctionRef;
using HalideIR::IR::FunctionBaseNode;
using HalideIR::Internal::Stmt;
using HalideIR::Internal::IRPrinter;
using HalideIR::Internal::Variable;

using HalideIR::Internal::make_const;
using HalideIR::Internal::make_zero;
using HalideIR::Internal::as_const_int;
using HalideIR::Internal::as_const_uint;
using HalideIR::Internal::const_true;
using HalideIR::Internal::const_false;
using HalideIR::Internal::is_no_op;

inline Type TVMShapeIndexType() {
  if (std::is_signed<tvm_index_t>::value) {
    return Int(sizeof(tvm_index_t) * 8);
  } else {
    return UInt(sizeof(tvm_index_t) * 8);
  }
}

inline Type TVMType2Type(TVMType t) {
  return Type(static_cast<halideir_type_code_t>(t.code), t.bits, t.lanes);
}

inline TVMType Type2TVMType(Type t) {
  TVMType ret;
  ret.code = static_cast<uint8_t>(t.code());
  ret.bits = static_cast<uint8_t>(t.bits());
  ret.lanes = static_cast<uint16_t>(t.lanes());
  return ret;
}

// Get number of bytes considering vector type.
inline int GetVectorBytes(Type dtype) {
  int data_bits = dtype.bits() * dtype.lanes();
  CHECK_EQ(data_bits % 8, 0U)
      << "Need to load/store by multiple of bytes";
  return data_bits / 8;
}

/*! \brief a named variable in TVM */
class Var : public HalideIR::VarExpr {
 public:
  EXPORT explicit Var(const std::string& name_hint = "v",
               Type t = Int(32)) : VarExpr(name_hint, t) {}
  explicit Var(std::shared_ptr<Node> n) : VarExpr(n) {}
  explicit Var(VarExpr v) : VarExpr(v) {}
  /*!
   * \brief Make a new copy of var with same type, append suffix
   * \param suffix The suffix to be appended.
   * \return the new Var copy
   */
  Var copy_with_suffix(const std::string& suffix) const {
    return Var((*this)->name_hint + suffix, (*this)->type);
  }
  /*! \brief type indicate the container type */
  using ContainerType = Variable;
};


/*! \brief container class of iteration variable. */
class IterVarNode;

/*!
 * \brief same as HalideIR::IR::Range
 *  except it provide an constructor with (begin, end)
 *
 *  \note Traditional Halide's Range have a constructor with
 *   (begin, extent), which does not match the convention in e.g. python.
 *   We decided to correct it by removing the constructor in HalideIR,
 *   and add it back in TVM's range.
 */
class Range : public HalideIR::IR::Range {
 public:
  /*! \brief constructor */
  Range() {}
  explicit Range(std::shared_ptr<Node> n) : HalideIR::IR::Range(n) {}
  /*!
   * \brief constructor by begin and end
   * \param begin The begin of the range.
   * \param end The end of the range.
   */
  TVM_DLL Range(Expr begin, Expr end);

  TVM_DLL static Range make_by_min_extent(Expr min, Expr extent);
};

/*!
 * \brief Type of iteration variable.
 *  Each IterVar have a specific type.
 *
 *  The type of iter var can be overriden via
 *  stage.iter_var_attrs given they are compatible.
 */
enum IterVarType : int {
  /*!
   * \brief Data parallel iteration.
   *  This normally corresponds to axis of Tensor.
   *  Allow all IterVar manipulations.
   *
   * \note This does not mean the loop
   *  have to be executed in parallel fashion.
   */
  kDataPar = 0,
  /*!
   * \brief The IterVar itself is a thread-index
   *  of a fixed thread launching group.
   *  Note that this is already assumed to be paralellized.
   *
   *  Disallow: split/fuse/vectorize/parallel
   */
  kThreadIndex = 1,
  /*!
   * \brief Communicative reduction.
   *  Cannot be directly parallelized.
   *
   *  Disallow: parallel/vectorize
   */
  kCommReduce = 2,
  /*!
   * \brief Serial loops with loop carry dependency,
   *  the iteration must execute in order.
   *  Cannot be re-ordered.
   *
   *  Disallow: reorder/parallel/vectorize
   */
  kOrdered = 3,
  /*!
   * \brief IterVar is opaque,
   *
   *  May not corresponds to any generated loop
   *  Disallow all IterVar manipulations and compute_at
   *
   * \note This is usually used to implement composite op
   *  or external op, where the
   */
  kOpaque = 4,
  // The following are possible additional
  // types that are provided during schedule
  /*!
   * \brief The execution is unrolled.
   */
  kUnrolled = 5,
  /*!
   * \brief The loop is vectorized.
   */
  kVectorized = 6,
  /*!
   * \brief The loop is parallelized.
   */
  kParallelized = 7,
  /*!
   * \brief Marks boundary of tensorization intrinsic.
   */
  kTensorized = 8
};

/*!
 * \brief Iteration Variable,
 *  represents an iteration over an integer interval.
 */
class IterVar : public NodeRef {
 public:
  // construct a new iter var without a domain
  IterVar() {}
  // construct from shared ptr.
  explicit IterVar(std::shared_ptr<Node> n) : NodeRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const IterVarNode* operator->() const;
  /*!
   * \return the corresponding var in the IterVar.
   */
  inline operator Expr() const;
  /*! \brief specify container node */
  using ContainerType = IterVarNode;
};

/*!
 * \brief Create a new IterVar that represents an axis in thread.
 *
 * \param dom Optional, domain of the thread axis.
 * \param tag The thread tag of the axis.
 */
TVM_DLL IterVar thread_axis(Range dom, std::string tag);

/*!
 * \brief Create a new IterVar for reduction operations.
 *
 * \param dom The domain of the reduction axis.
 * \param name The name of the reduction axis.
 */
TVM_DLL IterVar reduce_axis(Range dom, std::string name = "rv");

using Domain = Array<Range>;

// print functions for expr
TVM_DLL std::ostream& operator<<(std::ostream& os, const NodeRef& n);  // NOLINT(*)
// definition of Node.
/*!
 * \brief An iteration variable representing an iteration
 *  over a one dimensional interval.
 */
class IterVarNode : public Node {
 public:
  /*!
   * \brief the domain of iteration, if known, can be None
   *  For the intermediate schedule node, before schedule.
   */
  Range dom;
  /*! \brief The looping variable */
  Var var;
  /*! \brief The type of the IterVar */
  IterVarType iter_type;
  /*!
   * \brief additional tag on the iteration variable,
   *  set this if this is binded already to a known thread tag.
   */
  std::string thread_tag;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("dom", &dom);
    v->Visit("var", &var);
    v->Visit("iter_type", &iter_type);
    v->Visit("thread_tag", &thread_tag);
  }

  TVM_DLL static IterVar make(Range dom, Var var,
                              IterVarType iter_type,
                              std::string thread_tag = "");

  static constexpr const char* _type_key = "IterVar";
  TVM_DECLARE_NODE_TYPE_INFO(IterVarNode, Node);
};

// inline implementations
inline const IterVarNode* IterVar::operator->() const {
  return static_cast<const IterVarNode*>(node_.get());
}

inline IterVar::operator Expr() const {
  return (*this)->var;
}

inline const char* IterVarType2String(IterVarType t) {
  switch (t) {
    case kDataPar: return "DataPar";
    case kThreadIndex: return "ThreadIndex";
    case kCommReduce: return "CommReduce";
    case kOrdered: return "Ordered";
    case kOpaque: return "Opaque";
    case kUnrolled: return "Unrolled";
    case kVectorized: return "Vectorized";
    case kParallelized: return "Parallelized";
    case kTensorized: return "Tensorized";
  }
  return "Unknown";
}

/*!
 * \brief Construct a new Var expression
 * \param name_hint The name hint for the expression
 * \param t The type of the expression
 */
TVM_DLL Var var(const std::string& name_hint, Type t = Int(32));

/*
 * \brief Template function to convert Map to unordered_map
 *  Sometimes useful for API gluing when internal uses unordered_map
 * \param dmap The container map
 * \return The corresponding unordered_map.
 * \tparam K the key of the Map.
 * \tparam V the value of the Map.
 */
template<typename K, typename V>
inline std::unordered_map<K, V> as_unordered_map(const Map<K, V>& dmap) {
  std::unordered_map<K, V> ret;
  for (auto kv : dmap) {
    ret[kv.first] = kv.second;
  }
  return ret;
}
}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::IterVar> {
  std::size_t operator()(const ::tvm::IterVar& k) const {
    return k.hash();
  }
};
}
#endif  // TVM_EXPR_H_
