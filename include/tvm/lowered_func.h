/*!
 *  Copyright (c) 2017 by Contributors
 * \file lowered_func.h
 * \brief Information about a lowered TVM function.
 *  This data structure is final step toward codegen.
 */
#ifndef TVM_LOWERED_FUNC_H_
#define TVM_LOWERED_FUNC_H_

#include <tvm/container.h>
#include <ir/FunctionBase.h>
#include <string>

#include "./base.h"
#include "./expr.h"
#include "./tensor.h"

namespace tvm {

// Internal node container of lowered function.
class LoweredFuncNode;

/*!
 * \brief LoweredFunc represents function after lowering.
 *  This is the final IR representation before codegen.
 */
class LoweredFunc : public FunctionRef {
 public:
  LoweredFunc() {}
  explicit LoweredFunc(std::shared_ptr<Node> n) : FunctionRef(n) {}
  /*!
   * \brief access the internal node container
   * \return the pointer to the internal node container
   */
  inline const LoweredFuncNode* operator->() const;
  /*! \brief specify container node */
  using ContainerType = LoweredFuncNode;
};

/*! \brief Node container of LoweredFunc */
class LoweredFuncNode : public FunctionBaseNode {
 public:
  /*! \brief The name of the function */
  std::string name;
  /*!
   * \brief The arguments of the function
   *  This function can only take pod type(int, float) and void* as arguments.
   */
  Array<Var> args;
  /*!
   * \brief The IterVar axis of threads
   *  Each axis need host function to specify a size.
   * \note Calling convention into LoweredFunc
   *
   * Assume we have a LoweredFunc f, a call into f
   *   Call(f, arg1, arg2, ..., arg_n,
   *        size_axis_1, size_axis_2, ... size_axis_m)
   *
   * Here n = len(args), m = len(thread_axis)
   *
   * The CodeGen should take this and translate this call
   * to corresponding API specific kernel launchs or function calls.
   */
  Array<IterVar> thread_axis;
  /*!
   * \brief The hint data type of Var handles defined in LetStmt
   *  Can be used as hint when generating type signiture.
   *  The creation rule is given by
   *  handle_data_type[var_handle] = make_const(the_type, 0);
   *
   * \note Expr is used instead Type, because Type cannot be hold by Map.
   *  constant Expr of given type is used.
   */
  Map<Var, Expr> handle_data_type;
  /*! \brief Whether this function is packed function */
  bool is_packed_func{true};
  /*! \brief The body statment of the function */
  Stmt body;
  /*! \return name of the operation */
  const std::string& func_name() const final {
    return name;
  }
  // there is no return value, but return 1
  // to enable Call into this function.
  int num_outputs() const final {
    return 1;
  }
  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("args", &args);
    v->Visit("thread_axis", &thread_axis);
    v->Visit("handle_data_type", &handle_data_type);
    v->Visit("is_packed_func", &is_packed_func);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "LoweredFunc";
  TVM_DECLARE_NODE_TYPE_INFO(LoweredFuncNode, Node);
};

// Implementations of inline functions
inline const LoweredFuncNode* LoweredFunc::operator->() const {
  return static_cast<const LoweredFuncNode*>(node_.get());
}

}  // namespace tvm

namespace std {
template <>
struct hash<::tvm::LoweredFunc> {
  std::size_t operator()(const ::tvm::LoweredFunc& k) const {
    return k.hash();
  }
};
}

#endif  // TVM_LOWERED_FUNC_H_
