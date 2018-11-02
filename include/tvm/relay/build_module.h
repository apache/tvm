/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/relay/build_module.h
 * \brief The passes and data structures needed to build a
 * tvm::Module from a Relay program.
 */
#ifndef TVM_RELAY_BUILD_MODULE_H_
#define TVM_RELAY_BUILD_MODULE_H_

#include <tvm/lowered_func.h>
#include <tvm/relay/module.h>
#include <tvm/relay/expr.h>
#include <string>

namespace tvm {
namespace relay {

/*! \brief A lowered Relay operation.
 *
 * A lowered operation is a pair containing the "primitive" function used
 * to produce the lowered function as well as the lowered function itself.
 */
class LoweredOp;
/*! \brief Call container. */
class LoweredOpNode : public Node {
 public:
  /*!
   * \brief The primitive function to be lowered.
   *
   * A primitive function consists only of calls to relay::Op which
   * can be fused.
   */
  Function func;

  /*!
   * \brief The lowered function.
   */
  LoweredFunc lowered_func;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("func", &func);
    v->Visit("lowered_func", &lowered_func);
  }

  TVM_DLL static LoweredOp make(
      Function func,
      LoweredFunc lowered_func);

  static constexpr const char* _type_key = "relay.LoweredOp";
  TVM_DECLARE_NODE_TYPE_INFO(LoweredOpNode, Node);
};

RELAY_DEFINE_NODE_REF(LoweredOp, LoweredOpNode, NodeRef);

/*!
 * \brief Lower the operations contained in a Relay expression.
 *
 * The lowering pass will only lower functions marked as primitive,
 * the FuseOps pass will provide this behavior, if run before LowerOps.
 *
 * \note This will do a reachability analysis and lower all definitions
 * reachable from the provided expression.
 *
 * \param env  The environment.
 * \param expr The expression with operations to be lowered.
 * \param target The target to lower the functions to.
 *
 * \return The set of lowered operations.
 */
Array<LoweredOp> LowerOps(const Module& env, const Expr& expr,
                          const std::string& target = "llvm");

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BUILD_MODULE_H_
