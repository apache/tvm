/*!
 *  Copyright (c) 2017 by Contributors
 * \file arg_binder.h
 * \brief Helper utility to match and bind arguments.
 */
#ifndef TVM_PASS_ARG_BINDER_H_
#define TVM_PASS_ARG_BINDER_H_

#include <tvm/expr.h>
#include <tvm/buffer.h>
#include <string>
#include <vector>

namespace tvm {
namespace ir {

/*!
 * \brief Helper utility to generate match and bind of arguments.
 *
 * \note There is many places in TVM IR where we need argument bindings.
 *
 *  Consider a function f(tA(shape=var(n)), tB(shape=3), tC(shape=(n+2)).
 *  Here n is a undefined variable that is decided by the outside, tB imposes
 *  a constraint such that it can only take tensor with shape 3, tC imposes
 *  another constraint that it's shape must equals n + 2.
 *  So if we call it with f(bufferA, bufferB, bufferC), we need to generate
 *  the following binding sequence:
 *  - define n = bufferA.shape[0]
 *  - assert bufferB.shape[0] == 3
 *  - assert bufferB.shape[1] == n + 3
 *
 *  In general, this is a constraint solving problem. We have simplified assumption
 *  over the binding declaration, such that we require the variable occured in
 *  constraint must be declared in argument list. So it is illegal to have signature
 *  f(tA(shape=(n+3))) without any argument variable corresponds to n, even though
 *  it is already enough to derive n from the input argument.
 */
class ArgBinder {
 public:
  /*!
   * \brief Constructor
   * \param def_map A definition map that contains definition of known variables.
   *   ArgBinder will update this def_map when adding new definitions.
   */
  explicit ArgBinder(
      std::unordered_map<const Variable*, Expr>* def_map)
      : def_map_(def_map) {
  }
  /*!
   * \brief Try to bind arg to value, generate constraint if necessary.
   * \param arg The argument to be binded.
   * \param value The target expression value
   * \param arg_name argument name.
   * \param with_let Whether add lets during bind
   */
  void Bind(const Expr& arg,
            const Expr& value,
            const std::string& arg_name,
            bool with_let = false);
  /*!
   * \brief Bind array to array
   * \param arg The argument to be binded.
   * \param value The target expression value
   * \param arg_name argument name.
   */
  void BindArray(const Array<Expr>& arg,
                 const Array<Expr>& value,
                 const std::string& arg_name);
  /*!
   * \brief Bind symbolic buffer to another symbolic buffer
   * \param arg The argument to be binded.
   * \param value The target expression value
   * \param arg_name argument name.
   * \param fuzzy_match If enabled, we allow value's dimension to be smaller than arg, as long as arg's higher dimensions are of 1.
   */
  void BindBuffer(const Buffer& arg,
                  const Buffer& value,
                  const std::string& arg_name,
                  bool fuzzy_match);
  /*!
   * \brief Bind symbolic buffer to a DLTensor handle.
   * \param buffer The argument buffer to be binded.
   * \param device_type The device id to be binded.
   * \param device_id The device id to be binded.
   * \param handle The DLTensor handle.
   * \param arg_name argument name.
   */
  void BindDLTensor(const Buffer& buffer,
                    const Expr& device_type,
                    const Expr& device_id,
                    const Var& handle,
                    const std::string& arg_name);

  /*! \return The defs generated in binding. */
  const std::vector<Var>& defs() const {
    return defs_;
  }
  /*! \return The asserts generated in binding */
  const std::vector<Stmt>& asserts() const {
    return asserts_;
  }
  /*!
   * \brief Initialization nest generated
   *  This is only non-empty when BindDLTensor is called.
   *
   * \note The binder may choose to generate a let statement
   *  and simply put def_map to map Variable to itself,
   *  or update def_map to directly map to new value and not generate let statement.
   *
   *  Let statement is usually generated when bind to DLTensor and memory load is involved.
   * \return The initialization nest generated during binding.
   */
  const std::vector<Stmt>& init_nest() const {
    return init_nest_;
  }
  /*! \return Handle data type of the data */
  const Map<Var, Expr>& def_handle_dtype() const {
    return def_handle_dtype_;
  }

 private:
  // Internal bind function
  bool Bind_(const Expr& arg,
             const Expr& value,
             const std::string& arg_name,
             bool with_lets);
  /*! \brief The definition map, can be uses to substitute */
  std::unordered_map<const Variable*, Expr>* def_map_;
  /*! \brief defs generated in the current binder */
  std::vector<Var> defs_;
  /*! \brief Initialize nest */
  std::vector<Stmt> init_nest_;
  /*! \brief handle data type in the defintiions */
  Map<Var, Expr> def_handle_dtype_;
  /*! \brief asserts generated */
  std::vector<Stmt> asserts_;
};
}  // namespace ir
}  // namespace tvm
#endif  // TVM_PASS_ARG_BINDER_H_
