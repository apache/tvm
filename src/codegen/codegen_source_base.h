/*!
 *  Copyright (c) 2018 by Contributors
 * \file codegen_source_base.h
 * \brief Common utilities to source code in text form.
 */
#ifndef TVM_CODEGEN_CODEGEN_SOURCE_BASE_H_
#define TVM_CODEGEN_CODEGEN_SOURCE_BASE_H_

#include <tvm/ir.h>
#include <tvm/codegen.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace tvm {
namespace codegen {

/*!
 * \brief A base class to generate source code.
 * Contains helper utilities to generate nest and ssa form.
 */
class CodeGenSourceBase {
 public:
  /*!
   * \brief Register constant value appeared in expresion tree
   *  This avoid generated a ssa id for each appearance of the value
   * \param value The constant value.
   */
  void MarkConst(std::string value);

 protected:
  /*! \brief entry in ssa assign map */
  struct SSAEntry {
    /*! \brief The value id */
    std::string vid;
    /*! \brief The scope id, used to check if this entry is invalid. */
    int scope_id;
  };
  /*! \brief Clear the states that might relates to function generation */
  void ClearFuncState();
  /*! \brief print the current indented value */
  void PrintIndent();
  /*!
   * \brief Allocate a variable name for a newly defined var.
   * \param v The variable.
   * \return the variable name.
   */
  std::string AllocVarID(const Variable* v);
  /*!
   * \brief Get a variable name.
   * \param v The variable.
   * \return the variable name.
   */
  std::string GetVarID(const Variable* v) const;
  /*!
   * \brief Get the SSA ID corresponds to src
   *  If necessary, generate new assignment
   * \param src The source expression
   * \param t The type of the expression.
   */
  std::string SSAGetID(std::string src, Type t);
  /*!
   * \brief get a unique name with the corresponding prefix
   * \param prefix The prefix of the name
   * \return The returned name.
   */
  std::string GetUniqueName(std::string prefix);
  /*!
   * \brief mark the beginning of a new scope
   * \return The scope id.
   */
  int BeginScope();
  /*!
   * \brief mark the end of an old scope.
   * \param scope_id The scope id to be ended.
   */
  void EndScope(int scope_id);
  /*!
   * \brief Print assignment of src to the id in ssa entry.
   * \param target id of target variable.
   * \param src The source expression.
   * \param t The type of target.
   */
  virtual void PrintSSAAssign(
      const std::string& target, const std::string& src, Type t) = 0;

  /*! \brief the declaration stream */
  std::ostringstream decl_stream;
  /*! \brief the stream to be printed */
  std::ostringstream stream;
  /*! \brief name of each variable */
  std::unordered_map<const Variable*, std::string> var_idmap_;

 private:
  /*! \brief assignment map of ssa */
  std::unordered_map<std::string, SSAEntry> ssa_assign_map_;
  /*! \brief name allocation map */
  std::unordered_map<std::string, int> name_alloc_map_;
  /*! \brief array to check whether we are inside certain scope */
  std::vector<bool> scope_mark_;
  /*! \brief The current indentation value */
  int indent_{0};
};

/*!
 * \brief Create a source module for viewing.
 * \param code The code to be viewed.
 * \param fmt The code. format.
 */
runtime::Module SourceModuleCreate(std::string code, std::string fmt);
}  // namespace codegen
}  // namespace tvm
#endif  // TVM_CODEGEN_CODEGEN_SOURCE_BASE_H_
