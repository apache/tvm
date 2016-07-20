/*!
 *  Copyright (c) 2016 by Contributors
 * \file symbolic.h
 * \brief Symbolic graph construction API
 *
 *  This API is optional, but useful to allow user
 *  to construct NNVM Graph easily, and quickly create
 *  front-end host languages.
 */
#ifndef NNVM_SYMBOLIC_H_
#define NNVM_SYMBOLIC_H_

#include <string>
#include <vector>
#include <utility>

#include "./base.h"
#include "./node.h"

namespace nnvm {
/*!
 * \brief Symbol is used to represent the
 */
class Symbol {
 public:
  /*! \brief option passed to ListAttr */
  enum ListAttrOption {
    /*! \brief recursively list all attributes */
    kRecursive = 0,
    /*! \brief only list attributes in current node */
    kShallow = 1
  };

  /*! \brief output entries contained in the symbol */
  std::vector<NodeEntry> outputs;

  /*!
   * \brief copy the symbol
   * \return a deep copy of the symbolic graph.
   */
  Symbol Copy() const;
  /*!
   * \brief print the symbol info to output stream.
   * \param os the output stream we like to print to
   */
  void Print(std::ostream &os) const; // NOLINT(*)
  /*!
   * \brief get the index th element from the returned tuple.
   * \param index index of multi output
   * \return the symbol corresponds to the indexed element.
   */
  Symbol operator[] (size_t index) const;
  /*!
   * \brief List the arguments names.
   *
   * The position of the returned list also corresponds to calling position in operator()
   * \return the arguments list of this symbol, they can be either named or unnamed (empty string).
   */
  std::vector<std::string> ListArguments() const;
  /*!
   * \brief List the names of outputs for this symbol.
   *  For normal operators, it is usually symbol node name + "_output"
   * \return get the descriptions of outputs for this symbol.
   */
  std::vector<std::string> ListOutputs() const;
  /*!
   * \brief Compose the symbol with arguments, this changes the current symbol.
   * The kwargs passed in can be in-complete,
   *
   * The rest of the symbols will remain the same name.
   *
   * \param args positional arguments
   * \param kwargs keyword arguments for the symbol
   * \param name name of returned symbol.
   */
  void Compose(const array_view<const Symbol*>& args,
               const std::unordered_map<std::string, const Symbol*>& kwargs,
               const std::string& name);
  /*!
   * \brief Apply the symbol as a function, compose with arguments
   * This is equivalent to Copy then Compose.
   * \param args positional arguments for the symbol
   * \param kwargs keyword arguments for the symbol
   * \param name name of returned symbol.
   * \return a new Symbol which is the composition of current symbol with its arguments
   */
  Symbol operator () (const array_view<const Symbol*>& args,
                      const std::unordered_map<std::string, const Symbol*>& kwargs,
                      const std::string& name) const;
  /*!
   * \brief Add control flow depenencies to operators involved in symbols.
   *  For grouped sybmbol, an error will be raised.
   *  This mutate current symbolic Node.
   *
   * \param src The symbols to depend on.
   */
  void AddControlDeps(const Symbol& src);
  /*
   * \brief Get all the internal nodes of the symbol.
   * \return symbol A new symbol whose output contains all the outputs of the symbols
   *  Including input variables and intermediate outputs.
   */
  Symbol GetInternals() const;
  /*!
   * \brief set additional attributes to current node.
   *  This only works for symbol with outputs from single operators.
   *  For grouped sybmbol, an error will be raised.
   *
   *  This function mutate the node's symbol and is not recommended.
   *
   * \param attrs The attributes to set.
   */
  void SetAttrs(const std::vector<std::pair<std::string, std::string> >& attrs);
  /*!
   * \brief Get attributes from the symbol.
   *  This only works for symbol with outputs from single operators.
   *  For grouped sybmbol, an error will be raised.
   * \param key Key of the attribute. When key == "name", it returns the name attirbute.
   * \param out the output value of the attribute.
   * \return true if the attribute exists, false if the attribute do not exist.
   */
  bool GetAttr(const std::string& key, std::string* out) const;
  /*!
   * \brief Get attribute dictionary from the symbol.
   *  For grouped sybmbol, an error will be raised.
   * \param option If recursive is set, the attributes of all children are retrieved,
   *   The name of symbol will be pre-pended to each key.
   * \return The created attribute.
   */
  std::unordered_map<std::string, std::string> ListAttrs(ListAttrOption option) const;
  /*!
   * \brief create symbolic functor(AtomicSymbol) by given operator and attributes.
   * \param op The operator.
   * \param attrs The additional attributes.
   * \return Symbol that can be used to call compose further.
   */
  static Symbol CreateFunctor(const Op* op,
                              std::unordered_map<std::string, std::string> attrs);
  /*!
   * \brief create variable symbol node
   * \param name name of the variable
   * \return the new variable
   */
  static Symbol CreateVariable(const std::string& name);
  /*!
   * \brief create equivalence of symbol by grouping the symbols together
   * \param symbols list of symbols
   * \return the grouped symbol
   */
  static Symbol CreateGroup(const std::vector<Symbol>& symbols);
};

}  // namespace nnvm

#endif  // NNVM_SYMBOLIC_H_
