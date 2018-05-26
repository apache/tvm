/*!
 *  Copyright (c) 2016 by Contributors
 * \file op.h
 * \brief Operator information structor.
 */
#ifndef NNVM_OP_H_
#define NNVM_OP_H_

#include <string>
#include <vector>
#include <utility>
#include <typeinfo>
#include <limits>
#include <functional>
#include "./base.h"

namespace nnvm {

// forward declarations
class Node;
struct NodeAttrs;
template<typename ValueType>
class OpMap;
class OpRegistryEntry;

/*! \brief constant to indicate it take any length of positional inputs */
static const uint32_t kVarg = std::numeric_limits<uint32_t>::max();

/*!
 * \brief Operator structure.
 *
 *  Besides the fields in the structure,
 *  arbitary additional information can be associated with each op.
 *  See function GetAttr for details.
 *
 * \code
 *  // Example usage of Op
 *
 *  // registeration of oeprators
 *  // NOTE that the attr function can register any
 *  // additional attributes to the operator
 *  NNVM_REGISTER_OP(add)
 *  .describe("add two inputs together")
 *  .set_num_inputs(2)
 *  .attr<OpKernel>("gpu_kernel", AddKernel);
 *
 *  NNVM_REGISTER_OP(sub)
 *  .describe("substract one tensor from another")
 *  .set_num_inputs(2);
 *
 *  // Can call regster multiple times in different files
 *  // to register different part of information
 *  NNVM_REGISTER_OP(sub)
 *  .attr<OpKernel>("gpu_kernel", SubKernel);
 *
 *  // get operators from registry.
 *  void my_function() {
 *    const Op* add = Op::Get("add");
 *    const Op* sub = Op::Get("sub");
 *    // query basic information about each operator.
 *    assert(op->name == "plus");
 *    assert(op->num_inputs == 2);
 *
 *    // get additional registered information,
 *    // Assume user registered a OpKernel type attribute as gpu_kernel on each operator.
 *    const OpMap<OpKernel>& kernel = Op::GetAttr<OpKernel>("gpu_kernel");
 *    // we can get the kernel functions by using operator as key.
 *    auto add_kernel = kernel[add];
 *    auto sub_kernel = kernel[sub];
 *    // subsequent code can make use of the queried kernel functions.
 * }
 * \endcode
 */
class Op {
 public:
  /*! \brief name of the operator */
  std::string name;
  /*!
   * \brief detailed description of the operator
   *  This can be used to generate docstring automatically for the operator.
   */
  std::string description;
  /*!
   * \brief number of inputs to the operator,
   * -1 means it is variable length
   * When get_num_inputs is presented,
   * the number will be decided by get_num_inputs instead.
   * \sa get_num_inputs
   */
  uint32_t num_inputs = 1;
  /*!
   * \brief number of outputs of the operator
   *  When get_num_outputs is presented.
   *  The number of outputs will be decided by
   *  get_num_outputs function
   * \sa get_num_outputs
   */
  uint32_t num_outputs = 1;
  /*!
   * \brief get number of outputs given information about the node.
   * \param attrs The attribute of the node
   * \return number of outputs.
   */
  uint32_t (*get_num_outputs)(const NodeAttrs& attrs) = nullptr;
  /*!
   * \brief get number of inputs given information about the node.
   * \param attrs The attribute of the node
   * \return number of inputs
   */
  uint32_t (*get_num_inputs)(const NodeAttrs& attrs) = nullptr;
  /*!
   * \brief Attribute parser to parse the NodeAttrs information.
   *
   * This can help to get quick access to a parsed attribute
   * object
   *
   * \code
   *  // Example usage of attr_parser.
   *
   *  // Suppose we want to register operator sum.
   *  // The parameters about sum operator
   *  struct SumParam {
   *    int axis;
   *  };
   *  // The parser function
   *  void SumAttrParser(NodeAttrs* attrs) {
   *     // This will be invoked during node construction.
   *     SumParam param;
   *     // parse axis string to integer
   *     param.axis = atoi(attrs->dict["axis"].c_str());
   *     // set the parsed parameter
   *     attrs->parsed = std::move(param);
   *  }
   *  // The other function that can utilize the parsed result.
   *  TShape SumInferShape(const NodeAttrs& attrs,
   *                       const std::vector<TShape>& ishapes) {
   *     // we can use the parsed version of param
   *     // without repeatively parsing the parameter
   *     const SumParam& param = nnvm::get<SumParam>(attrs.parsed);
   *  }
   * \endcode
   */
  void (*attr_parser)(NodeAttrs* attrs) = nullptr;

  // function fields.
  /*!
   * \brief setter function during registration
   *  Set the description of operator
   * \param descr the description string.
   * \return reference to self.
   */
  inline Op& describe(const std::string& descr);  // NOLINT(*)
  /*!
   * \brief Set the num_inputs
   * \param n The number of inputs to be set.
   * \return reference to self.
   */
  inline Op& set_num_inputs(uint32_t n);  // NOLINT(*)
  /*!
   * \brief Set the get_num_outputs function.
   * \param fn The function to be set.
   * \return reference to self.
   */
  inline Op& set_num_inputs(uint32_t (*fn)(const NodeAttrs& attr));  // NOLINT(*)
  /*!
   * \brief Set the num_outputs
   * \param n The number of outputs to be set.
   * \return reference to self.
   */
  inline Op& set_num_outputs(uint32_t n);  // NOLINT(*)
  /*!
   * \brief Set the get_num_outputs function.
   * \param fn The function to be set.
   * \return reference to self.
   */
  inline Op& set_num_outputs(uint32_t (*fn)(const NodeAttrs& attr));  // NOLINT(*)
  /*!
   * \brief Set the attr_parser function.
   * \param fn The number of outputs to be set.
   * \return reference to self.
   */
  inline Op& set_attr_parser(void (*fn)(NodeAttrs* attrs));  // NOLINT(*)
  /*!
   * \brief Register additional attributes to operator.
   * \param attr_name The name of the attribute.
   * \param value The value to be set.
   * \tparam ValueType The type of the value to be set.
   */
  template<typename ValueType>
  inline Op& attr(const std::string& attr_name,  // NOLINT(*)
                  const ValueType& value);
  /*!
   * \brief Get an Op for a given operator name.
   *  Will raise an error if the op has not been registered.
   * \param op_name Name of the operator.
   * \return Pointer to a Op, valid throughout program lifetime.
   */
  static const Op* Get(const std::string& op_name);
  /*!
   * \brief Get additional registered attribute about operators.
   *  If nothing has been registered, an empty OpMap will be returned.
   * \param attr_name The name of the attribute.
   * \return An OpMap of specified attr_name.
   * \tparam ValueType The type of the attribute.
   */
  template<typename ValueType>
  static const OpMap<ValueType>& GetAttr(const std::string& attr_name);

 private:
  template<typename ValueType>
  friend class OpMap;
  friend class dmlc::Registry<Op>;
  // Program internal unique index of operator.
  // Used to help index the program.
  uint32_t index_{0};
  // internal constructor
  Op();
  // get const reference to certain attribute
  static const any* GetAttrMap(const std::string& key);
  // update the attribute OpMap
  static void UpdateAttrMap(const std::string& key,
                            std::function<void(any*)> updater);
};

/*!
 * \brief A map data structure that takes Op* as key
 *  and returns ValueType
 * \tparam ValueType The type of the value stored in map.
 */
template<typename ValueType>
class OpMap {
 public:
  /*!
   * \brief get the corresponding value element at op
   * \param op The key to the map
   * \return the const reference to the content value.
   */
  inline const ValueType& operator[](const Op* op) const;
  /*!
   * \brief get the corresponding value element at op with default value.
   * \param op The key to the map
   * \param def_value The default value when the key does not exist.
   * \return the const reference to the content value.
   */
  inline const ValueType& get(const Op* op, const ValueType& def_value) const;
  /*!
   * \brief Check if the map has op as key.
   * \param op The key to the map
   * \return 1 if op is contained in map, 0 otherwise.
   */
  inline int count(const Op* op) const;

 private:
  friend class Op;
  // internal attribute name
  std::string attr_name_;
  // internal data
  std::vector<std::pair<ValueType, int> > data_;
  OpMap() = default;
};

// internal macros to make
#define NNVM_STR_CONCAT_(__x, __y) __x##__y
#define NNVM_STR_CONCAT(__x, __y) NNVM_STR_CONCAT_(__x, __y)
#define NNVM_REGISTER_VAR_DEF(OpName)                              \
  static ::nnvm::Op & __make_ ## NnvmOp ## _ ## OpName

/*!
 * \def NNVM_REGISTER_OP
 * \brief Register
 * This macro must be used under namespace dmlc, and only used once in cc file.
 * \param OpName The name of registry
 *
 * \code
 *
 *  NNVM_REGISTER_OP(add)
 *  .describe("add two inputs together")
 *  .set_num_inputs(2)
 *  .attr<OpKernel>("gpu_kernel", AddKernel);
 *
 * \endcode
 */
#define NNVM_REGISTER_OP(OpName)                                     \
  NNVM_STR_CONCAT(NNVM_REGISTER_VAR_DEF(OpName), __COUNTER__) =   \
      ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(#OpName)

// implementations of template functions after this.
// member function of Op
template<typename ValueType>
inline const OpMap<ValueType>& Op::GetAttr(const std::string& key) {
  const any* ref = GetAttrMap(key);
  if (ref == nullptr) {
    UpdateAttrMap(key, [key](any* pmap) {
        if (pmap->empty()) {
          OpMap<ValueType> pm;
          pm.attr_name_ = key;
          *pmap = std::move(pm);
        }
      });
    ref = GetAttrMap(key);
  }
  return nnvm::get<OpMap<ValueType> >(*ref);
}

template<typename ValueType>
inline Op& Op::attr(  // NOLINT(*)
    const std::string& attr_name, const ValueType& value) {
  UpdateAttrMap(attr_name, [this, attr_name, value](any* pmap) {
      if (pmap->empty()) {
        OpMap<ValueType> pm;
        pm.attr_name_ = attr_name;
        *pmap = std::move(pm);
      }
      CHECK_EQ(pmap->type(), typeid(OpMap<ValueType>))
          << "Attribute " << attr_name
          << " of operator " << this->name
          << " is registered as inconsistent types"
          << " previously " << pmap->type().name()
          << " current " << typeid(OpMap<ValueType>).name();
      std::vector<std::pair<ValueType, int> >& vec =
          nnvm::get<OpMap<ValueType> >(*pmap).data_;
      // resize the value type.
      vec.resize(index_ + 1,
                 std::make_pair(ValueType(), 0));
      std::pair<ValueType, int>& p = vec[index_];
      CHECK(p.second == 0)
          << "Attribute " << attr_name
          << " of operator " << this->name
          << " is already registered.";
          vec[index_] = std::make_pair(value, 1);
    });
  return *this;
}

inline Op& Op::describe(const std::string& descr) {  // NOLINT(*)
  this->description = descr;
  return *this;
}

inline Op& Op::set_num_inputs(uint32_t n) {  // NOLINT(*)
  this->num_inputs = n;
  return *this;
}

inline Op& Op::set_num_inputs(uint32_t (*fn)(const NodeAttrs& attr)) {  // NOLINT(*)
  this->get_num_inputs = fn;
  return *this;
}

inline Op& Op::set_num_outputs(uint32_t n) {  // NOLINT(*)
  this->num_outputs = n;
  return *this;
}

inline Op& Op::set_num_outputs(uint32_t (*fn)(const NodeAttrs& attr)) {  // NOLINT(*)
  this->get_num_outputs = fn;
  return *this;
}

inline Op& Op::set_attr_parser(void (*fn)(NodeAttrs* attrs)) {  // NOLINT(*)
  this->attr_parser = fn;
  return *this;
}

// member functions of OpMap
template<typename ValueType>
inline int OpMap<ValueType>::count(const Op* op) const {
  const uint32_t idx = op->index_;
  return idx < data_.size() ? data_[idx].second : 0;
}

template<typename ValueType>
inline const ValueType& OpMap<ValueType>::operator[](const Op* op) const {
  const uint32_t idx = op->index_;
  CHECK(idx < data_.size() && data_[idx].second)
        << "Attribute " << attr_name_
        << " has not been registered for Operator " << op->name;
  return data_[idx].first;
}

template<typename ValueType>
inline const ValueType& OpMap<ValueType>::get(const Op* op, const ValueType& def_value) const {
  const uint32_t idx = op->index_;
  if (idx < data_.size() && data_[idx].second) {
    return data_[idx].first;
  } else {
    return def_value;
  }
}

}  // namespace nnvm

#endif  // NNVM_OP_H_
