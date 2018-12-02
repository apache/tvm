/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/Intrin.h
 * \brief instrinsic information structor.
 */
#ifndef INTRIN_H_
#define INTRIN_H_

#include <string>
#include <vector>
#include <utility>
#include <typeinfo>
#include <limits>
#include <functional>
#include "base.h"

namespace tvm {
using HalideIR::Expr;
// forward declarations
class Node;
struct NodeAttrs;
template<typename ValueType>
class IntrinMap;
typedef std::function<bool(const HalideIR::Expr& expr)> inplace_func;

/*! \brief constant to indicate it take any length of positional inputs */
static const uint32_t kVarg = std::numeric_limits<uint32_t>::max();

/*!
 * \brief instrinsic structure.
 *
 *  support inplace function for different target 
 *  arbitary additional information can be associated with each intrinsic.
 *  See function GetAttr for details.
 *
 * \code
 *  // Example usage of Intirn
 *
 *  // registeration of instrinsics
 *  // NOTE that the attr function can register any
 *  // additional attributes to the instrinsic
 *  TVM_REGISTER_INTRIN(vadd)
 *  .describe("add two vector inputs together")
 *  .set_detect_inplace("cce", cce_detect_inplace);
 *
 *  // get instrinsics from registry.
 *  void my_function(Expr& body) {
 *    const Intrin* add = Intrin::Get("vadd");
 *    // query basic information about instrinsic.
 *    assert(op->name == "vadd");
 *    bool inplace = add->detect_inplace("cce", body);
 * }
 * \endcode
 */
class Intrin {
 public:
  /*! \brief name of the instrinsic: Target_IntrinName */
  std::string name;
  /*!
   * \brief detailed description of the instrinsic
   *  This can be used to generate docstring automatically for the instrinsic.
   */
  std::string description;
    /*!
   * \brief inplace optimization for intrinsic.
   * \example: vadd dst_buffer, src_buffer1, src_buffer2
   * \if inplace is set, dst_buffer can inplace src_buffer1 or src_buffer2 in storage rewrite,
   * \which means they share the same buffer allocation
   */
  bool inplace;
  /*!
   * \brief inplace functions for targets.
   */
  std::unordered_map<std::string, inplace_func> inplace_map;
  // function fields.
  /*!
   * \brief setter function during registration
   *  Set the description of instrinsic
   * \param descr the description string.
   * \return reference to self.
   */
  inline Intrin& describe(const std::string& descr);  // NOLINT(*)
  /*!
   * \brief set inplace for intrinsic.
   * \param inplace flag for inplace optimization.
   * \return reference to self.
   */
  inline Intrin& set_inplace(bool inplace);  // NOLINT(*)
  /*!
   * \brief set func for detecting inplace of the operands.
   * \param target The target name
   * \param fn The function to be set for the target.
   * \return reference to self.
   */
  inline Intrin& set_detect_inplace(std::string& target, inplace_func fn);  // NOLINT(*)
  /*!
   * \brief detect inplace of the operands.
   * \return bool inplace or not.
   */
  inline bool detect_inplace() const;  // NOLINT(*)
  /*!
   * \brief detect inplace of the operands.
   * \param target The target name
   * \param expr The intrinsic body with operands.
   * \return bool can inplace or not.
   */
  inline bool detect_inplace(std::string& target, const HalideIR::Expr& expr) const;  // NOLINT(*)
  /*!
   * \brief Register additional attributes to instrinsic.
   * \param attr_name The name of the attribute.
   * \param value The value to be set.
   * \param plevel The priority level of this set,
   *  an higher priority level attribute
   *  will replace lower priority level attribute.
   *  Must be bigger than 0.
   *
   *  Cannot set with same plevel twice in the code.
   *
   * \tparam ValueType The type of the value to be set.
   */
  template<typename ValueType>
  inline Intrin& set_attr(const std::string& attr_name,  // NOLINT(*)
                          const ValueType& value,
                          int plevel = 10);
  /*!
   * \brief Get an Op for a given instrinsic name.
   *  Will raise an error if the op has not been registered.
   * \param intrin_name Name of the instrinsic.
   * \return Pointer to a Op, valid throughout program lifetime.
   */
  static const Intrin* Get(const std::string& intrin_name);
  /*!
   * \brief Get additional registered attribute about instrinsics.
   *  If nothing has been registered, an empty IntrinMap will be returned.
   * \param attr_name The name of the attribute.
   * \return An IntrinMap of specified attr_name.
   * \tparam ValueType The type of the attribute.
   */
  template<typename ValueType>
  static const IntrinMap<ValueType>& GetAttr(const std::string& attr_name);

 private:
  template<typename ValueType>
  friend class IntrinMap;
  friend class dmlc::Registry<Intrin>;
  // Program internal unique index of instrinsic.
  // Used to help index the program.
  uint32_t index_{0};
  // internal constructor
  Intrin();
  // get const reference to certain attribute
  static const dmlc::any* GetAttrMap(const std::string& key);
  // update the attribute IntrinMap
  static void UpdateAttrMap(const std::string& key,
                            std::function<void(dmlc::any*)> updater);
};

/*!
 * \brief A map data structure that takes Intrin* as key
 *  and returns ValueType
 * \tparam ValueType The type of the value stored in map.
 */
template<typename ValueType>
class IntrinMap {
 public:
  /*!
   * \brief get the corresponding value element at op
   * \param op The key to the map
   * \return the const reference to the content value.
   */
  inline const ValueType& operator[](const Intrin* op) const;
  /*!
   * \brief get the corresponding value element at op with default value.
   * \param op The key to the map
   * \param def_value The default value when the key does not exist.
   * \return the const reference to the content value.
   */
  inline const ValueType& get(const Intrin* op, const ValueType& def_value) const;
  /*!
   * \brief Check if the map has op as key.
   * \param op The key to the map
   * \return 1 if op is contained in map, 0 otherwise.
   */
  inline int count(const Intrin* op) const;

 private:
  friend class Intrin;
  // internal attribute name
  std::string attr_name_;
  // internal data
  std::vector<std::pair<ValueType, int> > data_;
  IntrinMap() = default;
};


// internal macros to make
#define TVM_REGISTER_VAR_DEF(IntrinName)                                   \
  static DMLC_ATTRIBUTE_UNUSED ::tvm::Intrin & __make_ ## TVMIntrin ## _ ## IntrinName

/*!
 * \def TVM_REGISTER_INTRIN
 * \brief Register a new instrinsic, or set attribute of the corresponding op.
 *
 * \param IntrinName The name of registry
 *
 * \code
 *
 *  TVM_REGISTER_INTRIN(add)
 *  .describe("add two inputs together")
 *  .set_num_inputs(2)
 *  .set_attr<IntrinKernel>("gpu_kernel", AddKernel);
 *
 * \endcode
 */
#define TVM_REGISTER_INTRIN(IntrinName)                                        \
  DMLC_STR_CONCAT(TVM_REGISTER_VAR_DEF(IntrinName), __COUNTER__) =             \
      ::dmlc::Registry<::tvm::Intrin>::Get()->__REGISTER_OR_GET__(#IntrinName)


// implementations of template functions after this.
// member function of Op
template<typename ValueType>
inline const IntrinMap<ValueType>& Intrin::GetAttr(const std::string& key) {
  const dmlc::any* ref = GetAttrMap(key);
  if (ref == nullptr) {
    // update the attribute map of the key by creating new empty IntrinMap
    UpdateAttrMap(key, [key](dmlc::any* pmap) {
        // use callback so it is in lockscope
        if (pmap->empty()) {
          IntrinMap<ValueType> pm;
          pm.attr_name_ = key;
          *pmap = std::move(pm);
        }
      });
    ref = GetAttrMap(key);
  }
  return dmlc::get<IntrinMap<ValueType> >(*ref);
}

template<typename ValueType>
inline Intrin& Intrin::set_attr(  // NOLINT(*)
    const std::string& attr_name,
    const ValueType& value,
    int plevel) {
  CHECK_GT(plevel, 0)
      << "plevel in set_attr must be greater than 0";
  // update the attribute map of the key by creating new empty if needed.
  UpdateAttrMap(attr_name,
                [this, attr_name, value, plevel](dmlc::any* pmap) {
      // the callback is in lockscope so is threadsafe.
      if (pmap->empty()) {
        IntrinMap<ValueType> pm;
        pm.attr_name_ = attr_name;
        *pmap = std::move(pm);
      }
      CHECK(pmap->type() == typeid(IntrinMap<ValueType>))
          << "Attribute " << attr_name
          << " of instrinsic " << this->name
          << " is registered as inconsistent types"
          << " previously " << pmap->type().name()
          << " current " << typeid(IntrinMap<ValueType>).name();
      std::vector<std::pair<ValueType, int> >& vec =
          dmlc::get<IntrinMap<ValueType> >(*pmap).data_;
      // resize the value type.
      if (vec.size() <= index_) {
        vec.resize(index_ + 1,
                   std::make_pair(ValueType(), 0));
      }
      std::pair<ValueType, int>& p = vec[index_];
      CHECK(p.second != plevel)
          << "Attribute " << attr_name
          << " of instrinsic " << this->name
          << " is already registered with same plevel=" << plevel;
      if (p.second < plevel) {
        vec[index_] = std::make_pair(value, plevel);
      }
    });
  return *this;
}


inline Intrin& Intrin::describe(const std::string& descr) {  // NOLINT(*)
  this->description = descr;
  return *this;
}

inline Intrin& Intrin::set_inplace(bool inplace) { // NOLINT(*)
  this->inplace = inplace;
  return *this;
}

inline Intrin& Intrin::set_detect_inplace(std::string& target, inplace_func fn) {  // NOLINT(*)
  CHECK(!target.empty())<<"Invalid target name!";
  this->inplace_map.at(target) = fn;
  return *this;
}

inline bool Intrin::detect_inplace() const { // NOLINT(*)
   return this->inplace;
}

inline bool Intrin::detect_inplace(std::string& target, const HalideIR::Expr& expr) const { // NOLINT(*)
   if (this->inplace) return true;
   if (this->inplace_map.count(target) == 0) return false;
   auto fn = this->inplace_map.at(target);
   return fn(expr);
}

// member functions of IntrinMap
template<typename ValueType>
inline int IntrinMap<ValueType>::count(const Intrin* op) const {
  if (op == nullptr) return 0;
  const uint32_t idx = op->index_;
  return idx < data_.size() ? (data_[idx].second != 0) : 0;
}

template<typename ValueType>
inline const ValueType& IntrinMap<ValueType>::operator[](const Intrin* op) const {
  CHECK(op != nullptr);
  const uint32_t idx = op->index_;
  CHECK(idx < data_.size() && data_[idx].second)
        << "Attribute " << attr_name_
        << " has not been registered for instrinsic " << op->name;
  return data_[idx].first;
}

template<typename ValueType>
inline const ValueType& IntrinMap<ValueType>::get(const Intrin* op, const ValueType& def_value) const {
  if (op == nullptr) return def_value;
  const uint32_t idx = op->index_;
  if (idx < data_.size() && data_[idx].second) {
    return data_[idx].first;
  } else {
    return def_value;
  }
}

}  // namespace tvm

#endif  // INTRIN_H_
