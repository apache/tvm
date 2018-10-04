/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/attrs.h
 * \brief TVM attribute module
 *
 *  This module enables declaration of named attributes
 *  which support default value setup and bound checking.
 *
 * \code
 *   struct MyAttrs : public tvm::AttrsNode<MyAttrs> {
 *     float learning_rate;
 *     int num_hidden;
 *     std::string name;
 *     // declare attribute fields in header file
 *     TVM_DECLARE_ATTRS(MyAttrs, "attrs.MyAttrs") {
 *       TVM_ATTR_FIELD(num_hidden).set_lower_bound(1);
 *       TVM_ATTR_FIELD(learning_rate).set_default(0.01f);
 *       TVM_ATTR_FIELD(name).set_default("hello");
 *     }
 *   };
 *   // register it in cc file
 *   TVM_REGISTER_NODE_TYPE(MyAttrs);
 * \endcode
 *
 * \sa AttrsNode, TVM_DECLARE_ATTRS, TVM_ATTR_FIELD
 */
#ifndef TVM_ATTRS_H_
#define TVM_ATTRS_H_

#include <unordered_map>
#include <vector>
#include <type_traits>
#include <string>
#include "ir.h"
#include "base.h"
#include "packed_func_ext.h"

namespace tvm {
/*!
 * \brief Declare an attribute function.
 * \param ClassName The name of the class.
 * \param TypeKey The type key to be used by the TVM node system.
 */
#define TVM_DECLARE_ATTRS(ClassName, TypeKey)                   \
  static constexpr const char* _type_key = TypeKey;             \
  TVM_DECLARE_NODE_TYPE_INFO(ClassName, ::tvm::BaseAttrsNode);  \
  template<typename FVisit>                                     \
  void __VisitAttrs__(FVisit& __fvisit__)  // NOLINT(*)


/*!
 * \brief Declare an attribute field.
 * \param FieldName The field name.
 */
#define TVM_ATTR_FIELD(FieldName) \
  __fvisit__(#FieldName, &FieldName)


/*!
 * \brief Create a NodeRef type that represents null.
 * \tparam TNodeRef the type to be created.
 * \return A instance that will represent None.
 */
template<typename TNodeRef>
inline TNodeRef NullValue() {
  return TNodeRef(NodePtr<Node>(nullptr));
}

template<>
inline Type NullValue<Type>() {
  return Type(Type::Handle, 0, 0);
}


/*! \brief Error thrown during attribute checking. */
struct AttrError : public dmlc::Error {
  /*!
   * \brief constructor
   * \param msg error message
   */
  explicit AttrError(const std::string &msg)
      : dmlc::Error(msg) {}
};

/*!
 * \brief Information about attribute fields in string representations.
 */
class AttrFieldInfoNode : public Node {
 public:
  /*! \brief name of the field */
  std::string name;
  /*! \brief type docstring information in str. */
  std::string type_info;
  /*! \brief detailed description of the type */
  std::string description;

  void VisitAttrs(AttrVisitor* v) final {
    v->Visit("name", &name);
    v->Visit("type_info", &type_info);
    v->Visit("description", &description);
  }
  static constexpr const char* _type_key = "AttrFieldInfo";
  TVM_DECLARE_NODE_TYPE_INFO(AttrFieldInfoNode, Node);
};

/*! \brief AttrFieldInfo */
TVM_DEFINE_NODE_REF(AttrFieldInfo, AttrFieldInfoNode);

/*!
 * \brief Base class of all attribute class
 * \note Do not subclass AttrBaseNode directly,
 *       subclass AttrsNode instead.
 * \sa AttrsNode
 */
class BaseAttrsNode : public Node {
 public:
  using TVMArgs = runtime::TVMArgs;
  using TVMRetValue = runtime::TVMRetValue;
  /*!
   * \brief Initialize the attributes by sequence of arguments
   * \param args The postional arguments in the form
   *        [key0, value0, key1, value1, ..., key_n, value_n]
   */
  template<typename... Args>
  inline void InitBySeq(Args&& ...args);
  /*!
   * \brief Print readible docstring to ostream, add newline.
   * \param os the stream to print the docstring to.
   */
  inline void PrintDocString(std::ostream &os) const;  // NOLINT(*)
  /*!
   * \brief Get the field information about the
   * \note This function throws when the required a field is not present.
   */
  TVM_DLL virtual Array<AttrFieldInfo> ListFieldInfo() const = 0;
  /*!
   * \brief Initialize the attributes by arguments.
   * \param kwargs The key value pairs for initialization.
   *        [key0, value0, key1, value1, ..., key_n, value_n]
   * \param allow_unknown Whether allow additional unknown fields.
   * \note This function throws when the required a field is not present.
   */
  TVM_DLL virtual void InitByPackedArgs(const TVMArgs& kwargs, bool allow_unknown = false) = 0;

  static constexpr const char* _type_key = "Attrs";
  TVM_DECLARE_BASE_NODE_INFO(BaseAttrsNode, Node);
};

/*! \brief Base attribute container for all attributes */
class Attrs : public NodeRef {
 public:
  // normal constructor
  Attrs() {}
  // construct from shared ptr.
  explicit Attrs(NodePtr<Node> n) : NodeRef(n) {}

  /*! \return The attribute node */
  const BaseAttrsNode* operator->() const {
    return ptr();
  }
  /*! \brief specify container node */
  using ContainerType = BaseAttrsNode;

 private:
  /*! \return the internal attribute node */
  const BaseAttrsNode* ptr() const {
    return static_cast<const BaseAttrsNode*>(node_.get());
  }
};

/*!
 * \brief Specialized attribute type that is backed by a map.
 *  The DictAttrsNode implements the Attrs behavior,
 *  its fields are directly accessible via object.field_name
 *  like other normal nodes.
 */
class DictAttrsNode : public BaseAttrsNode {
 public:
  /*! \brief internal attrs map */
  Map<std::string, NodeRef> dict;
  /*!
   * \brief Consruct a Attrs backed by DictAttrsNode.
   * \param dict The attributes.
   * \return The dict attributes.
   */
  TVM_DLL static Attrs make(Map<std::string, NodeRef> dict);
  // implementations
  void VisitAttrs(AttrVisitor* v) final;
  void InitByPackedArgs(const runtime::TVMArgs& args, bool allow_unknown) final;
  Array<AttrFieldInfo> ListFieldInfo() const final;
  // type info
  static constexpr const char* _type_key = "DictAttrs";
  TVM_DECLARE_NODE_TYPE_INFO(DictAttrsNode, BaseAttrsNode);
};

// Namespace containing detail implementations
namespace detail {
using runtime::TVMArgValue;

// helper entry that does nothing in set_default/bound/describe calls.
struct AttrNopEntry {
  using TSelf = AttrNopEntry;

  TSelf& describe(DMLC_ATTRIBUTE_UNUSED const char* str) {
    return *this;
  }
  template<typename T>
  TSelf& set_default(DMLC_ATTRIBUTE_UNUSED T value) {
    return *this;
  }
  template<typename T>
  TSelf& set_lower_bound(DMLC_ATTRIBUTE_UNUSED T begin) {
    return *this;
  }
  template<typename T>
  TSelf& set_upper_bound(DMLC_ATTRIBUTE_UNUSED T end) {
    return *this;
  }
};

// Wrapper for normal visitor.
class AttrNormalVisitor {
 public:
  explicit AttrNormalVisitor(AttrVisitor* visitor)
      : visitor_(visitor) {
  }
  template<typename T>
  AttrNopEntry operator()(const char* key, T* value) {
    visitor_->Visit(key, value);
    return AttrNopEntry();
  }

 private:
  AttrVisitor* visitor_;
};

// helper entry that does initialization, set default.
template<typename T>
struct AttrInitEntry {
  // The attributes
  using TSelf = AttrInitEntry<T>;
  // The type key
  const char* type_key_;
  // field name
  const char* key_;
  // internal value.
  T* value_;
  // whether the value is missing.
  bool value_missing_{true};
  // If the value is still missing in destruction time throw an error.
  ~AttrInitEntry() DMLC_THROW_EXCEPTION {
    if (value_missing_) {
      std::ostringstream os;
      os << type_key_ << ": Cannot find required field \'" << key_
         << "\' during initialization";
      throw AttrError(os.str());
    }
  }
  // override fields.
  // This function sets the lower bound of the attribute
  TSelf& set_lower_bound(DMLC_ATTRIBUTE_UNUSED const T& begin) {
    if (this->value_missing_)  return *this;
    const T& val = *value_;
    if (begin > val) {
      std::ostringstream os;
      os << type_key_ << "." << key_ << ": "
         << "value " << val
         << " is smaller than the lower bound " << begin;
      throw AttrError(os.str());
    }
    return *this;
  }
  // This function sets the upper bound of the attribute
  TSelf& set_upper_bound(DMLC_ATTRIBUTE_UNUSED const T& end) {
    if (this->value_missing_)  return *this;
    const T& val = *value_;
    if (val > end) {
      std::ostringstream os;
      os << type_key_ << "." << key_ << ": "
         << "value " << val
         << " is bigger than the upper bound " << end;
      throw AttrError(os.str());
    }
    return *this;
  }
  // set default when
  TSelf& set_default(DMLC_ATTRIBUTE_UNUSED const T& value) {
    if (!value_missing_) return *this;
    *value_ = value;
    value_missing_ = false;
    return *this;
  }
  TSelf& describe(DMLC_ATTRIBUTE_UNUSED const char* str) {
    return *this;
  }
};

// Template function to allow smart conversion
// from Expr types into the constants.
template<typename T>
inline void SetValue(T* ptr, const TVMArgValue& val) {
  *ptr = val.operator T();
}
template<typename T>
inline void SetIntValue(T* ptr, const TVMArgValue& val) {
  if (val.type_code() == kDLInt) {
    *ptr = static_cast<T>(val.value().v_int64);
  } else {
    Expr expr = val;
    CHECK(expr.defined());
    if (const ir::IntImm* op = expr.as<ir::IntImm>()) {
      *ptr = static_cast<T>(op->value);
    } else if (const ir::UIntImm* op = expr.as<ir::UIntImm>()) {
      *ptr = static_cast<T>(op->value);
    } else {
      LOG(FATAL) << "Expect int value, but get " << expr->type_key();
    }
  }
}
template<>
inline void SetValue<std::string>(std::string* ptr, const TVMArgValue& val) {
  if (val.type_code() == kStr) {
    *ptr = val.operator std::string();
  } else {
    Expr expr = val;
    const ir::StringImm* op = expr.as<ir::StringImm>();
    CHECK(op != nullptr);
    *ptr = op->value;
  }
}
template<>
inline void SetValue(Type* ptr, const TVMArgValue& val) {
  *ptr = val.operator Type();
}
template<>
inline void SetValue<double>(double* ptr, const TVMArgValue& val) {
  if (val.type_code() == kDLFloat || val.type_code() == kDLInt) {
    *ptr = val.operator double();
  } else {
    Expr expr = val;
    CHECK(expr.defined());
    if (const ir::IntImm* op = expr.as<ir::IntImm>()) {
      *ptr = static_cast<double>(op->value);
    } else if (const ir::IntImm* op = expr.as<ir::IntImm>()) {
      *ptr = static_cast<double>(op->value);
    } else if (const ir::UIntImm* op = expr.as<ir::UIntImm>()) {
      *ptr = static_cast<double>(op->value);
    } else {
      LOG(FATAL) << "Expect float value, but get " << expr->type_key();
    }
  }
}
template<>
inline void SetValue<int>(int* ptr, const TVMArgValue& val) {
  SetIntValue(ptr, val);
}
template<>
inline void SetValue<int64_t>(int64_t* ptr, const TVMArgValue& val) {
  SetIntValue(ptr, val);
}
template<>
inline void SetValue<uint64_t>(uint64_t* ptr, const TVMArgValue& val) {
  SetIntValue(ptr, val);
}
template<>
inline void SetValue<bool>(bool* ptr, const TVMArgValue& val) {
  SetIntValue(ptr, val);
}

// Visitor for value initialization
template<typename FFind>
class AttrInitVisitor {
 public:
  // Counter of number of matched attributes during visit.
  // This is used to decide if there is additional unmatched attributes.
  size_t hit_count_{0};
  // constructor
  AttrInitVisitor(const char* type_key, FFind ffind)
      : type_key_(type_key), ffind_(ffind) {
  }

  template<typename T>
  AttrInitEntry<T> operator()(const char* key, T* value) {
    TVMArgValue val;
    AttrInitEntry<T> opt;
    opt.type_key_ = type_key_;
    opt.key_ = key;
    opt.value_ = value;
    if (ffind_(key, &val)) {
      SetValue(value, val);
      opt.value_missing_ = false;
      ++hit_count_;
    } else {
      opt.value_missing_ = true;
    }
    return opt;
  }

 private:
  // the type key
  const char* type_key_;
  FFind ffind_;
};

template<typename FFind>
inline AttrInitVisitor<FFind> CreateInitVisitor(
    const char* type_key,
    FFind ffind) {
  return AttrInitVisitor<FFind>(type_key, ffind);
}

/*!
 * \brief Helper struct to get the type name known to tvm.
 * \tparam T the type we are interested in.
 */
template<typename T>
struct TypeName {
  static constexpr const char* value = T::ContainerType::_type_key;
};

template<>
struct TypeName<int> {
  static constexpr const char* value = "int";
};

template<>
struct TypeName<int64_t> {
  static constexpr const char* value = "int64";
};

template<>
struct TypeName<uint64_t> {
  static constexpr const char* value = "uint64_t";
};

template<>
struct TypeName<Type> {
  static constexpr const char* value = "Type";
};

template<>
struct TypeName<std::string> {
  static constexpr const char* value = "str";
};

template<>
struct TypeName<bool> {
  static constexpr const char* value = "bool";
};

template<>
struct TypeName<void*> {
  static constexpr const char* value = "handle";
};

template<>
struct TypeName<double> {
  static constexpr const char* value = "double";
};

class AttrDocEntry {
 public:
  using TSelf = AttrDocEntry;

  explicit AttrDocEntry(NodePtr<AttrFieldInfoNode> info)
      : info_(info) {
  }
  TSelf& describe(DMLC_ATTRIBUTE_UNUSED const char* str) {
    info_->description = str;
    return *this;
  }
  template<typename T>
  TSelf& set_default(DMLC_ATTRIBUTE_UNUSED T value) {
    std::ostringstream os;
    os << info_->type_info << ", default=" << value;
    info_->type_info = os.str();
    return *this;
  }
  template<typename T>
  TSelf& set_lower_bound(DMLC_ATTRIBUTE_UNUSED T begin) {
    return *this;
  }
  template<typename T>
  TSelf& set_upper_bound(DMLC_ATTRIBUTE_UNUSED T end) {
    return *this;
  }

 private:
  NodePtr<AttrFieldInfoNode> info_;
};

class AttrDocVisitor {
 public:
  template<typename T>
  AttrDocEntry operator()(const char* key, T* v) {
    NodePtr<AttrFieldInfoNode> info
        = make_node<AttrFieldInfoNode>();
    info->name = key;
    info->type_info = TypeName<T>::value;
    fields_.push_back(AttrFieldInfo(info));
    return AttrDocEntry(info);
  }

  Array<AttrFieldInfo> fields_;
};

class AttrExistVisitor {
 public:
  std::string key_;
  bool exist_{false};

  template<typename T>
  AttrNopEntry operator()(const char* key, T* v) {
    if (exist_) return AttrNopEntry();
    if (key == key_) exist_ = true;
    return AttrNopEntry();
  }
};
}  // namespace detail

/*!
 * \brief The base class of the all the
 *  Use "curiously recurring template pattern".
 *
 * \tparam DerivedType The final attribute type.
 */
template<typename DerivedType>
class AttrsNode : public BaseAttrsNode {
 public:
  void VisitAttrs(AttrVisitor* v) final {
    detail::AttrNormalVisitor vis(v);
    self()->__VisitAttrs__(vis);
  }

  void InitByPackedArgs(const runtime::TVMArgs& args, bool allow_unknown) final {
    CHECK_EQ(args.size() % 2, 0);
    const int kLinearSearchBound = 16;
    int hit_count = 0;
    // applies two stratgies to lookup
    if (args.size() < kLinearSearchBound) {
      // linear search.
      auto ffind = [&args](const char* key, runtime::TVMArgValue* val) {
        for (int i = 0; i < args.size(); i += 2) {
          CHECK_EQ(args.type_codes[i], kStr);
          if (!std::strcmp(key, args.values[i].v_str)) {
            *val = args[i + 1];
            return true;
          }
        }
        return false;
      };
      auto vis = detail::CreateInitVisitor(DerivedType::_type_key, ffind);
      self()->__VisitAttrs__(vis);
      hit_count = vis.hit_count_;
    } else {
      // construct a map then do lookup.
      std::unordered_map<std::string, runtime::TVMArgValue> kwargs;
      for (int i = 0; i < args.size(); i += 2) {
        CHECK_EQ(args.type_codes[i], kStr);
        kwargs[args[i].operator std::string()] = args[i + 1];
      }
      auto ffind = [&kwargs](const char *key, runtime::TVMArgValue* val) {
        auto it = kwargs.find(key);
        if (it != kwargs.end()) {
          *val = it->second;
          return true;
        }
        return false;
      };
      auto vis = detail::CreateInitVisitor(DerivedType::_type_key, ffind);
      self()->__VisitAttrs__(vis);
      hit_count = vis.hit_count_;
    }
    // error handling, slow path
    if (hit_count * 2 != args.size() && !allow_unknown) {
      for (int i = 0; i < args.size(); i += 2) {
        detail::AttrExistVisitor visitor;
        visitor.key_ = args[i].operator std::string();
        self()->__VisitAttrs__(visitor);
        if (!visitor.exist_) {
          std::ostringstream os;
          os << DerivedType::_type_key
             << ": does not have field \'" << visitor.key_
             << "\', Possible fields:\n";
          os << "----------------\n";
          this->PrintDocString(os);
          throw AttrError(os.str());
        }
      }
    }
  }

  Array<AttrFieldInfo> ListFieldInfo() const final {
    detail::AttrDocVisitor visitor;
    self()->__VisitAttrs__(visitor);
    return visitor.fields_;
  }

 private:
  DerivedType* self() const {
    return const_cast<DerivedType*>(
        static_cast<const DerivedType*>(this));
  }
};


template<typename... Args>
inline void BaseAttrsNode::InitBySeq(Args&& ...args) {
  runtime::PackedFunc pf([this](const TVMArgs& args, TVMRetValue *rv) {
      this->InitByPackedArgs(args);
    });
  pf(std::forward<Args>(args)...);
}

inline void BaseAttrsNode::PrintDocString(std::ostream &os) const { // NOLINT(*)
  Array<AttrFieldInfo> entry = this->ListFieldInfo();
  for (AttrFieldInfo info : entry) {
    os << info->name << " : " << info->type_info << '\n';
    if (info->description.length() != 0) {
      os << "    " << info->description << '\n';
    }
  }
}

}  // namespace tvm
#endif  // TVM_ATTRS_H_
