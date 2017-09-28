/*!
 *  Copyright (c) 2017 by Contributors
 * \file op_common.h
 * \brief Common operator utilities
 */
#ifndef NNVM_TOP_OP_COMMON_H_
#define NNVM_TOP_OP_COMMON_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <string>
#include <vector>
#include <unordered_set>

namespace nnvm {
namespace top {
/*!
 * \brief Parse keyword arguments as PType arguments and save to parsed
 * \tparam PType the arameter type.
 * \param attrs The attributes.
 */
template<typename PType>
inline void ParamParser(nnvm::NodeAttrs* attrs) {
  PType param;
  try {
    param.Init(attrs->dict);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }
  attrs->parsed = std::move(param);
}

/*!
 * \brief Parse keyword arguments as PType arguments and save to parsed
 * \tparam PType the arameter type.
 * \param attrs The attributes.
 */
template<typename PType>
inline std::unordered_map<std::string, std::string>
ParamGetAttrDict(const nnvm::NodeAttrs& attrs) {
  std::unordered_map<std::string, std::string> dict = attrs.dict;
  nnvm::get<PType>(attrs.parsed).UpdateDict(&dict);
  return dict;
}

/*! \brief check if shape is empty or contains unkown (0) dim. */
inline bool shape_is_none(const TShape& x) {
  return x.ndim() == 0 || x.Size() == 0;
}

/*! \brief check if type is none (-1) */
inline bool type_is_none(const int& x) {
  return x == -1;
}

/*! \brief check if shape is scalar({1}). */
inline bool shape_is_scalar(const TShape& x) {
  return x.ndim() == 1 && x.Size() == 1;
}

/*! \brief get string representation of shape */
inline std::string shape_string(const TShape& x) {
  std::ostringstream os;
  os << x;
  return os.str();
}

/*! \brief get string representation of shape */
inline std::string type_string(const int& x) {
  return std::to_string(x);
}

/*!
 * \brief Assign x to y. Checks for compatiblity when y is not empty.
 *  Allow missing dim in both x and y (as 0).
 * \param y target shape.
 * \param x source shape.
 * \return whether x and y are compatible.
 */
inline bool shape_assign(TShape *y, const TShape& x) {
  if (y->ndim() == 0) {
    *y = x;
    return true;
  } else if (y->ndim() != x.ndim()) {
    return x.ndim() == 0;
  } else {
    for (size_t i = 0; i < y->ndim(); ++i) {
      if ((*y)[i] == 0) {
        (*y)[i] = x[i];
      } else if ((*y)[i] != x[i] && x[i] != 0) {
        return false;
      }
    }
    return true;
  }
}

/*!
 * \brief Assign x to y. Checks for compatiblity when y is not -1.
 * \param y target type.
 * \param x source type.
 * \return whether x and y are compatible.
 */
inline bool type_assign(int *y, const int& x) {
  if (*y == -1) {
    *y = x;
    return true;
  } else if (*y != x && x != -1) {
    return false;
  }
  return true;
}

template<typename AttrType>
inline std::string attr_assign_error_msg(const NodeAttrs& attrs,
                                         int index, bool is_input,
                                         const AttrType& expected,
                                         const AttrType& actual,
                                         const char* attr_name) {
  static const auto& flist_inputs = Op::GetAttr<FListInputNames>("FListInputNames");
  static const auto& flist_outputs = Op::GetAttr<FListOutputNames>("FListOutputNames");
  const auto& flist = is_input ? flist_inputs : flist_outputs;
  std::string name;
  if (flist.count(attrs.op)) {
    name = flist[attrs.op](attrs)[index];
  } else {
    name = (is_input ? "data" : "output") + std::to_string(index);
  }
  std::ostringstream msg;
  msg << "Operator " << attrs.op->name << "(";
  for (const auto& kv : attrs.dict) msg << kv.first << "=" << kv.second << ", ";
  msg << "name=" << attrs.name << ") expects " << name << "\'s " << attr_name
      << " to be " << expected << ", but got " << actual << ".";
  return msg.str();
}

/*!
 * \brief macro assign shape to out if out is unknown otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param inputs the shape array to store the result
 * \param index the index of in the array
 * \param shape the inferred shape
 */
#define NNVM_ASSIGN_INPUT_SHAPE(attrs, inputs, index, shape)             \
  {                                                                      \
    if (!shape_assign(&(inputs)[index], TShape(shape))) {                \
      LOG(FATAL) << attr_assign_error_msg(attrs, index, true, shape,     \
                                          (inputs)[index], "shape");     \
    }                                                                    \
  }

/*!
 * \brief macro assign shape to out if out is unknown otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param inputs the shape array to store the result
 * \param index the index of in the array
 * \param shape the inferred shape
 */
#define NNVM_ASSIGN_OUTPUT_SHAPE(attrs, outputs, index, shape)           \
  {                                                                      \
    if (!shape_assign(&(outputs)[index], TShape(shape))) {               \
      LOG(FATAL) << attr_assign_error_msg(attrs, index, false, shape,    \
                                          (outputs)[index], "shape");    \
    }                                                                    \
  }

/*!
 * \brief macro assign type to out if out is unknown (-1) otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param inputs the type array to store the result
 * \param index the index of in the array
 * \param type the inferred type
 */
#define NNVM_ASSIGN_INPUT_TYPE(attrs, inputs, index, type)               \
  {                                                                      \
    if (!type_assign(&(inputs)[index], type)) {                          \
      LOG(FATAL) << attr_assign_error_msg(attrs, index, true, type,      \
                                          (inputs)[index], "type");      \
    }                                                                    \
  }

/*!
 * \brief macro assign type to out if out is unknown (-1) otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param inputs the type array to store the result
 * \param index the index of in the array
 * \param type the inferred type
 */
#define NNVM_ASSIGN_OUTPUT_TYPE(attrs, outputs, index, type)             \
  {                                                                      \
    if (!type_assign(&(outputs)[index], type)) {                         \
      LOG(FATAL) << attr_assign_error_msg(attrs, index, false, type,     \
                                          (outputs)[index], "type");     \
    }                                                                    \
  }

// simply return the shape as same
inline bool SameShape(const NodeAttrs& attrs,
                      std::vector<TShape> *ishape,
                      std::vector<TShape> *oshape) {
  if (ishape->size() == 0 || (*ishape)[0].ndim() == 0) return false;
  for (TShape& pshape : *oshape) {
    pshape = (*ishape)[0];
  }
  for (TShape& pshape : *ishape) {
    pshape = (*ishape)[0];
  }
  return true;
}

}  // namespace top
}  // namespace nnvm

#endif  // NNVM_TOP_OP_COMMON_H_
