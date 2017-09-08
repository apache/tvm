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

namespace nnvm {
namespace top {

/*! \brief exception throwed by InferShape error */
struct InferShapeError : public dmlc::Error {
  /*! \brief analyze message */
  std::string msg;
  /*! \brief corresponding input index */
  int index;
  // constructor
  InferShapeError(const std::string& msg_, int index)
    : dmlc::Error(msg_), msg(msg_), index(index) {}
};

/*! \brief exception throwed by InferShape error */
struct InferTypeError : public dmlc::Error {
  /*! \brief analyze message */
  std::string msg;
  /*! \brief corresponding input index */
  int index;
  // constructor
  InferTypeError(const std::string& msg_, int index)
    : dmlc::Error(msg_), msg(msg_), index(index) {}
};

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

/*!
 * \brief macro assign shape to out if out is unknown otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param shape_array the shape array to store the result
 * \param index the index of in the array
 * \param shape the inferred shape
 */
#define SHAPE_ASSIGN_CHECK(shape_array, index, shape)                   \
  {                                                                     \
    if (!shape_assign(&(shape_array)[index], TShape(shape))) {          \
      std::ostringstream os;                                            \
      os << "Shape inconsistent, Provided=" << (shape_array)[index] << ',' \
         << " inferred shape=" << shape;                                \
      throw InferShapeError(os.str(), index);                           \
    }                                                                   \
  }

/*!
 * \brief macro assign type to out if out is unknown (-1) otherwise check consistency
 *  Use macro so we can see the error file more clearly
 * \param type_array the type array to store the result
 * \param index the index of in the array
 * \param type the inferred type
 */
#define TYPE_ASSIGN_CHECK(type_array, index, type)                      \
  {                                                                     \
    if (!type_assign(&(type_array)[index], type)) {                     \
      std::ostringstream os;                                            \
      os << "Type inconsistent, Provided="                              \
         << type_string((type_array)[index]) << ','                     \
         << " inferred type=" << type_string(type);                     \
      throw InferTypeError(os.str(), index);                            \
    }                                                                   \
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
