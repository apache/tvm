/*!
 *  Copyright (c) 2016 by Contributors
 * \file base.h
 * \brief Configuation of nnvm as well as basic data structure.
 */
#ifndef NNVM_BASE_H_
#define NNVM_BASE_H_

#include <dmlc/base.h>
#include <dmlc/any.h>
#include <dmlc/memory.h>
#include <dmlc/logging.h>
#include <dmlc/registry.h>
#include <dmlc/array_view.h>

namespace nnvm {

/*! \brief any type */
using any = dmlc::any;

/*!
 * \brief array_veiw type
 * \tparam ValueType The value content of array view.
 */
template<typename ValueType>
using array_view = dmlc::array_view<ValueType>;

/*!
 * \brief get reference of type T stored in src.
 * \param src The source container
 * \return the reference to the type.
 * \tparam T The type to be fetched.
 */
template<typename T>
inline T& get(any& src) {  // NOLINT(*)
  return dmlc::get<T>(src);
}

/*!
 * \brief get const reference of type T stored in src.
 * \param src The source container
 * \return the reference to the type.
 * \tparam T The type to be fetched.
 */

template<typename T>
inline const T& get(const any& src) {
  return dmlc::get<T>(src);
}

}  // namespace nnvm

#endif  // NNVM_BASE_H_
