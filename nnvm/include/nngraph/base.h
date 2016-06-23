/*!
 *  Copyright (c) 2016 by Contributors
 * \file base.h
 * \brief Configuation of nngraph as well as basic data structure.
 */
#ifndef NNGRAPH_BASE_H_
#define NNGRAPH_BASE_H_

#include <dmlc/base.h>
#include <dmlc/any.h>
#include <dmlc/logging.h>

namespace nngraph {

/*! \brief any type */
using any = dmlc::any;

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

}  // namespace nngraph
#endif // NNGRAPH_BASE_H_
