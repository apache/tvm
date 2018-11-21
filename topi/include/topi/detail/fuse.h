/*!
*  Copyright (c) 2017 by Contributors
* \file fuse.h
* \brief Fuse operation
*/
#ifndef TOPI_DETAIL_FUSE_H_
#define TOPI_DETAIL_FUSE_H_

#include "tvm/tvm.h"

namespace topi {
namespace detail {
using namespace tvm;

/*!
 * \brief Fuse all of the given args
 *
 * \param stage The stage in which to apply the fuse
 * \param args The iteration variables to be fused
 *
 * \return The fused iteration variable
 */
inline IterVar Fuse(Stage stage, const Array<IterVar>& args) {
  IterVar res;
  stage.fuse(args, &res);
  return res;
}

}  // namespace detail
}  // namespace topi
#endif  // TOPI_DETAIL_FUSE_H_
