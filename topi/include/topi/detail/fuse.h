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
  CHECK_GE(args.size(), 1) << "Fuse requires at least 1 arg";

  auto fused = args[0];
  for (size_t i = 1; i < args.size(); ++i) {
    IterVar out;
    stage.fuse(fused, args[i], &out);
    fused = out;
  }
  return fused;
}

}  // namespace detail
}  // namespace topi
#endif  // TOPI_DETAIL_FUSE_H_
