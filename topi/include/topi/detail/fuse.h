/*!
*  Copyright (c) 2017 by Contributors
* \file fuse.h
* \brief Fuse operation
*/
#ifndef TOPI_DETAIL_FUSE_H_
#define TOPI_DETAIL_FUSE_H_

#include "tvm/tvm.h"

namespace topi {
using namespace tvm;

/*! \brief Fuse all of the given args */
IterVar Fuse(Stage stage, const Array<IterVar>& args) {
  CHECK_GE(args.size(), 1) << "Fuse requires at least 2 args";

  auto fused = args[0];
  for (size_t i = 1; i < args.size(); ++i) {
    IterVar out;
    stage.fuse(fused, args[i], &out);
    fused = out;
  }
  return fused;
}

}  // namespace topi
#endif  // TOPI_DETAIL_FUSE_H_
