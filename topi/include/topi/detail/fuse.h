/*!
*  Copyright (c) 2017 by Contributors
* \file fuse.h
* \brief Fuse operation
*/
#ifndef TOPI_FUSE_H_
#define TOPI_FUSE_H_

#include "tvm/tvm.h"

namespace topi {
using namespace tvm;

IterVar Fuse(Stage stage, const Array<IterVar>& args)
{
  CHECK_GE(args.size(), 1) << "Fuse requires at least 2 args";

  auto fused = args[0];
  for (auto i = 1; i < args.size(); ++i) {
    IterVar out;
    stage.fuse(fused, args[i], &out);
    fused = out;
  }
  return fused;
}

}  // namespace topi
#endif  // TOPI_FUSE_H_
