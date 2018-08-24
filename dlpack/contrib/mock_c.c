// Copyright by contributors
// This file is used to make sure the package is C compatible
#include <dlpack/dlpack.h>

int GetNDim(DLTensor *t) {
  return t->ndim;
}
