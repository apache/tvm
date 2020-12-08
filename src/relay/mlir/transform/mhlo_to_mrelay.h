#ifndef TRANSFORM_MHLO_TO_MRELAY_H
#define TRANSFORM_MHLO_TO_MRELAY_H

#include "src/ir/relay_ops.h"
#include <iostream>
#include <mlir-hlo/Dialect/mhlo/IR/hlo_ops.h>
#include <type_traits>

namespace mlir {
namespace mrelay {

template <typename MhloOpTy>
struct MhloToMrelayOpImpl {
using Type = std::false_type;
};
template <typename MhloOpTy>
using MhloToMrelayOp = typename MhloToMrelayOpImpl<MhloOpTy>::Type;

#define MAP_MHLO_TO_MRELAY(OpName)                                             \
  template <> struct MhloToMrelayOpImpl<mhlo::OpName> {                        \
    using Type = mrelay::OpName;                                               \
  }

MAP_MHLO_TO_MRELAY(AddOp);
MAP_MHLO_TO_MRELAY(ConstOp);
// MAP_MHLO_TO_MRELAY(ExpOp);
// MAP_MHLO_TO_MRELAY(MaxOp);
MAP_MHLO_TO_MRELAY(ReshapeOp);
// MAP_MHLO_TO_MRELAY(SliceOp);

#undef MAP_MHLO_TO_MRELAY

} // namespace mrelay
}  // namespace mlir

#endif  // TRANSFORM_MHLO_TO_MRELAY_H

