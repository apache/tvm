#ifndef RELAY_MLIR_PASSES_H
#define RELAY_MLIR_PASSES_H

#include <llvm/ADT/ArrayRef.h>
#include <memory>

namespace mlir {

class FuncOp;
class FunctionPass;
class ModuleOp;
class Operation;
template <typename T>
class OperationPass;
class Pass;

namespace mrelay {
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToMrelayPass();

} // namespace mrelay
}  // namespace mlir

#endif //RELAY_MLIR_PASSES_H
