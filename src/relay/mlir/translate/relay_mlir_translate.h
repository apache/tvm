#ifndef TRANSLATE_RELAY_MLIR_TRANSLATE_H_
#define TRANSLATE_RELAY_MLIR_TRANSLATE_H_

#include <mlir/IR/Module.h>

namespace mlir {
namespace mrelay {

OwningModuleRef RelayToMlirTranslateFunction(llvm::StringRef input,
                                             MLIRContext *context);

LogicalResult MlirToRelayTranslateFunction(ModuleOp module,
                                           llvm::raw_ostream &output);

} // namespace mrelay
} // namespace mlir

#endif // TRANSLATE_RELAY_MLIR_TRANSLATE_H_
