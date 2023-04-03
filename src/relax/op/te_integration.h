#include <tvm/relax/op_attr_types.h>
#include <tvm/relax/struct_info.h>

namespace tvm {
namespace relax {

FInferStructInfo InferStructInfoFromTE(std::string global_func_name);

FLegalize LegalizeFromTE(std::string global_func_name, std::string primfunc_name_hint);

}  // namespace relax
}  // namespace tvm
