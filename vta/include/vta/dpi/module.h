#ifndef VTA_DPI_MODULE_H_
#define VTA_DPI_MODULE_H_

#include <mutex>
#include <queue>
#include <condition_variable>
#include <tvm/runtime/module.h>

namespace vta {
namespace dpi {

class DPIModuleNode : public tvm::runtime::ModuleNode {
 public:
  virtual void Launch(uint64_t max_cycles) = 0;
  virtual void WriteReg(int addr, uint32_t value) = 0;
  virtual uint32_t ReadReg(int addr) = 0;
  virtual void Finish(uint32_t length) = 0;

  static tvm::runtime::Module Load(std::string dll_name);
};

}  // namespace dpi
}  // namespace vta
#endif // VTA_DPI_MODULE_H_

