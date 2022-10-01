#ifndef TVM_RUNTIME_CONTRIB_LIKWID_H_
#define TVM_RUNTIME_CONTRIB_LIKWID_H_

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/profiling.h>

namespace tvm {
namespace runtime {
namespace profiling {

TVM_DLL MetricCollector CreateLikwidMetricCollector(Array<DeviceWrapper> devices);

} // namespace profiling
} // namespace runtime
} // namespace tvm

#endif // TVM_RUNTIME_CONTRIB_LIKWID_H_
