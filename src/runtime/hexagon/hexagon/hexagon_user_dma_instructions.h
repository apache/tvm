// TODO: Figure out the copyright 

#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_HEXAGON_USER_DMA_INSTRUCTIONS_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_HEXAGON_USER_DMA_INSTRUCTIONS_H_

#include <stdint.h>

namespace tvm {
namespace runtime {
namespace hexagon {

inline uint32_t dmpause() {
    uint32_t status = 0;
    asm volatile (" %0 = dmpause" : "=r"(status));
    return status;
}

inline void dmstart(void *next) {
    asm volatile (" dmstart(%0)" : : "r"(next));
}

inline uint32_t dmpoll() {
    uint32_t status = 0;
    asm volatile (" %0 = dmpoll" : "=r"(status));
    return status;
}

inline uint32_t dmwait() {
    uint32_t status = 0;
    asm volatile (" %0 = dmwait" : "=r"(status));
    return status;
}

// TODO: strange that this is a void*
inline void dmresume(void *dm0) {
    asm volatile (" dmresume(%0)" : : "r"(dm0));
}

static inline uint32_t dmsyncht() {
    uint32_t status = 0;
    asm volatile (" %0 = dmsyncht" : "=r"(status));
    return status;
}

static inline uint32_t dmtlbsynch() {
    uint32_t status = 0;
    asm volatile (" %0 = dmtlbsynch" : "=r"(status));
    return status;
}

static inline uint32_t dmcfgrd(uint32_t dmindex) {
    uint32_t data = 0;
    asm volatile (" %0 = dmcfgrd(%1)" : "=r"(data) : "r"(dmindex));
    return data;
}

static inline void dmcfgwr(uint32_t dmindex, uint32_t data) {
    asm volatile (" dmcfgwr(%0, %1)" : : "r"(dmindex), "r"(data));
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif /* TVM_RUNTIME_HEXAGON_HEXAGON_HEXAGON_USER_DMA_INSTRUCTIONS_H_ */
