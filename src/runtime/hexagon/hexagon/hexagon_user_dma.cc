#include "hexagon_user_dma_descriptors.h"
#include "hexagon_user_dma_instructions.h"
#include "hexagon_user_dma_regs.h"
#include <algorithm>

namespace tvm {
namespace runtime {
namespace hexagon {

int hexagon_user_dma_wrapper(void *src, void *dst, uint32_t length) {
    //TODO: Configure DMA Regs at global level and make it thread safe 
    static int config_dma = 0;
    if (!config_dma) {
        // TODO: needed for qurt?
        // dma_config();
        uint32_t status = dmpause();
        if ((status & DM0_STATUS_MASK) != DM0_STATUS_IDLE) {
            return DMA_FAILURE;
        }
        config_dma = 1;
    }

    void *dma_desc = nullptr;
#ifdef _WIN32
    dma_desc = _aligned_malloc(DMA_DESC_2D_SIZE, DMA_DESC_2D_SIZE);
#else
    int ret = posix_memalign(&dma_desc, DMA_DESC_2D_SIZE, DMA_DESC_2D_SIZE);
    if(ret) {
        return DMA_FAILURE;
    }
#endif

    if(!dma_desc) {
        return DMA_FAILURE;
    }

    // TODO: this seems wrong
    uint64_t src_uint64_t = reinterpret_cast<uint64_t>(src);
    uint64_t dst_uint64_t = reinterpret_cast<uint64_t>(dst);
    uint32_t set_src = src_uint64_t & 0xFFFFFFFF;
    uint32_t set_dst = dst_uint64_t & 0xFFFFFFFF;
    // TODO: check for length > 24 bits

    // dword[0]
    dma_desc_set_next(dma_desc, DMA_NULL_PTR);
    // dword[1][31]
    dma_desc_set_dstate(dma_desc, DESC_DSTATE_INCOMPLETE);
    // dword[1][30]
    dma_desc_set_order(dma_desc, DESC_ORDER_ORDER);
    // dword[1][29]
    dma_desc_set_bypasssrc(dma_desc, DESC_BYPASS_OFF);
    // dword[1][28]
    dma_desc_set_bypassdst(dma_desc, DESC_BYPASS_OFF);
    // dword[1][27]
    dma_desc_set_srccomp(dma_desc, DESC_COMP_NONE);
    // dword[1][26]
    dma_desc_set_dstcomp(dma_desc, DESC_COMP_NONE);
    // dword[1][25:24]
    dma_desc_set_desctype(dma_desc, DESC_DESCTYPE_1D);
    // dword[1][23:0]
    dma_desc_set_length(dma_desc, length);
    // dword[2]
    dma_desc_set_src(dma_desc, set_src);
    // dword[3]
    dma_desc_set_dst(dma_desc, set_dst);

    dmstart(dma_desc);

    uint32_t done = dma_desc_get_dstate(dma_desc);
    uint32_t status = dmwait();
    if ((status & DM0_STATUS_MASK) == DM0_STATUS_ERROR) {
        dmpause();
    }

#ifdef _WIN32
    _aligned_free(dma_desc);
#else
    free(dma_desc);
#endif

    if (((status & DM0_STATUS_MASK) == DM0_STATUS_IDLE) &&
        (((done & DESC_DSTATE_MASK) >> DESC_DSTATE_SHIFT) == DESC_DSTATE_COMPLETE)) {
        return DMA_SUCCESS;
    }
    return DMA_FAILURE;
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
