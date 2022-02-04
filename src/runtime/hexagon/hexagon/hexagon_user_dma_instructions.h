#include "HalideRuntimeHexagonUserDMA.h"
#include "device_buffer_utils.h"
#include "device_interface.h"
#include "hexagon_user_dma_regs.h"
#include "mini_hexagon_user_dma.h"
#include "printer.h"
#include "runtime_internal.h"

namespace Halide {
namespace Runtime {
namespace Internal {
namespace HexagonUserDMA {

extern WEAK halide_device_interface_t hexagon_user_dma_device_interface;

#define descriptor_size 32

}  // namespace HexagonUserDma
}  // namespace Internal
}  // namespace Runtime
}  // namespace Halide

using namespace Halide::Runtime::Internal::HexagonUserDMA;

namespace {

inline unsigned int dmpause(void *user_context) {
    unsigned int ret = 0;
    debug(user_context) << "Hexagon User DMA: trying dmpause" << "\n";
    asm volatile (" %0 = dmpause" : "=r"(ret));
    debug(user_context) << "Hexagon User DMA: dmpause completed" << "\n";
    return ret;
}

void dma_clear_error(void *user_context)
{
    debug(user_context) << "Hexagon User DMA: Clear DMA error.\n";
#ifdef DEBUG_RUNTIME
    dma_dump_state(user_context);
#endif
    dmpause(user_context);
}


inline uint32_t dma_desc_get_dstate(void *user_context, void *d)
{
    return (((((dma_desc_1d_t*)d)->dstate_order_bypass_comp_desctype_length) & DESC_DSTATE_MASK) >> DESC_DSTATE_SHIFT);
}

inline void dmstart(void *user_context, void * next) {
    asm volatile (" dmstart(%0)" : : "r"(next));
    debug(user_context) << "Hexagon User DMA: dmstart completed" << "\n";
}

inline unsigned int dmpoll(void *user_context) {
    unsigned int ret = 0;
    asm volatile (" %0 = dmpoll" : "=r"(ret));
    if ((ret & DM0_STATUS_MASK) == DM0_STATUS_ERROR)
        dma_clear_error(user_context);
    debug(user_context) << "Hexagon User DMA: dmpoll completed" << "\n";
    return ret;
}

inline unsigned int dmwait(void *user_context) {
    unsigned int ret = 0;
    debug(user_context) << "Hexagon User DMA: trying dmwait" << "\n";
    asm volatile (" %0 = dmwait" : "=r"(ret));
    debug(user_context) << "Hexagon User DMA: dmwait completed" << "\n";
    if ((ret & DM0_STATUS_MASK) == DM0_STATUS_ERROR)
        dma_clear_error(user_context);
    return ret;
}

inline void dmresume(void *user_context, void * dm0) {
    asm volatile (" dmresume(%0)" : : "r"(dm0));
    debug(user_context) << "Hexagon User DMA: dmresume completed" << "\n";
}

void dma_pause_and_try_resume(void *user_context, void *desc)
{
    uint32_t dm0;

    if (!desc)
        return;

    debug(user_context) << "Hexagon User DMA: Inside dma_pause_and_try_resume" << "\n";

    // Halt engine and set to idle state
    dm0 = dmpause(user_context);

    debug(user_context) << "Hexagon User DMA: dmpause completed inside dma_pause_and_try_resume" << "\n";

    if ((dm0 & DM0_STATUS_MASK) != DM0_STATUS_IDLE) {
        if (dma_desc_get_dstate(user_context, (void *) desc) == DESC_DSTATE_INCOMPLETE) {
            dmresume(user_context, (void *)(uintptr_t) (dm0));
            debug(user_context) << "Hexagon User DMA: dmresume completed inside dma_paus_and_try_resume" << "\n";
        }
    }

    return;
}

void* halide_hexagon_user_dma_get_desc(void *user_context, unsigned int elem_size, unsigned int num_elem) {
    unsigned int size = (elem_size * num_elem) + ALIGN_SIZE(ALIGN32);
    void* dma_desc_ptr = halide_malloc(user_context, size);
    if (!dma_desc_ptr) {
        error(user_context) << "DMA descriptor alloc failed in halide_hexagon_user_dma_get_desc" << "\n";
    } else {
        debug(user_context) << "DMA descriptor alloc success in halide_hexagon_user_dma_get_desc" <<"\n";
    }
    return dma_desc_ptr;
}

void halide_hexagon_user_dma_free_desc(void *user_context, void *dma_desc_ptr) {
    if(dma_desc_ptr)
        halide_free(user_context, dma_desc_ptr);
}

void* halide_hexagon_user_dma_alloc_desc_and_set_value(void* user_context, unsigned int num_elem,
                                                       unsigned int set_src, unsigned int set_dst,
                                                       unsigned int roi_width, unsigned int roi_height,
                                                       unsigned int src_stride, unsigned int dst_stride,
                                                       unsigned int src_width_offset, unsigned int dst_width_offset) {
    void* dma_desc = halide_hexagon_user_dma_get_desc(user_context, DMA_DESC_2D_SIZE, num_elem);
    debug(user_context) << "Hexagon User DMA: user_dma_alloc_desc_and_set_value returned: " << dma_desc << "\n";

    if (!dma_desc)
        return dma_desc;

    dma_desc_set_next(dma_desc, DMA_NULL_PTR);
    dma_desc_set_src(dma_desc, set_src);
    dma_desc_set_dst(dma_desc, set_dst);
    dma_desc_set_desctype(dma_desc, DESC_DESCTYPE_2D);
    dma_desc_set_dstate(dma_desc, DESC_DSTATE_INCOMPLETE);
    dma_desc_set_roiwidth(dma_desc, roi_width);
    dma_desc_set_roiheight(dma_desc, roi_height);
    dma_desc_set_srcstride(dma_desc, src_stride);
    dma_desc_set_dststride(dma_desc, dst_stride);
    dma_desc_set_srcwidthoffset(dma_desc, src_width_offset);
    dma_desc_set_dstwidthoffset(dma_desc, dst_width_offset);
    dma_desc_set_padding(dma_desc, 0);
    dma_desc_set_cachealloc(dma_desc, DESC_CACHEALLOC_NONE);
    dma_desc_set_bypasssrc(dma_desc, CACHE_NO_BYPASS);
    dma_desc_set_bypassdst(dma_desc, CACHE_NO_BYPASS);
    dma_desc_set_srccomp(dma_desc, DESC_COMP_NONE);
    dma_desc_set_dstcomp(dma_desc, DESC_COMP_NONE);

    return dma_desc;
}

static inline unsigned int dmsyncht(void *user_context) {
    unsigned int ret = 0;
    asm volatile (" %0 = dmsyncht" : "=r"(ret));
    if ((ret & DM0_STATUS_MASK) == DM0_STATUS_ERROR)
        dma_clear_error(user_context);
    return ret;
}

static inline unsigned int dmtlbsynch(void *user_context) {
    unsigned int ret = 0;
    asm volatile (" %0 = dmtlbsynch" : "=r"(ret));
    if ((ret & DM0_STATUS_MASK) == DM0_STATUS_ERROR)
        dma_clear_error(user_context);
    return ret;
}

int dma_reset(void *user_context) {

    debug(user_context) << "Hexagon User DMA: Resetting DMA Engine." << "\n";
    int result = halide_error_code_success;
    uint32_t status = 0;

    // Stop DMA engine, clear error state, and set to idle state
    status = dmpause(user_context);

    if ((status & DM0_STATUS_MASK) != DM0_STATUS_IDLE) {
        result = halide_hexagon_user_dma_reset_fail;
    }
    debug(user_context) << "Hexagon User DMA: DMA Engine reset done." << "\n";
    return result;
}

int halide_hexagon_user_dma_wrapper(void *user_context, struct halide_buffer_t *src,
                                    struct halide_buffer_t *dst) {

    debug(user_context) << "Hexagon User DMA: Inside user_dma_wrapper" << "\n";

/* This is yet not thread safe. UserDMA over multiple threads is not supported yet
    TODO: Configure DMA Regs at global level and make it thread safe */

    static int config_dma = 0;
    if (!config_dma) {
        dma_config(user_context);
        dma_reset(user_context);
        debug(user_context) << "Hexagon User DMA: DMA reset done." << "\n";
#ifdef DEBUG_RUNTIME
        dma_dump_state(user_context);
#endif
        config_dma = 1;
    }

    // Since user dma itself doesn't do a byte by byte copy and is capable of
    // understanding that stride and width can be different, we ignore all the
    // other 'optimizations' (see src_stride_bytes, dst_stride_bytes in device_copy structure)
    // that make_buffer_copy does. We use make_buffer_copy solely to figure out the right
    // value of src_begin.
    device_copy c = make_buffer_copy(src, true, dst, true);

    unsigned int roi_width = dst->dim[0].extent * dst->dim[0].stride * dst->type.bytes();
    unsigned int roi_height = dst->dim[1].extent;

    unsigned int src_stride = src->dim[1].stride * src->type.bytes();
    unsigned int dst_stride = dst->dim[1].stride * dst->type.bytes();
    // c.src_begin is the offset (in bytes) from src->host that we need to do the copy
    // from.
    uint8_t *src_data_ptr = src->host + c.src_begin;
    uint64_t src_uint64_t = reinterpret_cast<uint64_t>(src_data_ptr);
    uint64_t dst_uint64_t = reinterpret_cast<uint64_t>(dst->host);
    uint32_t set_src = src_uint64_t & 0xFFFFFFFF;
    uint32_t set_dst = dst_uint64_t & 0xFFFFFFFF;

    uint32_t result = halide_error_code_generic_error;
    debug(user_context) << "Hexagon User DMA: roi_width: " << roi_width
                        << " roi_height: " << roi_height
                        << " src_stride: " << src_stride
                        << " dst_stride: " << dst_stride
                        << " set_src: " << set_src
                        << " set_dst: " << set_dst << "\n";

    void* dma_desc = halide_hexagon_user_dma_alloc_desc_and_set_value(user_context, 1,
                                                                      set_src, set_dst,
                                                                      roi_width, roi_height,
                                                                      src_stride, dst_stride,
                                                                      src->dim[0].min, dst->dim[0].min);

    if(!dma_desc)
        return halide_hexagon_user_dma_alloc_desc_and_set_value_fail;

    dmstart(user_context, dma_desc);

    dmwait(user_context);

    if (dma_desc_get_dstate(user_context, dma_desc) != DESC_DSTATE_COMPLETE)
        result = halide_hexagon_user_dma_fail;
    else
        result = halide_error_code_success;

    debug(user_context) << "Hexagon User DMA: result = " << result << "\n";

    if(dma_desc)
        halide_hexagon_user_dma_free_desc(user_context, dma_desc);
    return result;

}

} // namespace

extern "C" {

WEAK int halide_hexagon_user_dma_device_malloc(void *user_context, halide_buffer_t *buf) {
    debug(user_context)
        << "Hexagon: halide_hexagon_user_dma_malloc (user_context: " << user_context
        << ", buf: " << *buf << ")\n";

    if (buf->device) {
        debug(user_context) << "Hexagon: buffer already has a device. No action required\n";
        return halide_error_code_success;
    }

    size_t size = buf->size_in_bytes();
    halide_assert(user_context, size != 0);

    void *mem = halide_malloc(user_context, size);
    if (!mem) {
        error(user_context) << "Hexagon: Out of memory (halide_malloc failed in halide_hexagon_user_dma_device_malloc)\n";
        return halide_error_code_out_of_memory;
    }

    int err = halide_hexagon_user_dma_device_wrap_native(user_context, buf,
                                                         reinterpret_cast<uint64_t>(mem));
    if (err != halide_error_code_success) {
        halide_free(user_context, mem);
        return halide_error_code_device_malloc_failed;
    }

    return halide_error_code_success;
}

WEAK int halide_hexagon_user_dma_device_free(void *user_context, halide_buffer_t *buf) {
    debug(user_context)
        << "Hexagon: halide_hexagon_user_dma_free (user_context: " << user_context
        << ", buf: " << *buf << ")\n";

    void *mem = reinterpret_cast<void*>(buf->device);
    halide_hexagon_user_dma_device_detach_native(user_context, buf);

    halide_free(user_context, mem);
    buf->set_device_dirty(false);

    return halide_error_code_success;;
}

/* This is the entry function for halide_hexagon_user_dma runtime.*/
WEAK int halide_hexagon_user_dma_buffer_copy(void *user_context, struct halide_buffer_t *src,
                                             const struct halide_device_interface_t *dst_device_interface,
                                             struct halide_buffer_t *dst) {
    debug(user_context) << "Hexagon User DMA: Inside dma_buffer_copy" << "\n";
    halide_assert(user_context, dst_device_interface == nullptr ||
                  dst_device_interface == &hexagon_user_dma_device_interface);


    int nRet = halide_hexagon_user_dma_wrapper(user_context, src, dst);
    debug(user_context) << "Hexagon User DMA: user_dma_wrapper returned : " << nRet << "\n";
    return nRet;
}

WEAK int halide_hexagon_user_dma_copy_to_device(void *user_context, halide_buffer_t *buf) {
    debug(user_context)
        << "Hexagon: halide_hexagon_user_dma_copy_to_device (user_context: " << user_context
        << ", buf: " << *buf << ")\n";

    error(user_context) << "Hexagon: halide_hexagon_user_dma_copy_to_device not implemented\n";
    return halide_error_code_copy_to_device_failed;
}

WEAK int halide_hexagon_user_dma_copy_to_host(void *user_context, struct halide_buffer_t *buf) {
    debug(user_context)
        << "Hexagon: halide_hexagon_user_dma_copy_to_host (user_context: " << user_context
        << ", buf: " << *buf << ")\n";

    error(user_context) << "Hexagon: halide_hexagon_user_dma_copy_to_host not implemented\n";
    return halide_error_code_copy_to_host_failed;
}

WEAK int halide_hexagon_user_dma_device_crop(void *user_context,
                                             const struct halide_buffer_t *src,
                                             struct halide_buffer_t *dst) {
    debug(user_context)
        << "Hexagon: halide_hexagon_user_dma_device_crop (user_context: " << user_context
        << ", buf: " << *src << ")\n";

    error(user_context) << "Hexagon: halide_hexagon_user_dma_device_crop not implemented\n";
    return halide_error_code_device_crop_failed;
}

WEAK int halide_hexagon_user_dma_device_slice(void *user_context,
                                              const struct halide_buffer_t *src,
                                              int slice_dim, int slice_pos, struct halide_buffer_t *dst) {
    debug(user_context)
        << "Hexagon: halide_hexagon_user_dma_device_slice (user_context: " << user_context
       << ", buf: " << *src << ")\n";

    error(user_context) << "Hexagon: halide_hexagon_user_dma_device_slice not implemented\n";
    return halide_error_code_generic_error;
}

WEAK int halide_hexagon_user_dma_device_release_crop(void *user_context, struct halide_buffer_t *buf) {
    debug(user_context)
        << "Hexagon: halide_hexagon_user_dma_device_release_crop (user_context: " << user_context
        << ", buf: " << *buf << ")\n";

    error(user_context) << "Hexagon: halide_hexagon_user_dma_device_release_crop not implemented\n";
    return halide_error_code_generic_error;
}

WEAK int halide_hexagon_user_dma_device_sync(void *user_context, struct halide_buffer_t *buf) {
    debug(user_context)
        << "Hexagon: halide_hexagon_user_dma_device_sync (user_context: " << user_context
        << ", buf: " << *buf << ")\n";

    error(user_context) << "Hexagon: halide_hexagon_user_dma_device_sync not implemented\n";
    return halide_error_code_device_sync_failed;
}

WEAK int halide_hexagon_user_dma_device_wrap_native(void *user_context, struct halide_buffer_t *buf,
                                                    uint64_t handle) {
    debug(user_context)
        << "Hexagon: halide_hexagon_user_dma_device_wrap_native (user_context: " << user_context
        << ", buf: " << *buf << ")\n";

    halide_assert(user_context, buf->device == 0);
    if (buf->device != 0) {
        error(user_context) << "Hexagon: halide_hexagon_user_dma_device_wrap_native buffer already has a device\n";
        return halide_error_code_device_wrap_native_failed;
    }

    buf->device_interface = &hexagon_user_dma_device_interface;
    buf->device = handle;
    buf->device_interface->impl->use_module();
    return halide_error_code_success;
}

WEAK int halide_hexagon_user_dma_device_detach_native(void *user_context, struct halide_buffer_t *buf) {
    debug(user_context)
        << "Hexagon: halide_hexagon_user_dma_device_detach_native (user_context: " << user_context
        << ", buf: " << *buf << ")\n";

    if (buf->device == 0) {
        error(user_context) << "Hexagon: halide_hexagon_user_dma_device_detach_native buffer without a device\n";
        return halide_error_code_device_detach_native_failed;
    }

    halide_assert(user_context, buf->device_interface == &hexagon_user_dma_device_interface);
    buf->device_interface->impl->release_module();
    buf->device = 0;
    buf->device_interface = nullptr;

    return halide_error_code_success;
}

WEAK int halide_hexagon_user_dma_device_and_host_malloc(void *user_context, struct halide_buffer_t *buf) {
    debug(user_context)
        << "Hexagon: halide_hexagon_dma_device_and_host_malloc (user_context: " << user_context
        << " buf: " << *buf << ")\n";

    return halide_default_device_and_host_malloc(user_context, buf, &hexagon_user_dma_device_interface);
}

WEAK int halide_hexagon_user_dma_device_and_host_free(void *user_context, struct halide_buffer_t *buf) {
    debug(user_context)
        << "Hexagon: halide_hexagon_dma_device_and_host_free (user_context: " << user_context
        << " buf: " << *buf << ")\n";

    return halide_default_device_and_host_free(user_context, buf, &hexagon_user_dma_device_interface);
}

WEAK const halide_device_interface_t *halide_hexagon_user_dma_device_interface() {
    return &hexagon_user_dma_device_interface;
}

WEAK int halide_hexagon_user_dma_device_release(void *user_context) {
    debug(user_context)
        << "Hexagon: halide_hexagon_dma_device_release (user_context: " << user_context << ")\n";

    return 0;
}

}  // extern "C" linkage

namespace Halide {
namespace Runtime {
namespace Internal {
namespace HexagonUserDMA {

WEAK halide_device_interface_impl_t hexagon_user_dma_device_interface_impl = {
    halide_use_jit_module,
    halide_release_jit_module,
    halide_hexagon_user_dma_device_malloc,
    halide_hexagon_user_dma_device_free,
    halide_hexagon_user_dma_device_sync,
    halide_hexagon_user_dma_device_release,
    halide_hexagon_user_dma_copy_to_host,
    halide_hexagon_user_dma_copy_to_device,
    halide_hexagon_user_dma_device_and_host_malloc,
    halide_hexagon_user_dma_device_and_host_free,
    halide_hexagon_user_dma_buffer_copy,
    halide_hexagon_user_dma_device_crop,
    halide_hexagon_user_dma_device_slice,
    halide_hexagon_user_dma_device_release_crop,
    halide_hexagon_user_dma_device_wrap_native,
    halide_hexagon_user_dma_device_detach_native,
};

WEAK halide_device_interface_t hexagon_user_dma_device_interface = {
    halide_device_malloc,
    halide_device_free,
    halide_device_sync,
    halide_device_release,
    halide_copy_to_host,
    halide_copy_to_device,
    halide_device_and_host_malloc,
    halide_device_and_host_free,
    halide_buffer_copy,
    halide_device_crop,
    halide_device_slice,
    halide_device_release_crop,
    halide_device_wrap_native,
    halide_device_detach_native,
    nullptr,
    &hexagon_user_dma_device_interface_impl};

}  // namespace HexagonUserDMA
}  // namespace Internal
}  // namespace Runtime
}  // namespace Halide
