/*!
 *  Copyright (c) 2018 by Contributors
 * \file cma_api.h
 * \brief API for contigous memory allocation driver.
 */

#ifndef VTA_DE10_NANO_CMA_API_H_
#define VTA_DE10_NANO_CMA_API_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize CMA api (basically perform open() syscall).
 * 
 * @return Returns 0 on SUCCESS. On FAILURE returns -1 and errno is set
 * accordingly.
 */
int cma_init(void);


/**
 * @brief Release CMA api (basically perform close() syscall).
 * 
 * @return Returns 0 on SUCCESS. On FAILURE returns -1 and errno is set 
 * accordingly.
 */
int cma_release(void);


/**
 * @brief Allocate cached, physically contigous memory.
 *
 * @param size Size in bytes.
 *
 * @return Returns NULL on FAILURE. Otherwise pointer to valid userspace
 * memory.
 */
void *cma_alloc_cached(size_t size);


/**
 * @brief Allocate noncached, physically contigous memory.
 *
 * @param size Size in bytes.
 *
 * @return Returns NULL on FAILURE. Otherwise pointer to valid userspace
 * memory.
 */
void *cma_alloc_noncached(size_t size);


/**
 * @brief Release physically contigous memory.
 *
 * @param mem Pointer to previously allocated contiguous memory.
 *
 * @return Returns 0 on SUCCESS, -1 on FAILURE.
 */
int cma_free(void *mem);


/**
 * @brief Get physical memory of cma memory block (should be used for DMA).
 *
 * @param mem Pointer to previously allocated contiguous memory.
 *
 * @return Returns address on SUCCESS, 0 on FAILURE.
 */
unsigned cma_get_phy_addr(void *mem);


#ifdef __cplusplus
}
#endif
#endif  // VTA_DE10_NANO_CMA_API_H_
