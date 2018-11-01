/* cma_api.h - api library header file.
 *
 *
 * The MIT License (MIT)
 *
 * COPYRIGHT (C) 2017 Institute of Electronics and Computer Science (EDI), Latvia.
 * AUTHOR: Rihards Novickis (rihards.novickis@edi.lv)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *
 * DESCRIPTION:
 * Refer to this file for interface documentation.
 */

#ifndef CMA_API_H_
#define CMA_API_H_


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


#endif