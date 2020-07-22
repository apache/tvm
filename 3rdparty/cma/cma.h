/* cma.h
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
 */

#ifndef VTA_DE10_NANO_KERNEL_MODULE_CMA_H_
#define VTA_DE10_NANO_KERNEL_MODULE_CMA_H_

/* Should be defined in settings.mk file */
#ifndef CMA_IOCTL_MAGIC
#define CMA_IOCTL_MAGIC 0xf2
#endif

#define CMA_ALLOC_CACHED _IOC(_IOC_WRITE | _IOC_READ, CMA_IOCTL_MAGIC, 1, 4)
#define CMA_ALLOC_NONCACHED _IOC(_IOC_WRITE | _IOC_READ, CMA_IOCTL_MAGIC, 2, 4)
#define CMA_FREE _IOC(_IOC_WRITE, CMA_IOCTL_MAGIC, 3, 4)
#define CMA_GET_PHY_ADDR _IOC(_IOC_WRITE | _IOC_READ, CMA_IOCTL_MAGIC, 4, 4)
#define CMA_GET_SIZE _IOC(_IOC_WRITE | _IOC_READ, CMA_IOCTL_MAGIC, 5, 4)

#define CMA_IOCTL_MAXNR 5

#endif  // VTA_DE10_NANO_KERNEL_MODULE_CMA_H_
