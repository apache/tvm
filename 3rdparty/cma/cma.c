/* cma.c - ARM specific Linux driver for allocating physically contigious memory.
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
 * In some DMA use cases there is a need or it is more efficient to use large
 * physically contigous memory regions. When Linux kernel is compiled with CMA
 * (Contigous Memory Allocator) feature, this module allows to allocate this
 * contigous memory and pass it to the user space. Memory can be either cached
 * or uncached.
 *
 * For api description, see "cma_api.h" header file.
 *
 */

/* Linux driver includes */
#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>

/* CMA specific includes */
#include <linux/dma-mapping.h>
#include <linux/dma-contiguous.h>

/* IOCTL description */
#include "cma.h"


/* Handle defines */
#ifndef CONFIG_DMA_CMA
  #error "CMA configuration not set in kernel!"
#endif
#ifndef CMA_DEBUG
  #define CMA_DEBUG       0
#endif
#ifndef DRIVER_NODE_NAME
  #define DRIVER_NODE_NAME   "cma"
#endif


/* Commonly used printk statements */
#define __ERROR(fmt, args...)    printk(KERN_ERR "CMA_ERROR: " fmt, ##args)
#define __INFO(fmt, args...)      printk(KERN_INFO "CMA_INFO: " fmt, ##args)

#if CMA_DEBUG == 1
  #define __DEBUG(fmt, args...)  printk(KERN_INFO "CMA_DEBUG: " fmt, ##args)
#else
  #define __DEBUG(fmt, args...)
#endif


/* cma entry flags */
#define CMA_ENTRY_CACHED     0
#define CMA_ENTRY_MAPPED    (1<<0)
#define CMA_ENTRY_NONCACHED   (1<<1)


/* fops declarations */
static int cma_ioctl(struct file *filp, unsigned int cmd, unsigned int arg);
static int cma_mmap(struct file *filp, struct vm_area_struct *vm_area_dscr);

/* ioctl interface commands */
static int cma_ioctl_alloc(struct file *filp, unsigned int cmd, unsigned int arg, int cached_flag);
static int cma_ioctl_free(struct file *filp, unsigned int cmd, unsigned int arg);
static int cma_ioctl_get_phy_addr(struct file *filp, unsigned int cmd, unsigned int arg);
static int cma_ioctl_get_size(struct file *filp, unsigned int cmd, unsigned int arg);

/* CMA entry specific functions */
struct cma_entry *cma_entry_get_by_phy_addr(unsigned int phy_addr);
struct cma_entry *cma_entry_get_by_v_usr_addr(unsigned v_usr_addr);
static int cma_entry_add(struct cma_entry *entry);
static int cma_entry_release(unsigned v_usr_addr);

/* mmap ops declarations */
void cma_mmap_close(struct vm_area_struct *vma);


/* File operations */
struct file_operations fops = {
  .owner         = THIS_MODULE,
  .unlocked_ioctl   = cma_ioctl,
  .mmap         = cma_mmap
};

/* mmap operation structure */
static struct vm_operations_struct cma_ops = {
  .close     = cma_mmap_close
};


/* List structure for containing memory allocation information */
struct cma_entry{
  struct cma_entry *next;
  pid_t     pid;    /* calling process id */
  unsigned   size;    /* size of allocation */
  dma_addr_t  phy_addr;  /* physical address */
  void      *v_ptr;    /* kernel-space pointer */
  unsigned   v_usr_addr; /* user-space addr */
  int     flags;    /* memory allocation related flags */
};


/* Global variables */
int major;
static struct class   *class;
static struct device   *device;
struct cma_entry     *cma_start = NULL;
static struct mutex   mutex_cma_list_modify;


struct cma_entry *cma_entry_get_by_phy_addr(unsigned int phy_addr) {
  struct cma_entry *walk = cma_start;

  __DEBUG("cma_entry_get_by_phy_addr()\n");

  if (mutex_lock_interruptible(&mutex_cma_list_modify))
    return NULL;

  /* search for physical address */
  while (walk != NULL) {
    if (walk->phy_addr == phy_addr) {
      goto leave;
    }
    walk = walk->next;
  }

leave:
  mutex_unlock(&mutex_cma_list_modify);
  return walk;
}


struct cma_entry *cma_entry_get_by_v_usr_addr(unsigned v_usr_addr) {
  struct cma_entry *walk = cma_start;

  __DEBUG("cma_entry_get_by_v_usr_addr()\n");

  if (mutex_lock_interruptible(&mutex_cma_list_modify)) {
    __DEBUG("cma_entry_get_by_v_usr_addr: failed to call mutex_lock_interruptible().\n");
    return NULL;
  }

  /* search for user virtual address */
  while (walk != NULL) {
    if (walk->v_usr_addr == v_usr_addr) {
      __DEBUG("found an entry with v_usr_addr (0x%x).\n", v_usr_addr);
      goto leave;
    }
    __DEBUG("> walk->v_usr_addr=(0x%x), expected v_usr_addr=(0x%x).\n",
            walk->v_usr_addr, v_usr_addr);
    walk = walk->next;
  }
  __DEBUG("failed to find an entry with v_usr_addr (0x%x).\n", v_usr_addr);

leave:
  mutex_unlock(&mutex_cma_list_modify);
  return walk;
}


static int cma_entry_add(struct cma_entry *entry) {
  struct cma_entry *walk;
  __DEBUG("cma_entry_add() - phy_addr 0x%x; pid 0x%x\n", entry->phy_addr, entry->pid);

  if (mutex_lock_interruptible(&mutex_cma_list_modify))
    return -EAGAIN;

  /* add entry in start - this is more effective */
  entry->next = cma_start;
  cma_start = entry;

  /* print entry list for debugging */
  walk = cma_start;
  while (walk != NULL) {
    __DEBUG("> walk->phy_addr=(0x%x).\n", walk->phy_addr);
    walk = walk->next;
  }

  mutex_unlock(&mutex_cma_list_modify);

  return 0;
}


static int cma_entry_release(unsigned v_usr_addr) {
  int err;
  struct cma_entry *walk_prev, *walk_curr;

  /* print entry list for debugging */
  struct cma_entry *walk;
  __DEBUG("cma_entry_release() - v_usr_addr 0x%x; pid 0x%x\n", v_usr_addr, current->pid);

  if (mutex_lock_interruptible(&mutex_cma_list_modify))
    return -EAGAIN;

  walk_prev = NULL;
  walk_curr = cma_start;

  while (walk_curr != NULL) {
    if (walk_curr->v_usr_addr == v_usr_addr) {
      /* check if mapped */
      if (walk_curr->flags & CMA_ENTRY_MAPPED) {
        __DEBUG("failed to find a valid entry with v_usr_addr(0x%x), entry mapped.\n", v_usr_addr);
        err = -1;
        goto leave;
      }

      /* check if not the first entry */
      if (walk_prev != NULL)
        walk_prev->next = walk_curr->next;
      else
        cma_start = walk_curr->next;
      if ((walk_curr->next == NULL) && (cma_start == walk_curr))
        cma_start = NULL;

      __DEBUG("found an entry with v_usr_addr=0x%x, phy_addr=0x%x, next=0x%x\n",
              v_usr_addr, walk_curr->phy_addr, (int)walk_curr->next);
      dma_free_coherent(NULL, walk_curr->size, walk_curr->v_ptr, walk_curr->phy_addr);
      kfree(walk_curr);
      err = 0;
      goto leave;
    }
    __DEBUG("skip entry with v_usr_addr (0x%x).\n", walk_curr->v_usr_addr);

    /* prepare next walk */
    walk_prev = walk_curr;
    walk_curr = walk_curr->next;
  }

  __DEBUG("failed to find an entry with v_usr_addr (0x%x).\n", v_usr_addr);
  err = -1;

leave:

  /* print entry list for debugging */
  walk = cma_start;
  while (walk != NULL) {
    __DEBUG("> walk->v_usr_addr=(0x%x), walk->next=0x%x\n", walk->v_usr_addr, (int)walk->next);
    walk = walk->next;
  }

  mutex_unlock(&mutex_cma_list_modify);
  return err;
}


/* inline function for readability */
inline int check_entry_accordance(struct cma_entry *entry, struct vm_area_struct *vma) {
  if ( entry == NULL )
    return -EFAULT;

  if ( entry->phy_addr != vma->vm_pgoff << PAGE_SHIFT )
    return -EFAULT;

  if ( entry->size != vma->vm_end-vma->vm_start )
    return -EFAULT;

  if ( entry->pid != current->pid )
    return -EACCES;

  return 0;
}

static int cma_mmap(struct file *filp, struct vm_area_struct *vma) {
  int err;
  struct cma_entry *entry;

  __DEBUG("cma_mmap() - phy_addr 0x%lx, v_user_addr 0x%lx\n",
          vma->vm_pgoff << PAGE_SHIFT, vma->vm_start);

  entry = cma_entry_get_by_phy_addr(vma->vm_pgoff << PAGE_SHIFT);

  /* check if mmap is alligned with according entry */
  err = check_entry_accordance(entry, vma);
  if (err) return err;

  /* set user address for later reference (used when freeing the memory ) */
  entry->v_usr_addr = vma->vm_start;

  /* should memory be uncached? */
  if ( entry->flags & CMA_ENTRY_NONCACHED )
    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

  /* map memory to user space */
  if (remap_pfn_range(vma, vma->vm_start, vma->vm_pgoff,
                      vma->vm_end-vma->vm_start, vma->vm_page_prot)) {
    up_write(&current->mm->mmap_sem);
      return -EAGAIN;
  }

  /* save mmap ops and set entry mapped flag */
  vma->vm_ops = &cma_ops;
  entry->flags = (entry->flags & (~CMA_ENTRY_MAPPED)) | CMA_ENTRY_MAPPED;

  return 0;
}


void cma_mmap_close(struct vm_area_struct *vma) {
  struct cma_entry *entry;

  __DEBUG("cma_mmap_close()\n");

  /* remove custom mapped flag */
  entry = cma_entry_get_by_phy_addr(vma->vm_pgoff << PAGE_SHIFT);
  if ( entry != NULL )
    entry->flags &= (~CMA_ENTRY_MAPPED);
}


static int cma_ioctl(struct file *filp, unsigned int cmd, unsigned int arg) {
  /* routine check */
  __DEBUG("IOCTL command issued\n");

  /* check validity of the cmd */
  if (_IOC_TYPE(cmd) != CMA_IOCTL_MAGIC) {
    __ERROR("IOCTL Incorrect magic number\n");
    return -ENOTTY;
  }
  if (_IOC_NR(cmd) > CMA_IOCTL_MAXNR) {
    __ERROR("IOCTL Command is not valid\n");
    return -ENOTTY;
  }

  /* get size from userspace */
  switch (cmd) {
    case CMA_ALLOC_CACHED:
      return cma_ioctl_alloc(filp, cmd, arg, CMA_ENTRY_CACHED);
    case CMA_ALLOC_NONCACHED:
      return cma_ioctl_alloc(filp, cmd, arg, CMA_ENTRY_NONCACHED);
    case CMA_FREE:
      return cma_ioctl_free(filp, cmd, arg);
    case CMA_GET_PHY_ADDR:
      return cma_ioctl_get_phy_addr(filp, cmd, arg);
    case CMA_GET_SIZE:
      return cma_ioctl_get_size(filp, cmd, arg);
    default:
      __DEBUG("This should never happen!\n");
  }

  return 0;
}


static int cma_ioctl_alloc(struct file *filp, unsigned int cmd, unsigned int arg, int cached_flag) {
  int err;
  struct cma_entry *entry;
  __DEBUG("cma_ioctl_alloc() called!\n");

  if (!access_ok(VERIFY_READ, (void __user*) arg, _IOC_SIZE(cmd))) {
    __DEBUG("fail to get read access to %d bytes of memory.\n", entry->size);
    return -EFAULT;
  }
  if (!access_ok(VERIFY_WRITE, (void __user*) arg, _IOC_SIZE(cmd))) {
    __DEBUG("fail to get write access to %d bytes of memory.\n", entry->size);
    return -EFAULT;
  }

  /* create new cma entry  */
  entry = kmalloc(sizeof(struct cma_entry), GFP_KERNEL);

  /* set entry params */
  __get_user(entry->size, (typeof(&entry->size))arg);
  entry->pid     = current->pid;
  entry->flags   = cached_flag;

  /* allocate contigous memory */
  entry->v_ptr = dma_alloc_coherent(NULL, entry->size, &entry->phy_addr, GFP_KERNEL);
  if ( entry->v_ptr == NULL ) {
    err = -ENOMEM;
    __DEBUG("==== FAILED TO ALLOCATE 0x%X BYTES OF COHERENT MEMORY ====\n", entry->size);
    goto error_dma_alloc_coherent;
  }

  /* add entry */
  err = cma_entry_add(entry);
  if (err)  goto error_cma_entry_add;

  /* put physical address to user space */
  __put_user(entry->phy_addr, (typeof(&entry->phy_addr))arg);

  __DEBUG("allocated 0x%x bytes of coherent memory at phy_addr=0x%x\n", entry->size, entry->phy_addr);
  return entry->phy_addr;


error_cma_entry_add:
  dma_free_coherent(NULL, entry->size, entry->v_ptr, entry->phy_addr);

error_dma_alloc_coherent:
  kfree(entry);

  return err;
}


static int cma_ioctl_free(struct file *filp, unsigned int cmd, unsigned int arg) {
  dma_addr_t v_usr_addr;
  __DEBUG("cma_ioctl_free() called!\n");

  if (!access_ok(VERIFY_READ, (void __user*) arg, _IOC_SIZE(cmd)))
    return -EFAULT;

  __get_user(v_usr_addr, (typeof(&v_usr_addr))arg);

  return cma_entry_release(v_usr_addr);
}


static struct cma_entry *cma_ioctl_get_entry_from_v_usr_addr(unsigned int cmd, unsigned int arg) {
  unsigned v_usr_addr;

  __DEBUG("cma_ioctl_get_entry_from_v_usr_addr() called!\n");

  /* routine check */
  if (!access_ok(VERIFY_READ, (void __user*) arg, _IOC_SIZE(cmd))) {
    __DEBUG("failed to get read access to virtual user address: 0x%x\n", arg);
    return NULL;
  }
  if (!access_ok(VERIFY_WRITE, (void __user*) arg, _IOC_SIZE(cmd))) {
    __DEBUG("failed to get write access to virtual user address: 0x%x\n", arg);
    return NULL;
  }

  /* get process user address */
  __get_user(v_usr_addr, (typeof(&v_usr_addr))arg);

  /* search for appropriate entry */
  return cma_entry_get_by_v_usr_addr(v_usr_addr);
}


static int cma_ioctl_get_phy_addr(struct file *filp, unsigned int cmd, unsigned int arg) {
  struct cma_entry *entry;

  __DEBUG("cma_ioctl_get_phy_addr() called!\n");

  /* get entry */
  entry = cma_ioctl_get_entry_from_v_usr_addr(cmd, arg);
  if (entry == NULL) {
    __DEBUG("cma entry has not been found.\n");
    return -EFAULT;
  }

  /* put physical address into user space */
  __put_user(entry->phy_addr, (typeof(&entry->phy_addr))arg);

  return 0;
}


static int cma_ioctl_get_size(struct file *filp, unsigned int cmd, unsigned int arg) {
  struct cma_entry *entry;

  __DEBUG("cma_ioctl_get_size() called!\n");

  /* get entry */
  entry = cma_ioctl_get_entry_from_v_usr_addr(cmd, arg);
  if (entry == NULL) {
    __DEBUG("cma_ioctl_get_size: failed to get_entry_from_v_usr_addr.\n");
    return -EFAULT;
  }

  /* put size into user space */
  __put_user(entry->size, (typeof(&entry->size))arg);

  return 0;
}


static int cma_init(void) {
  int err;
  __INFO("Initializeing Contigous Memory Allocator module\n");

  /* obtain major number */
  major = register_chrdev(0, DRIVER_NODE_NAME, &fops);
  if ( major < 0 ) {
    __ERROR("Failed to allocate major number\n");
    return -major;
  }

  /* create class */
  class = class_create(THIS_MODULE, DRIVER_NODE_NAME);
  if ( IS_ERR(class) ) {
    __ERROR("Failed to create class\n");
    err = PTR_ERR(class);
    goto error_class_create;
  }

  /* create device node */
  device = device_create(class, NULL, MKDEV(major, 0), NULL, DRIVER_NODE_NAME);
  if ( IS_ERR(device) ) {
    __ERROR("Failed to create device\n");
    err = PTR_ERR(device);
    goto error_device_create;
  }

  mutex_init(&mutex_cma_list_modify);

  return 0;


error_device_create:
  class_destroy(class);

error_class_create:
  unregister_chrdev(major, DRIVER_NODE_NAME);

  return err;
}


static void cma_exit(void) {
  __INFO("Releasing Contigous Memory Allocator module\n");

  /* TODO: walk_list_remove_pid */

  device_destroy(class, MKDEV(major, 0));

  class_destroy(class);

  unregister_chrdev(major, DRIVER_NODE_NAME);
}


MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Driver for allocating cached and noncached physically contigous memory. "
                   "Exploits kernel CMA feature.");
module_init(cma_init);
module_exit(cma_exit);
