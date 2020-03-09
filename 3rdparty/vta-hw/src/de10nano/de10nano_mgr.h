/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 * \file de10nano_mgr.h
 * \brief DE10-Nano fpga manager.
 */

#ifndef VTA_DE10NANO_DE10NANO_MGR_H_
#define VTA_DE10NANO_DE10NANO_MGR_H_

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/time.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Register definition and address map taken from cv_5v4.pdf,
// Cyclone V Hard Processor System Technical Reference Manual,
// chapter 5: FPGA Manager.
struct De10NanoMgr {
  // Reg32 is a static base class interface and implementation
  // of a generic 32 bit register that avoids the use of a virtual
  // class and ugly bit shift manipulations.
  struct Reg32 {
    explicit Reg32(uint32_t offset, uint32_t reset = 0) :
      m_offset(offset),
      m_reset(reset)
    {}
    void map(uint8_t *base) {
      m_addr = reinterpret_cast<uint32_t*>(base + m_offset);
      m_reg  = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(this)+sizeof(Reg32));
    }
    uint32_t read() {
      *m_reg = *m_addr;
      return *m_reg;
    }
    void write() { *m_addr = *m_reg; }
    void write(uint32_t value) { *m_addr = *m_reg = value; }
    void clear() { *m_reg = 0; }
    void reset() { *m_reg = m_reset; }
    void print(const char *name, bool addr = false) {
      if (addr)
        printf("DE10-Nano-Mgr: %16s: 0x%08x addr: %p\n", name, read(), m_addr);
      else
        printf("DE10-Nano-Mgr: %16s: 0x%08x\n", name, read());
    }

    uint32_t m_offset, m_reset, *m_reg;
    volatile uint32_t *m_addr;

   private:  // Do not use this class on its own.
    Reg32(const Reg32 &rhs);
  };

  // Register definitions. All registers are of 32 bit size.
  // Add one structure for each register, making sure that all
  // bit fields come first and pack exactly into 32 bits.

  struct data : public Reg32 {
    data() : Reg32(0x0, 0x0) {}
    uint32_t value;
  } data;

  struct stat : public Reg32 {
    stat() : Reg32(0x0, 0x45) {}
    enum mode_values {
      FPGA_POWER_OFF    = 0x0,
      FPGA_RESET_PHASE  = 0x1,
      FPGA_CONFIG_PHASE = 0x2,
      FPGA_INIT_PHASE   = 0x3,
      FPGA_USER_MODE    = 0x4,
      FPGA_ZOMBIE_MODE  = 0x5
    };

    enum msel_values {
      FPP16_AESN_ZIPN = 0x0,
      FPP32_AESO_ZIPY = 0xA
    };

    const char * mode_str() {
      const char *str = "UNKNOWN";
      switch (mode) {
        case FPGA_POWER_OFF    : str = "POWER_OFF"    ; break;
        case FPGA_RESET_PHASE  : str = "RESET_PHASE"  ; break;
        case FPGA_CONFIG_PHASE : str = "CONFIG_PHASE" ; break;
        case FPGA_INIT_PHASE   : str = "INIT_PHASE"   ; break;
        case FPGA_USER_MODE    : str = "USER_MODE"    ; break;
        case FPGA_ZOMBIE_MODE  : str = "UNDEF_MODE"   ; break;
      }
      return str;
    }

    bool msel_is_invalid() {
      return msel & 0x10 || (msel & 0x3) == 0x3;
    }

    void print(bool addr = false, bool fields = true) {
      Reg32::print("stat", addr);
      if (fields) {
        printf("DE10-Nano-Mgr: %16s: %x\n", "msel", msel);
        printf("DE10-Nano-Mgr: %16s: %s\n", "mode", mode_str());
      }
    }

    uint32_t mode :  3;  //  2:0 RW
    uint32_t msel :  5;  //  7:3 RO
    uint32_t rsvd : 24;  // 31:8
  } stat;

  struct ctrl : public Reg32 {
    ctrl() : Reg32(0x4, 0x200) {}

    uint32_t           en :  1;  //     0 RW
    uint32_t          nce :  1;  //     1 RW
    uint32_t  nconfigpull :  1;  //     2 RW
    uint32_t  nstatuspull :  1;  //     3 RW
    uint32_t confdonepull :  1;  //     4 RW
    uint32_t        prreq :  1;  //     5 RW
    uint32_t      cdratio :  2;  //   7:6 RW
    uint32_t     axicfgen :  1;  //     8 RW
    uint32_t      cfgwdth :  1;  //     9 RW
    uint32_t         rsvd : 22;  // 31:10

    void print(bool addr = false, bool fields = true) {
      Reg32::print("ctrl", addr);
      if (fields) {
        printf("DE10-Nano-Mgr: %16s: %x\n", "en"          , en);
        printf("DE10-Nano-Mgr: %16s: %x\n", "nce"         , nce);
        printf("DE10-Nano-Mgr: %16s: %x\n", "nconfigpull" , nconfigpull);
        printf("DE10-Nano-Mgr: %16s: %x\n", "nstatuspull" , nstatuspull);
        printf("DE10-Nano-Mgr: %16s: %x\n", "confdonepull", confdonepull);
        printf("DE10-Nano-Mgr: %16s: %x\n", "prreq"       , prreq);
        printf("DE10-Nano-Mgr: %16s: %x\n", "cdratio"     , cdratio);
        printf("DE10-Nano-Mgr: %16s: %x\n", "axicfgen"    , axicfgen);
        printf("DE10-Nano-Mgr: %16s: %x\n", "cfgwdth"     , cfgwdth);
      }
    }
  } ctrl;

  struct dclkcnt : public Reg32 {
    dclkcnt() : Reg32(0x8, 0x0) {}
    void print() { return Reg32::print("dclkcnt"); }

    uint32_t cnt;  // RW
  } dclkcnt;

  struct dclkstat : public Reg32 {
    dclkstat() : Reg32(0xC, 0x0) {}
    void print() { return Reg32::print("dclkstat"); }

    uint32_t dcntdone :  1;  // RW
    uint32_t     rsvd : 31;
  } dclkstat;

  struct gpio_inten : public Reg32 {
    gpio_inten() : Reg32(0x830, 0x0) {}
    void print() { return Reg32::print("gpio_inten"); }

    uint32_t    value : 32;  // RW
  } gpio_inten;

  struct gpio_porta_eoi : public Reg32 {
    gpio_porta_eoi() : Reg32(0x84C, 0x0) {}
    void print() { return Reg32::print("gpio_porta_eoi"); }

    uint32_t   ns :  1;  //     0 WO
    uint32_t   cd :  1;  //     1 WO
    uint32_t   id :  1;  //     2 WO
    uint32_t  crc :  1;  //     3 WO
    uint32_t  ccd :  1;  //     4 WO
    uint32_t  prr :  1;  //     5 WO
    uint32_t  pre :  1;  //     6 WO
    uint32_t  prd :  1;  //     7 WO
    uint32_t  ncp :  1;  //     8 WO
    uint32_t  nsp :  1;  //     9 WO
    uint32_t  cdp :  1;  //    10 WO
    uint32_t  fpo :  1;  //    11 WO
    uint32_t rsvd : 20;  // 31:12
  } gpio_porta_eoi;

  struct gpio_ext_porta : public Reg32 {
    gpio_ext_porta() : Reg32(0x850, 0x0) {}
    void print(bool addr = false, bool fields = true) {
      Reg32::print("gpio_ext_porta", addr);
      if (fields) {
        printf("DE10-Nano-Mgr: %16s: %x\n", "nSTATUS"       , ns);
        printf("DE10-Nano-Mgr: %16s: %x\n", "CONF_DONE"     , cd);
        printf("DE10-Nano-Mgr: %16s: %x\n", "INIT_DONE"     , id);
        printf("DE10-Nano-Mgr: %16s: %x\n", "CRC_ERROR"     , crc);
        printf("DE10-Nano-Mgr: %16s: %x\n", "CVP_CONF_DONE" , ccd);
        printf("DE10-Nano-Mgr: %16s: %x\n", "PR_READY"      , prr);
        printf("DE10-Nano-Mgr: %16s: %x\n", "PR_ERROR"      , pre);
        printf("DE10-Nano-Mgr: %16s: %x\n", "PR_DONE"       , prd);
        printf("DE10-Nano-Mgr: %16s: %x\n", "nCONFIG_PIN"   , ncp);
        printf("DE10-Nano-Mgr: %16s: %x\n", "nSTATUS_PIN"   , nsp);
        printf("DE10-Nano-Mgr: %16s: %x\n", "CONF_DONE_PIN" , cdp);
        printf("DE10-Nano-Mgr: %16s: %x\n", "FPGA_POWER_ON" , fpo);
      }
    }

    uint32_t   ns :  1;  //     0 RO
    uint32_t   cd :  1;  //     1 RO
    uint32_t   id :  1;  //     2 RO
    uint32_t  crc :  1;  //     3 RO
    uint32_t  ccd :  1;  //     4 RO
    uint32_t  prr :  1;  //     5 RO
    uint32_t  pre :  1;  //     6 RO
    uint32_t  prd :  1;  //     7 RO
    uint32_t  ncp :  1;  //     8 RO
    uint32_t  nsp :  1;  //     9 RO
    uint32_t  cdp :  1;  //    10 RO
    uint32_t  fpo :  1;  //    11 RO
    uint32_t rsvd : 20;  // 31:12
  } gpio_ext_porta;

  struct monitor {
    // This is used to both break a polling loop if the specified number
    // of milliseconds have passed and to relax the polling yielding the
    // cpu every millisecond.
    monitor() : msg(""), m_status(true), m_ticks(0), m_counter(0) {
      m_epoc_us = time_stamp();
    }

    void init(const char *message, uint32_t ticks_ms = 1000) {
      msg = message;
      m_ticks = m_counter = ticks_ms;
      m_init_us = time_stamp();
      printf("DE10-Nano-Mgr: %-32s : ", msg);
    }

    bool status() { return m_status; }

    void reset() { m_counter = m_ticks; }

    void done(bool status = true) {
      uint32_t elapsed = time_stamp(m_init_us);
      const char *rs = "FAIL";
      if (!m_counter) {
        status = false;
        rs = "TOUT";
      } else if (status) {
        rs = "PASS";
      }
      printf("\rDE10-Nano-Mgr: %-32s : %s in %u us\n", msg, rs, elapsed);
      if (!status) {
        m_status = false;
        throw 1;
      }
    }

    ~monitor() {
      uint32_t elapsed = time_stamp(m_epoc_us);
      const char *rs = m_status ? "SUCCESS" : "FAILURE";
      printf("DE10-Nano-Mgr: EXIT %s in %u us\n", rs, elapsed);
    }

    uint64_t time_stamp(uint64_t base_us = 0) {
      struct timeval tv;
      gettimeofday(&tv, NULL);
      return tv.tv_sec * 1000000L + tv.tv_usec - base_us;
    }

    bool operator() (bool cond) {
      if (m_counter) {
        if (!cond)
          return false;
        m_counter--;
        usleep(1000);
      }
      return m_counter;
    }
    const char *msg;

   private:
    bool m_status;
    uint32_t m_ticks, m_counter;
    uint64_t m_init_us, m_epoc_us;
  };

  enum BaseAddr {
    REGS_BASE_ADDR = 0xFF706000U,
    DATA_BASE_ADDR = 0xFFB90000U
  };

  De10NanoMgr() {
    m_page_size = sysconf(_SC_PAGE_SIZE);
    #ifdef MOCK_DEVMEM
    m_regs_base = reinterpret_cast<uint8_t*>(malloc(m_page_size));
    m_data_base = reinterpret_cast<uint8_t*>(malloc(m_page_size));
    #else
    m_regs_base = map_mem(REGS_BASE_ADDR);
    m_data_base = map_mem(DATA_BASE_ADDR);
    #endif  // MOCK_DEVMEM
    data.map(m_data_base);
    stat.map(m_regs_base);
    ctrl.map(m_regs_base);
    dclkcnt.map(m_regs_base);
    dclkstat.map(m_regs_base);
    gpio_inten.map(m_regs_base);
    gpio_porta_eoi.map(m_regs_base);
    gpio_ext_porta.map(m_regs_base);
  }

  ~De10NanoMgr() {
    #ifdef MOCK_DEVMEM
    free(m_regs_base);
    free(m_data_base);
    #else
    unmap_mem(m_regs_base);
    unmap_mem(m_data_base);
    #endif  // MOCK_DEVMEM
  }

  bool mapped() const { return m_regs_base && m_data_base; }

  void print(bool addr = false) {
    stat.print(addr, false);
    ctrl.print(addr, false);
    gpio_inten.print();
    gpio_porta_eoi.print();
    gpio_ext_porta.print(addr, false);
  }

 private:
  uint32_t msel_to_cfgwdth(uint32_t msel) {
    return(msel & 0b1000) >> 3;
  }

  uint32_t msel_to_cdratio(uint32_t msel) {
    uint32_t cfgwdth = msel_to_cfgwdth(msel);
    uint32_t cdratio = msel & 0b11;
    if (cfgwdth && cdratio)
      cdratio++;
    return cdratio;
  }

  uint8_t * map_mem(off_t addr, size_t pages = 1) {
    if (m_page_size <= 0) { return NULL; }

    int mem_fd = open("/dev/mem", O_SYNC | O_RDWR);
    if (mem_fd < 0) { return NULL; }

    void *vbase = mmap(NULL, pages*m_page_size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, mem_fd, addr & ~(pages*m_page_size-1));
    if (vbase == MAP_FAILED) { return NULL; }

    close(mem_fd);
    return reinterpret_cast<uint8_t*>(vbase);
  }

  void unmap_mem(void *base, size_t pages = 1) {
    if (base)
      munmap(base, pages * m_page_size);
  }

  uint8_t *m_regs_base, *m_data_base;
  size_t m_page_size;

 public:
  // Configuration sequence documented at page A-34.
  bool program_rbf(const char *rbf) {
    monitor mon;
    int rbf_fd;
    uint32_t count = 0;
    printf("DE10-Nano-Mgr: Programming FPGA from image %s\n", rbf);

    try {
      mon.init("Open RBF file");
      rbf_fd = open(rbf, (O_RDONLY | O_SYNC));
      mon.done(rbf_fd >= 0);

      // 1. Set the cdratio and cfgwdth bits of the ctrl register in the
      // FPGA manager registers (fpgamgrregs) to match the characteristics
      // of the configuration image. Tese settings are dependent on the
      // MSEL pins input.
      // 2. Set the nce bit of the ctrl register to 0 to enable HPS
      // configuration.
      // 3. Set the en bit of the ctrl register to 1 to give the FPGA
      // manager control of the configuration input signals.
      // 4. Set the nconfigpull bit of the ctrl register to 1 to pull
      // down the nCONFIG pin and put the FPGA portion of the device
      // into the reset phase.
      mon.init("Enable FPGA configuration");
      stat.read();
      if (stat.msel_is_invalid()) {
        printf("DE10-Nano-Mgr: msel %x is not a valid HPS configuration\n", stat.msel);
      } else {
        ctrl.read();
        ctrl.cdratio = msel_to_cdratio(stat.msel);
        ctrl.cfgwdth = msel_to_cfgwdth(stat.msel);
        ctrl.nce = 0;
        ctrl.en = 1;
        ctrl.nconfigpull = 1;
        ctrl.write();
      }
      mon.done(!stat.msel_is_invalid());

      // 5. Poll the mode bit of the stat register and wait until
      // the FPGA enters the reset phase.
      mon.init("Wait for FPGA to reset");
      do {
        stat.read();
      } while (mon(stat.mode != stat::FPGA_RESET_PHASE));
      mon.done();
      stat.print();

      // 6. Set the nconfigpull bit of the ctrl register to 0 to
      // release the FPGA from reset.
      mon.init("Release FPGA from reset");
      ctrl.nconfigpull = 0;
      ctrl.write();
      mon.done();

      // 7. Read the mode bit of the stat register and wait until
      // the FPGA enters the configuration phase.
      mon.init("Wait for configuration phase");
      do {
        stat.read();
      } while (mon(stat.mode != stat::FPGA_CONFIG_PHASE));
      mon.done();
      stat.print();

      // 8. Clear the interrupt bit of nSTATUS (ns) in the gpio interrupt
      // register (fpgamgrregs.mon.gpio_porta_eoi).
      mon.init("Clear nSTATUS interrupt bit");
      gpio_porta_eoi.clear();
      gpio_porta_eoi.ns = 1;
      gpio_porta_eoi.write();
      mon.done();

      // 9. Set the axicfgen bit of the ctrl register to 1 to enable
      // sending configuration data to the FPGA.
      mon.init("Enable configuration on AXI");
      ctrl.axicfgen = 1;
      ctrl.write();
      mon.done();

      // 10. Write the configuration image to the configuration data register
      // (data) in the FPGA manager module configuration data registers
      // (fpgamgrdata). You can also choose to use a DMA controller to
      // transfer the configuration image from a peripheral device to the
      // FPGA manager.
      ssize_t bytes;
      mon.init("Write configuration Image");
      do {
        data.value = 0;
        bytes = read(rbf_fd, &data.value, sizeof(data.value));
        if (bytes > 0) {
          if (!(count % (1<<16))) {
            printf("\rDE10-Nano-Mgr: %-32s : %u B", mon.msg, count);
            fflush(stdout);
          }
          data.write();
          count += bytes;
        }
      } while (bytes == 4);
      mon.done(count > 0);
      printf("DE10-Nano-Mgr: %-32s : written %u B\n", mon.msg, count);
      close(rbf_fd);

      // 11. Use the fpgamgrregs.mon.gpio_ext_porta registers to monitor
      // the CONF_DONE (cd) and nSTATUS (ns) bits.
      mon.init("Wait for CONF_DONE");
      do {
        gpio_ext_porta.read();
      } while (mon(gpio_ext_porta.cd != 1 && gpio_ext_porta.ns != 1));
      mon.done();
      stat.print();

      // 12. Set the axicfgen bit of the ctrl register to 0 to disable
      // configuration data on AXI slave.
      mon.init("Disable configuration on AXI");
      ctrl.axicfgen = 0;
      ctrl.write();
      mon.done();

      // 13. Clear any previous DONE status by writing a 1 to the dcntdone
      // bit of the DCLK status register (dclkstat) to clear the completed
      // status ﬂag.
      mon.init("Clear DCLK DONE status");
      dclkstat.dcntdone = 1;
      dclkstat.write();
      mon.done();

      // 14. Send the DCLKs required by the FPGA to enter the
      // initialization phase.
      mon.init("Send DCLK for init phase");
      dclkcnt.cnt = 4;
      dclkcnt.write();
      mon.done();

      // 15. Poll the dcntdone bit of the DCLK status register (dclkstat)
      // until it changes to 1, which indicates that all the DCLKs have
      // been sent.
      mon.init("Wait for DCLK");
      do {
        dclkstat.read();
      } while (mon(dclkstat.dcntdone != 1));
      mon.done();

      // 16. Write a 1 to the dcntdone bit of the DCLK status register to
      // clear the completed status ﬂag.
      mon.init("Clear DCLK status flag");
      dclkstat.dcntdone = 1;
      dclkstat.write();
      mon.done();

      // 17. Read the mode bit of the stat register to wait for the FPGA
      // to enter user mode.
      mon.init("Wait for FPGA user mode");
      do {
        stat.read();
      } while (mon(stat.mode != stat::FPGA_USER_MODE));
      mon.done();

      // 18. Set the en bit of the ctrl register to 0 to allow the
      // external pins to drive the configuration input signals.
      mon.init("Release control");
      ctrl.en = 0;
      ctrl.write();
      mon.done();
    }
    catch(int i) {
      close(rbf_fd);
      printf("DE10-Nano-Mgr: %-32s : written %u B\n", mon.msg, count);
      print();
    }

    return mon.status();
  }
};

#endif  // VTA_DE10NANO_DE10NANO_MGR_H_
