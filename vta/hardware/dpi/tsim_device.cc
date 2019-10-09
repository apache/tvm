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
 */

#include <chrono>
#include <thread>
#include <vta/dpi/tsim.h>

#if VM_TRACE
#include <verilated_vcd_c.h>
#endif

#if VM_TRACE
#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)
#endif

static VTAContextHandle _ctx = nullptr;
static VTASimDPIFunc _sim_dpi = nullptr;
static VTAHostDPIFunc _host_dpi = nullptr;
static VTAMemDPIFunc _mem_dpi = nullptr;

void VTASimDPI(dpi8_t* wait,
               dpi8_t* exit) {
  assert(_sim_dpi != nullptr);
  (*_sim_dpi)(_ctx, wait, exit);
}

void VTAHostDPI(dpi8_t* req_valid,
                dpi8_t* req_opcode,
                dpi8_t* req_addr,
                dpi32_t* req_value,
                dpi8_t req_deq,
                dpi8_t resp_valid,
                dpi32_t resp_value) {
  assert(_host_dpi != nullptr);
  (*_host_dpi)(_ctx, req_valid, req_opcode,
               req_addr, req_value, req_deq,
               resp_valid, resp_value);
}

void VTAMemDPI(dpi8_t req_valid,
               dpi8_t req_opcode,
               dpi8_t req_len,
               dpi64_t req_addr,
               dpi8_t wr_valid,
               dpi64_t wr_value,
               dpi8_t* rd_valid,
               dpi64_t* rd_value,
               dpi8_t rd_ready) {
  assert(_mem_dpi != nullptr);
  (*_mem_dpi)(_ctx, req_valid, req_opcode, req_len,
              req_addr, wr_valid, wr_value,
              rd_valid, rd_value, rd_ready);

}

void VTADPIInit(VTAContextHandle handle,
                VTASimDPIFunc sim_dpi,
                VTAHostDPIFunc host_dpi,
                VTAMemDPIFunc mem_dpi) {
  _ctx = handle;
  _sim_dpi = sim_dpi;
  _host_dpi = host_dpi;
  _mem_dpi = mem_dpi;
}


// Override Verilator finish definition
// VL_USER_FINISH needs to be defined when compiling Verilator code
void vl_finish(const char* filename, int linenum, const char* hier) {
  Verilated::gotFinish(true);
}

int VTADPISim() {
  uint64_t trace_count = 0;
  Verilated::flushCall();
  Verilated::gotFinish(false);

#if VM_TRACE
  uint64_t start = 0;
#endif

  VL_TSIM_NAME* top = new VL_TSIM_NAME;

#if VM_TRACE
  Verilated::traceEverOn(true);
  VerilatedVcdC* tfp = new VerilatedVcdC;
  top->trace(tfp, 99);
  tfp->open(STRINGIZE_VALUE_OF(TSIM_TRACE_FILE));
#endif

  // reset
  for (int i = 0; i < 10; i++) {
    top->reset = 1;
    top->clock = 0;
    top->eval();
#if VM_TRACE
    if (trace_count >= start)
      tfp->dump(static_cast<vluint64_t>(trace_count * 2));
#endif
    top->clock = 1;
    top->eval();
#if VM_TRACE
    if (trace_count >= start)
      tfp->dump(static_cast<vluint64_t>(trace_count * 2 + 1));
#endif
    trace_count++;
  }
  top->reset = 0;

  // start simulation
  while (!Verilated::gotFinish()) {
    top->sim_clock = 0;
    top->clock = 0;
    top->eval();
#if VM_TRACE
    if (trace_count >= start)
      tfp->dump(static_cast<vluint64_t>(trace_count * 2));
#endif
    top->sim_clock = 1;
    top->clock = 1;
    top->eval();
#if VM_TRACE
    if (trace_count >= start)
      tfp->dump(static_cast<vluint64_t>(trace_count * 2 + 1));
#endif
    trace_count++;
    while (top->sim_wait) {
      top->clock = 0;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      top->sim_clock = 0;
      top->eval();
      top->sim_clock = 1;
      top->eval();
    }
  }

#if VM_TRACE
  tfp->close();
#endif

  delete top;

  return 0;
}
