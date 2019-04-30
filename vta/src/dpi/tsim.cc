#include <vta/dpi/tsim.h>

#if VM_TRACE
#include <verilated_vcd_c.h>
#endif

#if VM_TRACE
#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)
#endif

static VTAContextHandle _ctx = nullptr;
static VTAMemDPIFunc _mem_dpi = nullptr;
static VTAHostDPIFunc _host_dpi = nullptr;

void VTAHostDPI(unsigned char *exit,
                unsigned char *req_valid,
                unsigned char *req_opcode,
                unsigned char *req_addr,
                unsigned int *req_value,
                unsigned char req_deq,
                unsigned char resp_valid,
                unsigned int resp_value) {
  assert(_host_dpi != nullptr);
  (*_host_dpi)(_ctx, exit, req_valid, req_opcode,
               req_addr, req_value, req_deq,
               resp_valid, resp_value);
}

void VTAMemDPI(unsigned char req_valid,
               unsigned char req_opcode,
               unsigned char req_len,
               unsigned long long req_addr,
               unsigned char wr_valid,
               unsigned long long wr_value,
               unsigned char *rd_valid,
               unsigned long long *rd_value,
               unsigned char rd_ready) {
  assert(_mem_dpi != nullptr);
  (*_mem_dpi)(_ctx, req_valid, req_opcode, req_len,
              req_addr, wr_valid, wr_value,
              rd_valid, rd_value, rd_ready);

}

void VTADPIInit(VTAContextHandle handle,
                VTAHostDPIFunc host_dpi,
                VTAMemDPIFunc mem_dpi) {
  _ctx = handle;
  _host_dpi = host_dpi;
  _mem_dpi = mem_dpi;
}

int VTADPISim(uint64_t max_cycles) {
  uint64_t trace_count = 0;

#if VM_TRACE
  uint64_t start = 0;
#endif

  VL_TSIM_NAME *top = new VL_TSIM_NAME;

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
  while (!Verilated::gotFinish() && trace_count < max_cycles) {
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

#if VM_TRACE
  tfp->close();
#endif

  delete top;

  return 0;
}
