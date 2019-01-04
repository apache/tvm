#ifndef __HLS_H__
#define __HLS_H__

#ifndef __INTELFPGA_COMPILER__
#ifndef component
#define component
#endif
#define HLS_X86
#endif

#include "HLS/hls_internal.h"

/* Deprecated APIs and names after intel rebranding */
#ifdef __IHC_USE_DEPRECATED_NAMES
#pragma message "Warning: Enabling deprecated names - these names will not be supported in future releases."
namespace ihc {}
namespace altera = ihc;
#define altera_hls_component_run_all ihc_hls_component_run_all
#define altera_fence ihc_fence
#define altera_hls_get_sim_time ihc_hls_get_sim_time
#define altera_hls_enqueue ihc_hls_enqueue
#define altera_hls_enqueue_noret ihc_hls_enqueue_noret
#endif

#ifdef __INTELFPGA_COMPILER__
// Memory attributes
#define hls_register                                  __attribute__((__register__))
#define hls_memory                                    __attribute__((__memory__))
#define hls_numbanks(__x)                             __attribute__((__numbanks__(__x)))
#define hls_bankwidth(__x)                            __attribute__((__bankwidth__(__x)))
#define hls_singlepump                                __attribute__((__singlepump__))
#define hls_doublepump                                __attribute__((__doublepump__))
#define hls_numports_readonly_writeonly(__rd, __wr)   __attribute__((__numports_readonly_writeonly__(__rd, __wr)))
#define hls_bankbits(__x, ...)                        __attribute__((__bank_bits__(__x, ##__VA_ARGS__)))
#define hls_merge(__x, __y)                           __attribute__((merge(__x, __y)))
#define hls_init_on_reset                             __attribute__((__static_array_reset__(1)))
#define hls_init_on_powerup                           __attribute__((__static_array_reset__(0)))
#define hls_numreadports(__x)                         __attribute__((__numreadports__(__x)))
#define hls_numwriteports(__x)                        __attribute__((__numwriteports__(__x)))

// Memory attribute macros
#define hls_simple_dual_port_memory hls_memory hls_singlepump hls_numports_readonly_writeonly(1,1)

// Interface synthesis attributes
#define hls_avalon_streaming_component         __attribute__((component_interface("avalon_streaming")))
#define hls_avalon_slave_component             __attribute__((component_interface("avalon_mm_slave"))) __attribute__((stall_free_return))
#define hls_always_run_component               __attribute__((component_interface("always_run"))) __attribute__((stall_free_return))
#define hls_conduit_argument                   __attribute__((argument_interface("wire")))
#define hls_avalon_slave_register_argument     __attribute__((argument_interface("avalon_mm_slave")))
#define hls_avalon_slave_memory_argument(__x)  __attribute__((local_mem_size(__x))) __attribute__((slave_memory_argument))
#define hls_stable_argument                    __attribute__((stable_argument))
#define hls_stall_free_return                  __attribute__((stall_free_return))

// Component attributes
#define hls_max_concurrency(__x)               __attribute__((max_concurrency(__x)))

#else
#define hls_register
#define hls_memory
#define hls_numbanks(__x)
#define hls_bankwidth(__x)
#define hls_singlepump
#define hls_doublepump
#define hls_numports_readonly_writeonly(__rd, __wr)
#define hls_bankbits(__x, ...)
#define hls_merge(__x, __y)
#define hls_init_on_reset
#define hls_init_on_powerup

#define hls_numreadports(__x)
#define hls_numwriteports(__x)

#define hls_simple_dual_port_memory

#define hls_avalon_streaming_component
#define hls_avalon_slave_component
#define hls_always_run_component
#define hls_conduit_argument
#define hls_avalon_slave_register_argument
#define hls_avalon_slave_memory_argument(__x)
#define hls_stable_argument
#define hls_stall_free_return

#define hls_max_concurrency(__x)

#endif

////////////////////////////////////////////////////////////////////////////////
// Interfaces Declarations
////////////////////////////////////////////////////////////////////////////////

namespace ihc {

  ////////////////////////////////
 /// memory master interface  ///
////////////////////////////////

template <int n>         class dwidth:public internal::param {};
template <int n>         class awidth:public internal::param {};
template <int n>         class latency: public internal::param {};
template <readwrite_t n> class readwrite_mode: public internal::param{}; // declared in hls_internal.h as enum readwrite_t {readwrite = 0, readonly = 1, writeonly = 2};
template <int n>         class maxburst: public internal::param {};
template <int n>         class align: public internal::param {};
template <int n>         class aspace: public internal::param {};
template <bool n>        class waitrequest: public internal::param{};

template <typename DT, typename p1 = internal::notinit, typename p2 = internal::notinit, typename p3 = internal::notinit, typename p4 = internal::notinit, typename p5 = internal::notinit, typename p6 = internal::notinit, typename p7 = internal::notinit, typename p8 = internal::notinit>
class mm_master
#ifdef HLS_X86
  : public internal::memory_base
#endif
{
public:

#ifdef HLS_X86
  template<typename T> explicit mm_master(T *data, int size=0, bool use_socket=false):internal::memory_base(data,size,sizeof(DT),use_socket) {
  }
#else
  template<typename T> explicit mm_master(T *data, int size=0, bool use_socket=false);
#endif

  //////////////////////////////////////////////////////////////////////////////
  // The following operators apply to the mm_master object and are only
  // supported in the testbench:
  //   mm_master()
  //   getInterfaceAtIndex()
  //////////////////////////////////////////////////////////////////////////////
  // The following operators apply to the base pointer and should only be used
  // in the component:
  //   operator[]()
  //   operator*()
  //   operator->()
  //   operator T()
  //   operator+()
  //   operator&()
  //   operator|()
  //   operator^()
  //////////////////////////////////////////////////////////////////////////////

  DT &operator[](int index);
  DT &operator*();
  DT *operator->();
  template<typename T> operator T();
  DT *operator+(int index);
  template<typename T> DT *operator&(T value);
  template<typename T> DT *operator|(T value);
  template<typename T> DT *operator^(T value);
  // This function is only supported in the testbench:
  mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>& getInterfaceAtIndex(int index);

#ifdef HLS_X86
  // The copy constructor and assignment operator are necessary to ensure
  // new_masters doesn't get copied.
  mm_master(const mm_master &other) {
    mem = other.mem;
    size = other.size;
    data_size = other.data_size;
    use_socket = other.use_socket;
  }
  mm_master& operator=(const mm_master& other) {
    mem = other.mem;
    size = other.size;
    data_size = other.data_size;
    use_socket = other.use_socket;
  }
  // Clean up any derrived mm_masters when this object is destroyed.
  ~mm_master() {
    for(std::vector<internal::memory_base* >::iterator it = new_masters.begin(),
        ie = new_masters.end(); it != ie; it++) {
      delete *it;
    }
    new_masters.clear();
  }
private:
  std::vector<internal::memory_base* > new_masters;
#endif

};

  /////////////////////////////
 /// streaming interfaces  ///
//////////////////////////////

template <int n> class buffer:public internal::param {};
template <int n> class readyLatency:public internal::param {};
template <int n> class bitsPerSymbol:public internal::param {};
template <bool b> class usesPackets:public internal::param {};
template <bool b> class usesValid:public internal::param {};
template <bool b> class usesReady:public internal::param {};

template <typename T, typename p1 = internal::notinit , typename p2 = internal::notinit, typename p3 = internal::notinit, typename p4 = internal::notinit, typename p5 = internal::notinit>
class stream_in : public internal::stream<T,p1,p2,p3,p4,p5> {
public:
  stream_in();
  T read();
  void write(T arg);
  T tryRead(bool &success);
  bool tryWrite(T arg);

  // for packet based stream
  T read(bool& sop, bool& eop);
  void write(T arg, bool sop, bool eop);
  T tryRead(bool &success, bool& sop, bool& eop);
  bool empty();
  bool tryWrite(T arg, bool sop, bool eop);
  void setStallCycles(unsigned average_stall, unsigned stall_delta=0);
  void setValidCycles(unsigned average_valid, unsigned valid_delta=0);
};

template <typename T, typename p1 = internal::notinit , typename p2 = internal::notinit, typename p3 = internal::notinit, typename p4 = internal::notinit, typename p5 = internal::notinit>
class stream_out : public internal::stream<T,p1,p2,p3,p4,p5> {

public:
  stream_out();
  T read();
  void write(T);
  T tryRead(bool &success);
  bool tryWrite(T arg);

  // for packet based stream
  T read(bool& sop, bool& eop);
  void write(T arg, bool sop, bool eop);
  T tryRead(bool &success, bool& sop, bool& eop);
  bool empty();
  bool tryWrite(T arg, bool sop, bool eop);
  void setStallCycles(unsigned average_stall, unsigned stall_delta=0);
  void setReadyCycles(unsigned average_ready, unsigned ready_delta=0);
};

}//namespace ihc

////////////////////////////////////////////////////////////////////////////////
// HLS Cosimulation Support API
////////////////////////////////////////////////////////////////////////////////

#define ihc_hls_enqueue(retptr, func, ...) \
  { \
    if (__ihc_hls_async_call_capable()){ \
      __ihc_enqueue_handle=(retptr); \
      (void) (*(func))(__VA_ARGS__); \
      __ihc_enqueue_handle=0; \
    } else { \
      *(retptr) = (*(func))(__VA_ARGS__); \
    } \
  }

#define ihc_hls_enqueue_noret(func, ...) \
  { \
  __ihc_enqueue_handle=& __ihc_enqueue_handle; \
  (*(func))(__VA_ARGS__); \
  __ihc_enqueue_handle=0; \
  }

#define ihc_hls_component_run_all(component_address) \
  __ihc_hls_component_run_all((void*) (component_address))

// When running a simulation, this function will issue a reset to all components
// in the testbench
// Returns: 0 if reset did not occur (ie. if the component target is x86)
//          1 if reset occured (ie. if the component target is an FPGA)
extern "C" int ihc_hls_sim_reset(void);

////////////////////////////////////////////////////////////////////////////////
// HLS Component Built-Ins
////////////////////////////////////////////////////////////////////////////////

//Builtin memory fence function call
#ifdef HLS_X86
inline void ihc_fence() {};

#else
extern "C" void mem_fence(int);
inline void ihc_fence() {
  // fence on all types of fences from OpenCL
  mem_fence(-1);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Implementions, no declarations below
////////////////////////////////////////////////////////////////////////////////

namespace ihc {
#ifdef HLS_X86

  //////////////////
 /// mm_master  ///
//////////////////

template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
DT &mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::operator[](int index) {
  assert(size==0 || index*data_size<size);
  return ((DT*)mem)[index];
}

template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
DT &mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::operator*() {
  return ((DT*)mem)[0];
}

template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
DT *mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::operator->() {
  return (DT*)mem;
}

template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
template<typename T> mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::operator T() {
  return (T)((unsigned long long)mem);
}

template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
DT *mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::operator+(int index) {
  assert(size==0 || index*data_size<size);
  return &((DT*)mem)[index];
}

// Bitwise operators
template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
template<typename T> DT *mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::operator&(T value) {
  return (DT*)((unsigned long long)mem & (unsigned long long)value);
}

template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
template<typename T> DT *mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::operator|(T value) {
  return (DT*)((unsigned long long)mem | (unsigned long long)value);
}

template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
template<typename T> DT *mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::operator^(T value) {
  return (DT*)((unsigned long long)mem ^ (unsigned long long)value);
}

// Function for creating new mm_master at an offset
template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>& mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::getInterfaceAtIndex(int index) {
  assert(size==0 || index*data_size<size);
  // This new object is cleaned up when this' destructor is called.
  mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8> *temp = new mm_master(&(((DT*)mem)[index]), size - index * sizeof(DT), use_socket);
  new_masters.push_back(temp);
  return *temp;
}

  ///////////////////
 /// stream_in   ///
///////////////////

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
stream_in<T,p1,p2,p3,p4,p5>::stream_in() {}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
T stream_in<T,p1,p2,p3,p4,p5>::tryRead(bool &success) {
  success = !internal::stream<T,p1,p2,p3,p4,p5>::_internal_cosim_empty();
  if (success) {
    return read();
  } else {
    return T();
  }
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
bool stream_in<T,p1,p2,p3,p4,p5>::empty() {
  bool isempty = internal::stream<T,p1,p2,p3,p4,p5>::_internal_cosim_empty();
  return isempty;
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
T stream_in<T,p1,p2,p3,p4,p5>::read() {
    T elem = internal::stream<T,p1,p2,p3,p4,p5>::read();
    return elem;
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
bool stream_in<T,p1,p2,p3,p4,p5>::tryWrite(T arg) {
  bool success = true; /* stl::queue has no full */
  if (success) {
    write(arg);
  }
  return success;
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
void stream_in<T,p1,p2,p3,p4,p5>::write(T arg) {
    internal::stream<T,p1,p2,p3,p4,p5>::write(arg);
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
T stream_in<T,p1,p2,p3,p4,p5>::tryRead(bool &success, bool& sop, bool& eop) {
  success = !internal::stream<T,p1,p2,p3,p4,p5>::_internal_cosim_empty();
  if (success) {
    return read(sop, eop);
  } else {
    return T();
  }
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
T stream_in<T,p1,p2,p3,p4,p5>::read(bool& sop, bool& eop) {
    T elem = internal::stream<T,p1,p2,p3,p4,p5>::read(sop, eop);
    return elem;
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
bool stream_in<T,p1,p2,p3,p4,p5>::tryWrite(T arg, bool sop, bool eop) {
  bool success = true; /* stl::queue has no full */
  if (success) {
    write(arg, sop, eop);
  }
  return success;
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
void stream_in<T,p1,p2,p3,p4,p5>::write(T arg, bool sop, bool eop) {
    internal::stream<T,p1,p2,p3,p4,p5>::write(arg, sop, eop);
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
void stream_in<T,p1,p2,p3,p4,p5>::setStallCycles(unsigned average_stall, unsigned stall_delta) {
  if (stall_delta > average_stall) {
    __ihc_hls_runtime_error_x86("The stall delta in setStallCycles cannot be larger than the average stall value");
  }
  internal::stream<T,p1,p2,p3,p4,p5>::setStallCycles(average_stall, stall_delta);
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
void stream_in<T,p1,p2,p3,p4,p5>::setValidCycles(unsigned average_valid, unsigned valid_delta) {
  if (average_valid == 0) {
    __ihc_hls_runtime_error_x86("The valid average in setValidCycles must be at least 1");
  }
  if (valid_delta > average_valid) {
    __ihc_hls_runtime_error_x86("The valid delta in setValidCycles cannot be larger than the average valid value");
  }
  internal::stream<T,p1,p2,p3,p4,p5>::setReadyorValidCycles(average_valid, valid_delta);
}

  ///////////////////
 /// stream_out  ///
///////////////////

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
stream_out<T,p1,p2,p3,p4,p5>::stream_out() {
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
T stream_out<T,p1,p2,p3,p4,p5>::tryRead(bool &success) {
  success = !internal::stream<T,p1,p2,p3,p4,p5>::_internal_cosim_empty();
  if (success) {
    return read();
  } else {
    return T();
  }
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
bool stream_out<T,p1,p2,p3,p4,p5>::empty() {
  bool isempty = internal::stream<T,p1,p2,p3,p4,p5>::_internal_cosim_empty();
  return isempty;
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
T stream_out<T,p1,p2,p3,p4,p5>::read() {
    T elem = internal::stream<T,p1,p2,p3,p4,p5>::read();
    return elem;
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
void stream_out<T,p1,p2,p3,p4,p5>::write(T arg) {
    internal::stream<T,p1,p2,p3,p4,p5>::write(arg);
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
bool stream_out<T,p1,p2,p3,p4,p5>::tryWrite(T arg) {
  bool success = true; /* stl::queue has no full */
  if (success) {
    write(arg);
  }
  return success;
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
T stream_out<T,p1,p2,p3,p4,p5>::tryRead(bool &success, bool& sop, bool& eop) {
  success = !internal::stream<T,p1,p2,p3,p4,p5>::_internal_cosim_empty();
  if (success) {
    return read(sop, eop);
  } else {
    return T();
  }
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
T stream_out<T,p1,p2,p3,p4,p5>::read(bool& sop, bool& eop) {
    T elem = internal::stream<T,p1,p2,p3,p4,p5>::read(sop, eop);
    return elem;
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
void stream_out<T,p1,p2,p3,p4,p5>::write(T arg, bool sop, bool eop) {
    internal::stream<T,p1,p2,p3,p4,p5>::write(arg, sop, eop);
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
bool stream_out<T,p1,p2,p3,p4,p5>::tryWrite(T arg, bool sop, bool eop) {
  bool success = true; /* stl::queue has no full */
  if (success) {
    write(arg, sop, eop);
  }
  return success;
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
void stream_out<T,p1,p2,p3,p4,p5>::setStallCycles(unsigned average_stall, unsigned stall_delta) {
  if (stall_delta > average_stall) {
    __ihc_hls_runtime_error_x86("The stall delta in setStallCycles cannot be larger than the average stall value");
  }
  internal::stream<T,p1,p2,p3,p4,p5>::setStallCycles(average_stall, stall_delta);
}

template<typename T, typename p1, typename p2, typename p3, typename p4, typename p5>
void stream_out<T,p1,p2,p3,p4,p5>::setReadyCycles(unsigned average_ready, unsigned ready_delta) {
  if (average_ready == 0) {
    __ihc_hls_runtime_error_x86("The ready average in setReadCycles must be at least 1");
  }
  if (ready_delta > average_ready) {
    __ihc_hls_runtime_error_x86("The ready delta in setReadyCycles cannot be larger than the average ready value");
  }
  internal::stream<T,p1,p2,p3,p4,p5>::setReadyorValidCycles(average_ready, ready_delta);
}
#else //fpga path. Ignore the class just return a consistant pointer/reference

  //////////////////
 /// mm_master  ///
//////////////////

template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
DT &mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::operator[](int index) {
  return ((DT*)this)[index];
}

template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
DT &mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::operator*(){
  return *((DT*)this);
}

template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
DT *mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::operator->() {
  return (DT*)this;
}

template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
DT *mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::operator+(int index) {
  return ((DT*)this)+index;
}

template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
template<typename T> mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::operator T() {
  return (T)((unsigned long long)this);
}

// Bitwise operators
template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
template<typename T> DT *mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::operator&(T value) {
  return (DT*)((unsigned long long)this & (unsigned long long)value);
}

template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
template<typename T> DT *mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::operator|(T value) {
  return (DT*)((unsigned long long)this | (unsigned long long)value);
}

template <typename DT, typename p1, typename p2, typename p3, typename p4, typename p5, typename p6, typename p7, typename p8>
template<typename T> DT *mm_master<DT, p1, p2, p3, p4, p5, p6, p7, p8>::operator^(T value) {
  return (DT*)((unsigned long long)this ^ (unsigned long long)value);
}

#endif
} // namespace ihc

#endif

