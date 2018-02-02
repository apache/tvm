/*!
 *  Copyright (c) 2017 by Contributors
 * \file rpc_session.h
 * \brief Base RPC session interface.
 */
#ifndef TVM_RUNTIME_RPC_RPC_SESSION_H_
#define TVM_RUNTIME_RPC_RPC_SESSION_H_

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/device_api.h>
#include <mutex>
#include <string>
#include "../../common/ring_buffer.h"

namespace tvm {
namespace runtime {

const int kRPCMagic = 0xff271;

/*! \brief The remote functio handle */
using RPCFuncHandle = void*;

struct RPCArgBuffer;

/*! \brief The RPC code */
enum class RPCCode : int {
  kNone,
  kCallFunc,
  kReturn,
  kException,
  kShutdown,
  kCopyFromRemote,
  kCopyToRemote,
  kCopyAck,
  // The following are code that can send over CallRemote
  kSystemFuncStart,
  kGetGlobalFunc,
  kGetTimeEvaluator,
  kFreeFunc,
  kDevSetDevice,
  kDevGetAttr,
  kDevAllocData,
  kDevFreeData,
  kDevStreamSync,
  kCopyAmongRemote,
  kModuleLoad,
  kModuleImport,
  kModuleFree,
  kModuleGetFunc,
  kModuleGetSource,
};

/*!
 * \brief Abstract channel interface used to create RPCSession.
 */
class RPCChannel {
 public:
  /*! \brief virtual destructor */
  virtual ~RPCChannel() {}
  /*!
   * \brief Send data over to the channel.
   * \param data The data pointer.
   * \param size The size fo the data.
   * \return The actual bytes sent.
   */
  virtual size_t Send(const void* data, size_t size) = 0;
  /*!
e   * \brief Recv data from channel.
   *
   * \param data The data pointer.
   * \param size The size fo the data.
   * \return The actual bytes received.
   */
  virtual size_t Recv(void* data, size_t size) = 0;
};

// Bidirectional Communication Session of PackedRPC
class RPCSession {
 public:
  /*! \brief virtual destructor */
  ~RPCSession();
  /*!
   *  \brief The server loop that server runs to handle RPC calls.
   */
  void ServerLoop();
  /*!
   * \brief Message handling function for event driven server.
   *  Called when the server receives a message.
   *  Event driven handler will never call recv on the channel
   *  and always relies on the ServerEventHandler.
   *  to receive the data.
   *
   * \param in_bytes The incoming bytes.
   * \param event_flag  1: read_available, 2: write_avaiable.
   * \return State flag.
   *     1: continue running, no need to write,
   *     2: need to write
   *     0: shutdown
   */
  int ServerEventHandler(const std::string& in_bytes,
                         int event_flag);
  /*!
   * \brief Call into remote function
   * \param handle The function handle
   * \param args The arguments
   * \param rv The return value.
   * \param fwrap Wrapper function to turn Function/Module handle into real return.
   */
  void CallFunc(RPCFuncHandle handle,
                TVMArgs args,
                TVMRetValue* rv,
                const PackedFunc* fwrap);
  /*!
   * \brief Copy bytes into remote array content.
   * \param from The source host data.
   * \param from_offset The byte offeset in the from.
   * \param to The target array.
   * \param to_offset The byte offset in the to.
   * \param size The size of the memory.
   * \param ctx_to The target context.
   */
  void CopyToRemote(void* from,
                    size_t from_offset,
                    void* to,
                    size_t to_offset,
                    size_t size,
                    TVMContext ctx_to);
  /*!
   * \brief Copy bytes from remote array content.
   * \param from The source host data.
   * \param from_offset The byte offeset in the from.
   * \param to The target array.
   * \param to_offset The byte offset in the to.
   * \param size The size of the memory.
   * \param ctx_from The source context.
   */
  void CopyFromRemote(void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from);
  /*!
   * \brief Get a remote timer function on ctx.
   *  This function consumes fhandle, caller should not call Free on fhandle.
   *
   * \param fhandle The function handle.
   * \param ctx The ctx to run measurement on.
   * \param number How many steps to run in each time evaluation
   * \param repeat How many times to repeat the timer
   * \return A remote timer function
   */
  RPCFuncHandle GetTimeEvaluator(RPCFuncHandle fhandle,
                                 TVMContext ctx,
                                 int number,
                                 int repeat);
  /*!
   * \brief Call a remote defined system function with arguments.
   * \param fcode The function code.
   * \param args The arguments
   * \return The returned remote value.
   */
  template<typename... Args>
  inline TVMRetValue CallRemote(RPCCode fcode, Args&& ...args);
  /*!
   * \return The session table index of the session.
   */
  int table_index() const {
    return table_index_;
  }
  /*!
   * \brief Create a RPC session with given channel.
   * \param channel The communication channel.
   * \param name The name of the session, used for debug
   * \return The session.
   */
  static std::shared_ptr<RPCSession> Create(
      std::unique_ptr<RPCChannel> channel,
      std::string name);
  /*!
   * \brief Try get session from the global session table by table index.
   * \param table_index The table index of the session.
   * \return The shared_ptr to the session, can be nullptr.
   */
  static std::shared_ptr<RPCSession> Get(int table_index);

 private:
  class EventHandler;
  // Handle events until receives a return
  // Also flushes channels so that the function advances.
  RPCCode HandleUntilReturnEvent(
      TVMRetValue* rv, bool client_mode, const PackedFunc* fwrap);
  // Initalization
  void Init();
  // Shutdown
  void Shutdown();
  // Internal channel.
  std::unique_ptr<RPCChannel> channel_;
  // Internal mutex
  std::recursive_mutex mutex_;
  // Internal ring buffer.
  common::RingBuffer reader_, writer_;
  // Event handler.
  std::shared_ptr<EventHandler> handler_;
  // call remote with specified function code.
  PackedFunc call_remote_;
  // The index of this session in RPC session table.
  int table_index_{0};
  // The name of the session.
  std::string name_;
};

/*!
 * \brief Wrap a timer function for a given packed function.
 * \param f The function argument.
 * \param ctx The context.
 * \param number Number of steps in the inner iteration
 * \param repeat How many steps to repeat the time evaluation.
 */
PackedFunc WrapTimeEvaluator(PackedFunc f, TVMContext ctx, int number, int repeat);

/*!
 * \brief Create a Global RPC module that refers to the session.
 * \param sess The RPC session of the global module.
 * \return The created module.
 */
Module CreateRPCModule(std::shared_ptr<RPCSession> sess);

// Remote space pointer.
struct RemoteSpace {
  void* data;
  std::shared_ptr<RPCSession> sess;
};

// implementation of inline functions
template<typename... Args>
inline TVMRetValue RPCSession::CallRemote(RPCCode code, Args&& ...args) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  writer_.Write(&code, sizeof(code));
  return call_remote_(std::forward<Args>(args)...);
}
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_SESSION_H_
