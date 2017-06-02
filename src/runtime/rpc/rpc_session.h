/*!
 *  Copyright (c) 2017 by Contributors
 * \file rpc_session.h
 * \brief Base RPC session interface.
 */
#ifndef TVM_RUNTIME_RPC_RPC_SESSION_H_
#define TVM_RUNTIME_RPC_RPC_SESSION_H_

#include <tvm/runtime/packed_func.h>
#include <mutex>
#include <string>
#include "../device_api.h"
#include "../../common/socket.h"

namespace tvm {
namespace runtime {

/*! \brief The remote functio handle */
using RPCFuncHandle = void*;

struct RPCArgBuffer;

/*! \brief The RPC code */
enum class RPCCode : int {
  kCallFunc,
  kReturn,
  kException,
  kShutdown,
  kCopyFromRemote,
  kCopyToRemote,
  kCopyAck,
  // The following are code that can send over CallRemote
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
  kModuleFree,
  kModuleGetFunc,
  kModuleGetSource
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
   * \brief Call into remote function
   * \param handle The function handle
   * \param args The arguments
   * \param rv The return value.
   */
  void CallFunc(RPCFuncHandle handle,
                TVMArgs args,
                TVMRetValue* rv);
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
   * \param nstep Number of steps to run.
   * \return A remote timer function
   */
  RPCFuncHandle GetTimeEvaluator(RPCFuncHandle fhandle,
                                 TVMContext ctx,
                                 int nstep);
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
   * \brief Create a RPC session with given socket
   * \param sock The socket.
   * \return The session.
   */
  static std::shared_ptr<RPCSession> Create(common::TCPSocket sock);
  /*!
   * \brief Try get session from the global session table by table index.
   * \param table_index The table index of the session.
   * \return The shared_ptr to the session, can be nullptr.
   */
  static std::shared_ptr<RPCSession> Get(int table_index);

 private:
  /*!
   * \brief Handle the remote call with f
   * \param f The handle function
   * \tparam F the handler function.
   */
  template<typename F>
  void CallHandler(F f);
  void Init();
  void Shutdown();
  void SendReturnValue(int succ, TVMValue value, int tcode);
  void SendPackedSeq(const TVMValue* arg_values, const int* type_codes, int n);
  void RecvPackedSeq(RPCArgBuffer *buf);
  RPCCode HandleNextEvent(TVMRetValue *rv);
  TVMContext StripSessMask(TVMContext ctx);
  // special handler.
  void HandleCallFunc();
  void HandleException();
  void HandleCopyFromRemote();
  void HandleCopyToRemote();
  void HandleReturn(TVMRetValue* rv);
  // Internal mutex
  std::recursive_mutex mutex_;
  // Internal socket
  common::TCPSocket sock_;
  // Internal temporal data space.
  std::string temp_data_;
  // call remote with the specified function coede.
  PackedFunc call_remote_;
  // The index of this session in RPC session table.
  int table_index_{0};
};

/*!
 * \brief Wrap a timer function for a given packed function.
 * \param f The function argument.
 * \param ctx The context.
 * \param nstep Number of repeative steps.
 */
PackedFunc WrapTimeEvaluator(PackedFunc f, TVMContext ctx, int nstep);

// Remote space pointer.
struct RemoteSpace {
  void* data;
  std::shared_ptr<RPCSession> sess;
};

// implementation of inline functions
template<typename... Args>
inline TVMRetValue RPCSession::CallRemote(RPCCode code, Args&& ...args) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  CHECK_EQ(sock_.SendAll(&code, sizeof(code)), sizeof(code));
  return call_remote_(std::forward<Args>(args)...);
}
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_SESSION_H_
