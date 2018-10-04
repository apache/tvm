/*!
 *  Copyright (c) 2017 by Contributors
 * \file socket.h
 * \brief this file aims to provide a wrapper of sockets
 * \author Tianqi Chen
 */
#ifndef TVM_COMMON_SOCKET_H_
#define TVM_COMMON_SOCKET_H_

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
using ssize_t = int;
#ifdef _MSC_VER
#pragma comment(lib, "Ws2_32.lib")
#endif
#else
#include <fcntl.h>
#include <netdb.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/ioctl.h>
#endif
#include <dmlc/logging.h>
#include <string>
#include <cstring>
#include <vector>
#include "../common/util.h"


namespace tvm {
namespace common {
/*!
 * \brief Get current host name.
 * \return The hostname.
 */
inline std::string GetHostName() {
  std::string buf; buf.resize(256);
  CHECK_NE(gethostname(&buf[0], 256), -1);
  return std::string(buf.c_str());
}

/*!
 * \brief ValidateIP validates an ip address.
 * \param ip The ip address in string format localhost or x.x.x.x format
 * \return result of operation.
 */
inline bool ValidateIP(std::string ip) {
    if (ip == "localhost") {
      return true;
    }
    std::vector<std::string> list = Split(ip, '.');
    if (list.size() != 4)
        return false;
    for (std::string str : list) {
      if (!IsNumber(str) || std::stoi(str) > 255 || std::stoi(str) < 0)
        return false;
    }
    return true;
}

/*!
 * \brief Common data structure fornetwork address.
 */
struct SockAddr {
  sockaddr_in addr;
  SockAddr() {}
  /*!
   * \brief construc address by url and port
   * \param url The url of the address
   * \param port The port of the address.
   */
  SockAddr(const char *url, int port) {
    this->Set(url, port);
  }

  /*!
   * \brief SockAddr Get the socket address from tracker.
   * \param tracker The url containing the ip and port number. Format is ('192.169.1.100', 9090)
   * \return SockAddr parsed from url.
   */
  explicit SockAddr(const std::string &url) {
    size_t sep = url.find(",");
    std::string host = url.substr(2, sep - 3);
    std::string port = url.substr(sep + 1, url.length() - 1);
    CHECK(ValidateIP(host)) << "Url address is not valid " << url;
    if (host == "localhost") {
      host = "127.0.0.1";
    }
    this->Set(host.c_str(), std::stoi(port));
  }

  /*!
   * \brief set the address
   * \param host the url of the address
   * \param port the port of address
   */
  void Set(const char *host, int port) {
    addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_protocol = SOCK_STREAM;
    addrinfo *res = NULL;
    int sig = getaddrinfo(host, NULL, &hints, &res);
    CHECK(sig == 0 && res != NULL)
        << "cannot obtain address of " <<  host;
    CHECK(res->ai_family == AF_INET)
        << "Does not support IPv6";
    memcpy(&addr, res->ai_addr, res->ai_addrlen);
    addr.sin_port = htons(port);
    freeaddrinfo(res);
  }
  /*! \brief return port of the address */
  int port() const {
    return ntohs(addr.sin_port);
  }
  /*! \return a string representation of the address */
  std::string AsString() const {
    std::string buf; buf.resize(256);
#ifdef _WIN32
    const char *s = inet_ntop(AF_INET, (PVOID)&addr.sin_addr,
                              &buf[0], buf.length());
#else
    const char *s = inet_ntop(AF_INET, &addr.sin_addr,
                              &buf[0], static_cast<socklen_t>(buf.length()));
#endif
    CHECK(s != nullptr) << "cannot decode address";
    std::ostringstream os;
    os << s << ":" << port();
    return os.str();
  }
};
/*!
 * \brief base class containing common operations of TCP and UDP sockets
 */
class Socket {
 public:
#if defined(_WIN32)
  using sock_size_t = int;
  using SockType = SOCKET;
#else
  using SockType = int;
  using sock_size_t = size_t;
  static constexpr int INVALID_SOCKET = -1;
#endif
  /*! \brief the file descriptor of socket */
  SockType sockfd;
  /*!
   * \brief set this socket to use non-blocking mode
   * \param non_block whether set it to be non-block, if it is false
   *        it will set it back to block mode
   */
  void SetNonBlock(bool non_block) {
#ifdef _WIN32
    u_long mode = non_block ? 1 : 0;
    if (ioctlsocket(sockfd, FIONBIO, &mode) != NO_ERROR) {
      Socket::Error("SetNonBlock");
    }
#else
    int flag = fcntl(sockfd, F_GETFL, 0);
    if (flag == -1) {
      Socket::Error("SetNonBlock-1");
    }
    if (non_block) {
      flag |= O_NONBLOCK;
    } else {
      flag &= ~O_NONBLOCK;
    }
    if (fcntl(sockfd, F_SETFL, flag) == -1) {
      Socket::Error("SetNonBlock-2");
    }
#endif
  }
  /*!
   * \brief bind the socket to an address
   * \param addr The address to be binded
   */
  void Bind(const SockAddr &addr) {
    if (bind(sockfd, reinterpret_cast<const sockaddr*>(&addr.addr),
             sizeof(addr.addr)) == -1) {
      Socket::Error("Bind");
    }
  }
  /*!
   * \brief try bind the socket to host, from start_port to end_port
   * \param host host_address to bind the socket
   * \param start_port starting port number to try
   * \param end_port ending port number to try
   * \return the port successfully bind to, return -1 if failed to bind any port
   */
  inline int TryBindHost(std::string host, int start_port, int end_port) {
    for (int port = start_port; port < end_port; ++port) {
      SockAddr addr(host.c_str(), port);
      if (bind(sockfd, reinterpret_cast<sockaddr*>(&addr.addr),
               sizeof(addr.addr)) == 0) {
        return port;
      } else {
        LOG(WARNING) << "Bind failed to  " << host << ":" << port;
      }
#if defined(_WIN32)
      if (WSAGetLastError() != WSAEADDRINUSE) {
        Socket::Error("TryBindHost");
      }
#else
      if (errno != EADDRINUSE) {
        Socket::Error("TryBindHost");
      }
#endif
    }
    return -1;
  }
  /*! \brief get last error code if any */
  int GetSockError() const {
    int error = 0;
    socklen_t len = sizeof(error);
    if (getsockopt(sockfd,  SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&error), &len) != 0) {
      Error("GetSockError");
    }
    return error;
  }
  /*! \brief check if anything bad happens */
  bool BadSocket() const {
    if (IsClosed()) return true;
    int err = GetSockError();
    if (err == EBADF || err == EINTR) return true;
    return false;
  }
  /*! \brief check if socket is already closed */
  bool IsClosed() const {
    return sockfd == INVALID_SOCKET;
  }
  /*! \brief close the socket */
  void Close() {
    if (sockfd != INVALID_SOCKET) {
#ifdef _WIN32
      closesocket(sockfd);
#else
      close(sockfd);
#endif
      sockfd = INVALID_SOCKET;
    } else {
      Error("Socket::Close double close the socket or close without create");
    }
  }
  /*!
   * \return last error of socket 2operation
   */
  static int GetLastError() {
#ifdef _WIN32
    return WSAGetLastError();
#else
    return errno;
#endif
  }
  /*! \return whether last error was would block */
  static bool LastErrorWouldBlock() {
    int errsv = GetLastError();
#ifdef _WIN32
    return errsv == WSAEWOULDBLOCK;
#else
    return errsv == EAGAIN || errsv == EWOULDBLOCK;
#endif
  }
  /*!
   * \brief start up the socket module
   *   call this before using the sockets
   */
  static void Startup() {
#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) == -1) {
      Socket::Error("Startup");
    }
    if (LOBYTE(wsa_data.wVersion) != 2 || HIBYTE(wsa_data.wVersion) != 2) {
      WSACleanup();
      LOG(FATAL) << "Could not find a usable version of Winsock.dll";
    }
#endif
  }
  /*!
   * \brief shutdown the socket module after use, all sockets need to be closed
   */
  static void Finalize() {
#ifdef _WIN32
    WSACleanup();
#endif
  }
  /*!
   * \brief Report an socket error.
   * \param msg The error message.
   */
  static void Error(const char *msg) {
    int errsv = GetLastError();
#ifdef _WIN32
    LOG(FATAL) << "Socket " << msg << " Error:WSAError-code=" << errsv;
#else
    LOG(FATAL) << "Socket " << msg << " Error:" << strerror(errsv);
#endif
  }

 protected:
  explicit Socket(SockType sockfd) : sockfd(sockfd) {
  }
};

/*!
 * \brief a wrapper of TCP socket that hopefully be cross platform
 */
class TCPSocket : public Socket {
 public:
  TCPSocket() : Socket(INVALID_SOCKET) {
  }
  /*!
   * \brief construct a TCP socket from existing descriptor
   * \param sockfd The descriptor
   */
  explicit TCPSocket(SockType sockfd) : Socket(sockfd) {
  }
  /*!
   * \brief enable/disable TCP keepalive
   * \param keepalive whether to set the keep alive option on
   */
  void SetKeepAlive(bool keepalive) {
    int opt = static_cast<int>(keepalive);
    if (setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE,
                   reinterpret_cast<char*>(&opt), sizeof(opt)) < 0) {
      Socket::Error("SetKeepAlive");
    }
  }
  /*!
   * \brief create the socket, call this before using socket
   * \param af domain
   */
  void Create(int af = PF_INET) {
    sockfd = socket(PF_INET, SOCK_STREAM, 0);
    if (sockfd == INVALID_SOCKET) {
      Socket::Error("Create");
    }
  }
  /*!
   * \brief perform listen of the socket
   * \param backlog backlog parameter
   */
  void Listen(int backlog = 16) {
    listen(sockfd, backlog);
  }
  /*!
   * \brief get a new connection
   * \return The accepted socket connection.
   */
  TCPSocket Accept() {
    SockType newfd = accept(sockfd, NULL, NULL);
    if (newfd == INVALID_SOCKET) {
      Socket::Error("Accept");
    }
    return TCPSocket(newfd);
  }

  /*!
   * \brief get a new connection
   * \param addr client address from which connection accepted
   * \return The accepted socket connection.
   */
  TCPSocket Accept(SockAddr *addr) {
    socklen_t addrlen = sizeof(addr->addr);
    SockType newfd = accept(sockfd, reinterpret_cast<sockaddr*>(&addr->addr),
             &addrlen);
    if (newfd == INVALID_SOCKET) {
      Socket::Error("Accept");
    }
    return TCPSocket(newfd);
  }
  /*!
   * \brief decide whether the socket is at OOB mark
   * \return 1 if at mark, 0 if not, -1 if an error occured
   */
  int AtMark() const {
#ifdef _WIN32
    unsigned long atmark;  // NOLINT(*)
    if (ioctlsocket(sockfd, SIOCATMARK, &atmark) != NO_ERROR) return -1;
#else
    int atmark;
    if (ioctl(sockfd, SIOCATMARK, &atmark) == -1) return -1;
#endif
    return static_cast<int>(atmark);
  }
  /*!
   * \brief connect to an address
   * \param addr the address to connect to
   * \return whether connect is successful
   */
  bool Connect(const SockAddr &addr) {
    return connect(sockfd, reinterpret_cast<const sockaddr*>(&addr.addr),
                   sizeof(addr.addr)) == 0;
  }
  /*!
   * \brief send data using the socket
   * \param buf_ the pointer to the buffer
   * \param len the size of the buffer
   * \param flag extra flags
   * \return size of data actually sent
   *         return -1 if error occurs
   */
  ssize_t Send(const void *buf_, size_t len, int flag = 0) {
    const char *buf = reinterpret_cast<const char*>(buf_);
    return send(sockfd, buf, static_cast<sock_size_t>(len), flag);
  }
  /*!
   * \brief receive data using the socket
   * \param buf_ the pointer to the buffer
   * \param len the size of the buffer
   * \param flags extra flags
   * \return size of data actually received
   *         return -1 if error occurs
   */
  ssize_t Recv(void *buf_, size_t len, int flags = 0) {
    char *buf = reinterpret_cast<char*>(buf_);
    return recv(sockfd, buf, static_cast<sock_size_t>(len), flags);
  }
  /*!
   * \brief peform block write that will attempt to send all data out
   *    can still return smaller than request when error occurs
   * \param buf_ the pointer to the buffer
   * \param len the size of the buffer
   * \return size of data actually sent
   */
  size_t SendAll(const void *buf_, size_t len) {
    const char *buf = reinterpret_cast<const char*>(buf_);
    size_t ndone = 0;
    while (ndone <  len) {
      ssize_t ret = send(sockfd, buf, static_cast<ssize_t>(len - ndone), 0);
      if (ret == -1) {
        if (LastErrorWouldBlock()) return ndone;
        Socket::Error("SendAll");
      }
      buf += ret;
      ndone += ret;
    }
    return ndone;
  }
  /*!
   * \brief peform block read that will attempt to read all data
   *    can still return smaller than request when error occurs
   * \param buf_ the buffer pointer
   * \param len length of data to recv
   * \return size of data actually sent
   */
  size_t RecvAll(void *buf_, size_t len) {
    char *buf = reinterpret_cast<char*>(buf_);
    size_t ndone = 0;
    while (ndone <  len) {
      ssize_t ret = recv(sockfd, buf,
                         static_cast<sock_size_t>(len - ndone), MSG_WAITALL);
      if (ret == -1) {
        if (LastErrorWouldBlock())  {
          LOG(FATAL) << "would block";
          return ndone;
        }
        Socket::Error("RecvAll");
      }
      if (ret == 0) return ndone;
      buf += ret;
      ndone += ret;
    }
    return ndone;
  }
  /*!
   * \brief Send the data to remote.
   * \param data The data to be sent.
   */
  void SendBytes(std::string data) {
    int datalen = data.length();
    CHECK_EQ(SendAll(&datalen, sizeof(datalen)), sizeof(datalen));
    CHECK_EQ(SendAll(data.c_str(), datalen), datalen);
  }
  /*!
   * \brief Receive the data to remote.
   * \return The data received.
   */
  std::string RecvBytes() {
    int datalen = 0;
    CHECK_EQ(RecvAll(&datalen, sizeof(datalen)), sizeof(datalen));
    std::string data;
    data.resize(datalen);
    CHECK_EQ(RecvAll(&data[0], datalen), datalen);
    return data;
  }
};

/*! \brief helper data structure to perform select */
struct SelectHelper {
 public:
  SelectHelper(void) {
    FD_ZERO(&read_set);
    FD_ZERO(&write_set);
    FD_ZERO(&except_set);
    maxfd = 0;
  }
  /*!
   * \brief add file descriptor to watch for read
   * \param fd file descriptor to be watched
   */
  inline void WatchRead(TCPSocket::SockType fd) {
    FD_SET(fd, &read_set);
    if (fd > maxfd) maxfd = fd;
  }
  /*!
   * \brief add file descriptor to watch for write
   * \param fd file descriptor to be watched
   */
  inline void WatchWrite(TCPSocket::SockType fd) {
    FD_SET(fd, &write_set);
    if (fd > maxfd) maxfd = fd;
  }
  /*!
   * \brief add file descriptor to watch for exception
   * \param fd file descriptor to be watched
   */
  inline void WatchException(TCPSocket::SockType fd) {
    FD_SET(fd, &except_set);
    if (fd > maxfd) maxfd = fd;
  }
  /*!
   * \brief Check if the descriptor is ready for read
   * \param fd file descriptor to check status
   */
  inline bool CheckRead(TCPSocket::SockType fd) const {
    return FD_ISSET(fd, &read_set) != 0;
  }
  /*!
   * \brief Check if the descriptor is ready for write
   * \param fd file descriptor to check status
   */
  inline bool CheckWrite(TCPSocket::SockType fd) const {
    return FD_ISSET(fd, &write_set) != 0;
  }
  /*!
   * \brief Check if the descriptor has any exception
   * \param fd file descriptor to check status
   */
  inline bool CheckExcept(TCPSocket::SockType fd) const {
    return FD_ISSET(fd, &except_set) != 0;
  }
  /*!
   * \brief wait for exception event on a single descriptor
   * \param fd the file descriptor to wait the event for
   * \param timeout the timeout counter, can be 0, which means wait until the event happen
   * \return 1 if success, 0 if timeout, and -1 if error occurs
   */
  inline static int WaitExcept(TCPSocket::SockType fd, long timeout = 0) { // NOLINT(*)
    fd_set wait_set;
    FD_ZERO(&wait_set);
    FD_SET(fd, &wait_set);
    return Select_(static_cast<int>(fd + 1),
                   NULL, NULL, &wait_set, timeout);
  }
  /*!
   * \brief peform select on the set defined
   * \param select_read whether to watch for read event
   * \param select_write whether to watch for write event
   * \param select_except whether to watch for exception event
   * \param timeout specify timeout in micro-seconds(ms) if equals 0, means select will always block
   * \return number of active descriptors selected,
   *         return -1 if error occurs
   */
  inline int Select(long timeout = 0) {  // NOLINT(*)
    int ret =  Select_(static_cast<int>(maxfd + 1),
                       &read_set, &write_set, &except_set, timeout);
    if (ret == -1) {
      Socket::Error("Select");
    }
    return ret;
  }

 private:
  inline static int Select_(int maxfd, fd_set *rfds,
                            fd_set *wfds, fd_set *efds, long timeout) { // NOLINT(*)
#if !defined(_WIN32)
    CHECK(maxfd < FD_SETSIZE) << "maxfd must be smaller than FDSETSIZE";
#endif
    if (timeout == 0) {
      return select(maxfd, rfds, wfds, efds, NULL);
    } else {
      timeval tm;
      tm.tv_usec = (timeout % 1000) * 1000;
      tm.tv_sec = timeout / 1000;
      return select(maxfd, rfds, wfds, efds, &tm);
    }
  }

  TCPSocket::SockType maxfd;
  fd_set read_set, write_set, except_set;
};

}  // namespace common
}  // namespace tvm
#endif  // TVM_COMMON_SOCKET_H_
