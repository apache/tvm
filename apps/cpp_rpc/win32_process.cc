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

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <cstdio>
#include <memory>
#include <conio.h>
#include <string>
#include <stdexcept>
#include <dmlc/logging.h>
#include "win32_process.h"
#include "rpc_server.h"

using namespace std::chrono;
using namespace tvm::runtime;

namespace {
// The prefix path for the memory mapped file used to store IPC information
const std::string kMemoryMapPrefix = "/MAPPED_FILE/TVM_RPC";
// Used to construct unique names for named resources in the parent process
const std::string kParent = "parent";
// Used to construct unique names for named resources in the child process
const std::string kChild = "child";
// The timeout of the WIN32 events, in the parent and the child
const milliseconds kEventTimeout(2000);

// Used to create unique WIN32 mmap paths and event names
int child_counter_ = 0;

/*!
 * \brief HandleDeleter Deleter for UniqueHandle smart pointer
 * \param handle The WIN32 HANDLE to manage
 */
struct HandleDeleter {
  void operator()(HANDLE handle) const {
    if (handle != INVALID_HANDLE_VALUE && handle != nullptr) {
      CloseHandle(handle);
    }
  }
};

/*!
 * \brief UniqueHandle Smart pointer to manage a WIN32 HANDLE
 */
using UniqueHandle = std::unique_ptr<void, HandleDeleter>;

/*!
 * \brief MakeUniqueHandle Helper method to construct a UniqueHandle
 * \param handle The WIN32 HANDLE to manage
 */
UniqueHandle MakeUniqueHandle(HANDLE handle) {
  if (handle == INVALID_HANDLE_VALUE || handle == nullptr) {
    return nullptr;
  }

  return UniqueHandle(handle);
}

/*!
 * \brief GetSocket Gets the socket info from the parent process and duplicates the socket
 * \param mmap_path The path to the memory mapped info set by the parent
 */
SOCKET GetSocket(const std::string& mmap_path) {
  WSAPROTOCOL_INFO protocol_info;
 
  const std::string parent_event_name = mmap_path + kParent;
  const std::string child_event_name = mmap_path + kChild;

  // Open the events
  UniqueHandle parent_file_mapping_event;
  if ((parent_file_mapping_event = MakeUniqueHandle(OpenEventA(SYNCHRONIZE, false, parent_event_name.c_str()))) == nullptr) {
    LOG(FATAL) << "OpenEvent() failed: " << GetLastError();
  }

  UniqueHandle child_file_mapping_event;
  if ((child_file_mapping_event = MakeUniqueHandle(OpenEventA(EVENT_MODIFY_STATE, false, child_event_name.c_str()))) == nullptr) {
    LOG(FATAL) << "OpenEvent() failed: " << GetLastError();
  }
  
  // Wait for the parent to set the event, notifying WSAPROTOCOL_INFO is ready to be read
  if (WaitForSingleObject(parent_file_mapping_event.get(), uint32_t(kEventTimeout.count())) != WAIT_OBJECT_0) {
      LOG(FATAL) << "WaitForSingleObject() failed: " << GetLastError();
  }

  const UniqueHandle file_map = MakeUniqueHandle(OpenFileMappingA(FILE_MAP_READ | FILE_MAP_WRITE,
                                                  false,
                                                  mmap_path.c_str()));
  if (!file_map) {
      LOG(INFO) << "CreateFileMapping() failed: " << GetLastError();
  }

  void* map_view = MapViewOfFile(file_map.get(),
                  FILE_MAP_READ | FILE_MAP_WRITE,
                  0, 0, 0);

  SOCKET sock_duplicated = INVALID_SOCKET;

  if (map_view != nullptr) {
    memcpy(&protocol_info, map_view, sizeof(WSAPROTOCOL_INFO));
    UnmapViewOfFile(map_view);

    // Creates the duplicate socket, that was created in the parent
    sock_duplicated = WSASocket(FROM_PROTOCOL_INFO,
                        FROM_PROTOCOL_INFO,
                        FROM_PROTOCOL_INFO,
                        &protocol_info,
                        0,
                        0);

    // Let the parent know we are finished dupicating the socket
    SetEvent(child_file_mapping_event.get());
  } else {
    LOG(FATAL) << "MapViewOfFile() failed: " << GetLastError();
  }

  return sock_duplicated;
}
}// Anonymous namespace

namespace tvm {
namespace runtime {
/*!
 * \brief SpawnRPCChild Spawns a child process with a given timeout to run
 * \param fd The client socket to duplicate in the child
 * \param timeout The time in seconds to wait for the child to complete before termination
 */
void SpawnRPCChild(SOCKET fd, seconds timeout) {
  STARTUPINFOA startup_info;
  
  memset(&startup_info, 0, sizeof(startup_info));
  startup_info.cb = sizeof(startup_info);

  std::string file_map_path = kMemoryMapPrefix + std::to_string(child_counter_++);

  const std::string parent_event_name = file_map_path + kParent;
  const std::string child_event_name = file_map_path + kChild;

  // Create an event to let the child know the socket info was set to the mmap file
  UniqueHandle parent_file_mapping_event;
  if ((parent_file_mapping_event = MakeUniqueHandle(CreateEventA(nullptr, true, false, parent_event_name.c_str()))) == nullptr) {
    LOG(FATAL) << "CreateEvent for parent file mapping failed";
  }

  UniqueHandle child_file_mapping_event;
  // An event to let the parent know the socket info was read from the mmap file
  if ((child_file_mapping_event = MakeUniqueHandle(CreateEventA(nullptr, true, false, child_event_name.c_str()))) == nullptr) {
    LOG(FATAL) << "CreateEvent for child file mapping failed";
  }

  char current_executable[MAX_PATH];

  // Get the full path of the current executable
  GetModuleFileNameA(nullptr, current_executable, MAX_PATH);

  std::string child_command_line = current_executable;
  child_command_line += " server --child_proc=";
  child_command_line += file_map_path;

  // CreateProcessA requires a non const char*, so we copy our std::string
  std::unique_ptr<char[]> command_line_ptr(new char[child_command_line.size() + 1]);
  strcpy(command_line_ptr.get(), child_command_line.c_str());

  PROCESS_INFORMATION child_process_info;
  if (CreateProcessA(nullptr,
                     command_line_ptr.get(),
                     nullptr,
                     nullptr,
                     false,
                     CREATE_NO_WINDOW,
                     nullptr,
                     nullptr,
                     &startup_info,
                     &child_process_info)) {
    // Child process and thread handles must be closed, so wrapped in RAII
    auto child_process_handle = MakeUniqueHandle(child_process_info.hProcess);
    auto child_process_thread_handle = MakeUniqueHandle(child_process_info.hThread);

    WSAPROTOCOL_INFO protocol_info;
    // Get info needed to duplicate the socket
    if (WSADuplicateSocket(fd,
                           child_process_info.dwProcessId,
                           &protocol_info) == SOCKET_ERROR) {
      LOG(FATAL) << "WSADuplicateSocket(): failed. Error =" << WSAGetLastError();
    }

    // Create a mmap file to store the info needed for duplicating the SOCKET in the child proc
    UniqueHandle file_map = MakeUniqueHandle(CreateFileMappingA(INVALID_HANDLE_VALUE,
                                                  nullptr,
                                                  PAGE_READWRITE,
                                                  0,
                                                  sizeof(WSAPROTOCOL_INFO),
                                                  file_map_path.c_str()));
    if (!file_map) {
      LOG(INFO) << "CreateFileMapping() failed: " << GetLastError();
    }

    if (GetLastError() == ERROR_ALREADY_EXISTS) {
      LOG(FATAL) << "CreateFileMapping(): mapping file already exists";
    } else {
      void* map_view = MapViewOfFile(file_map.get(), FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);

      if (map_view != nullptr) {
        memcpy(map_view, &protocol_info, sizeof(WSAPROTOCOL_INFO));
        UnmapViewOfFile(map_view);

        // Let child proc know the mmap file is ready to be read
        SetEvent(parent_file_mapping_event.get());
       
        // Wait for the child to finish reading mmap file
        if (WaitForSingleObject(child_file_mapping_event.get(), uint32_t(kEventTimeout.count())) != WAIT_OBJECT_0) {
          TerminateProcess(child_process_handle.get(), 0);
          LOG(FATAL) << "WaitForSingleObject for child file mapping timed out.  Terminating child process.";
        }
      } else {
        TerminateProcess(child_process_handle.get(), 0);
        LOG(FATAL) << "MapViewOfFile() failed: " << GetLastError();
      }
    }

    const DWORD process_timeout = timeout.count()
        ? uint32_t(duration_cast<milliseconds>(timeout).count())
        : INFINITE;

    // Wait for child process to exit, or hit configured timeout
    if (WaitForSingleObject(child_process_handle.get(), process_timeout) != WAIT_OBJECT_0) {
      LOG(INFO) << "Child process timeout.  Terminating.";
      TerminateProcess(child_process_handle.get(), 0);
    }
  } else {
    LOG(INFO) << "Create child process failed: " << GetLastError();
  }
}
/*!
 * \brief ChildProcSocketHandler Ran from the child process and runs server to handle the client socket
 * \param mmap_path The memory mapped file path that will contain the information to duplicate the client socket from the parent
 */
void ChildProcSocketHandler(const std::string& mmap_path) {
  SOCKET socket;

  // Set high thread priority to avoid the thread scheduler from
  // interfering with any measurements in the RPC server.
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
 
  if ((socket = GetSocket(mmap_path)) != INVALID_SOCKET) {
    tvm::runtime::ServerLoopFromChild(socket);
  }
  else {
    LOG(FATAL) << "GetSocket() failed";
  }
  
}
}  // namespace runtime
}  // namespace tvm