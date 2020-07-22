# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""RPC server implementation.

Note
----
Server is TCP based with the following protocol:
- Initial handshake to the peer
  - [RPC_MAGIC, keysize(int32), key-bytes]
- The key is in format
   - {server|client}:device-type[:random-key] [-timeout=timeout]
"""
# pylint: disable=invalid-name
import os
import ctypes
import socket
import select
import struct
import logging
import multiprocessing
import subprocess
import time
import sys
import signal
import platform
import tvm._ffi

from tvm._ffi.base import py_str
from tvm._ffi.libinfo import find_lib_path
from tvm.runtime.module import load_module as _load_module
from tvm.contrib import util
from . import _ffi_api
from . import base
from . base import TrackerCode

logger = logging.getLogger('RPCServer')

def _server_env(load_library, work_path=None):
    """Server environment function return temp dir"""
    if work_path:
        temp = work_path
    else:
        temp = util.tempdir()

    # pylint: disable=unused-variable
    @tvm._ffi.register_func("tvm.rpc.server.workpath", override=True)
    def get_workpath(path):
        return temp.relpath(path)

    @tvm._ffi.register_func("tvm.rpc.server.load_module", override=True)
    def load_module(file_name):
        """Load module from remote side."""
        path = temp.relpath(file_name)
        m = _load_module(path)
        logger.info("load_module %s", path)
        return m

    libs = []
    load_library = load_library.split(":") if load_library else []
    for file_name in load_library:
        file_name = find_lib_path(file_name)[0]
        libs.append(ctypes.CDLL(file_name, ctypes.RTLD_GLOBAL))
        logger.info("Load additional library %s", file_name)
    temp.libs = libs
    return temp

def _serve_loop(sock, addr, load_library, work_path=None):
    """Server loop"""
    sockfd = sock.fileno()
    temp = _server_env(load_library, work_path)
    _ffi_api.ServerLoop(sockfd)
    if not work_path:
        temp.remove()
    logger.info("Finish serving %s", addr)

def _parse_server_opt(opts):
    # parse client options
    ret = {}
    for kv in opts:
        if kv.startswith("-timeout="):
            ret["timeout"] = float(kv[9:])
    return ret

def _listen_loop(sock, port, rpc_key, tracker_addr, load_library, custom_addr):
    """Listening loop of the server master."""
    def _accept_conn(listen_sock, tracker_conn, ping_period=2):
        """Accept connection from the other places.

        Parameters
        ----------
        listen_sock: Socket
            The socket used by listening process.

        tracker_conn : connnection to tracker
            Tracker connection

        ping_period : float, optional
            ping tracker every k seconds if no connection is accepted.
        """
        old_keyset = set()
        # Report resource to tracker
        if tracker_conn:
            matchkey = base.random_key(rpc_key + ":")
            base.sendjson(tracker_conn,
                          [TrackerCode.PUT, rpc_key, (port, matchkey), custom_addr])
            assert base.recvjson(tracker_conn) == TrackerCode.SUCCESS
        else:
            matchkey = rpc_key

        unmatch_period_count = 0
        unmatch_timeout = 4
        # Wait until we get a valid connection
        while True:
            if tracker_conn:
                trigger = select.select([listen_sock], [], [], ping_period)
                if not listen_sock in trigger[0]:
                    base.sendjson(tracker_conn, [TrackerCode.GET_PENDING_MATCHKEYS])
                    pending_keys = base.recvjson(tracker_conn)
                    old_keyset.add(matchkey)
                    # if match key not in pending key set
                    # it means the key is acquired by a client but not used.
                    if matchkey not in pending_keys:
                        unmatch_period_count += 1
                    else:
                        unmatch_period_count = 0
                    # regenerate match key if key is acquired but not used for a while
                    if unmatch_period_count * ping_period > unmatch_timeout + ping_period:
                        logger.info("no incoming connections, regenerate key ...")
                        matchkey = base.random_key(rpc_key + ":", old_keyset)
                        base.sendjson(tracker_conn,
                                      [TrackerCode.PUT, rpc_key, (port, matchkey),
                                       custom_addr])
                        assert base.recvjson(tracker_conn) == TrackerCode.SUCCESS
                        unmatch_period_count = 0
                    continue
            conn, addr = listen_sock.accept()
            magic = struct.unpack("<i", base.recvall(conn, 4))[0]
            if magic != base.RPC_MAGIC:
                conn.close()
                continue
            keylen = struct.unpack("<i", base.recvall(conn, 4))[0]
            key = py_str(base.recvall(conn, keylen))
            arr = key.split()
            expect_header = "client:" + matchkey
            server_key = "server:" + rpc_key
            if arr[0] != expect_header:
                conn.sendall(struct.pack("<i", base.RPC_CODE_MISMATCH))
                conn.close()
                logger.warning("mismatch key from %s", addr)
                continue
            conn.sendall(struct.pack("<i", base.RPC_CODE_SUCCESS))
            conn.sendall(struct.pack("<i", len(server_key)))
            conn.sendall(server_key.encode("utf-8"))
            return conn, addr, _parse_server_opt(arr[1:])

    # Server logic
    tracker_conn = None
    while True:
        try:
            # step 1: setup tracker and report to tracker
            if tracker_addr and tracker_conn is None:
                tracker_conn = base.connect_with_retry(tracker_addr)
                tracker_conn.sendall(struct.pack("<i", base.RPC_TRACKER_MAGIC))
                magic = struct.unpack("<i", base.recvall(tracker_conn, 4))[0]
                if magic != base.RPC_TRACKER_MAGIC:
                    raise RuntimeError("%s is not RPC Tracker" % str(tracker_addr))
                # report status of current queue
                cinfo = {"key" : "server:" + rpc_key}
                base.sendjson(tracker_conn,
                              [TrackerCode.UPDATE_INFO, cinfo])
                assert base.recvjson(tracker_conn) == TrackerCode.SUCCESS

            # step 2: wait for in-coming connections
            conn, addr, opts = _accept_conn(sock, tracker_conn)
        except (socket.error, IOError):
            # retry when tracker is dropped
            if tracker_conn:
                tracker_conn.close()
                tracker_conn = None
            continue
        except RuntimeError as exc:
            raise exc

        # step 3: serving
        work_path = util.tempdir()
        logger.info("connection from %s", addr)
        server_proc = multiprocessing.Process(target=_serve_loop,
                                              args=(conn, addr, load_library, work_path))
        server_proc.deamon = True
        server_proc.start()
        # close from our side.
        conn.close()
        # wait until server process finish or timeout
        server_proc.join(opts.get("timeout", None))
        if server_proc.is_alive():
            logger.info("Timeout in RPC session, kill..")
            # pylint: disable=import-outside-toplevel
            import psutil
            parent = psutil.Process(server_proc.pid)
            # terminate worker childs
            for child in parent.children(recursive=True):
                child.terminate()
            # terminate the worker
            server_proc.terminate()
        work_path.remove()


def _connect_proxy_loop(addr, key, load_library):
    key = "server:" + key
    retry_count = 0
    max_retry = 5
    retry_period = 5
    while True:
        try:
            sock = socket.socket(base.get_addr_family(addr), socket.SOCK_STREAM)
            sock.connect(addr)
            sock.sendall(struct.pack("<i", base.RPC_MAGIC))
            sock.sendall(struct.pack("<i", len(key)))
            sock.sendall(key.encode("utf-8"))
            magic = struct.unpack("<i", base.recvall(sock, 4))[0]
            if magic == base.RPC_CODE_DUPLICATE:
                raise RuntimeError("key: %s has already been used in proxy" % key)

            if magic == base.RPC_CODE_MISMATCH:
                logger.warning("RPCProxy do not have matching client key %s", key)
            elif magic != base.RPC_CODE_SUCCESS:
                raise RuntimeError("%s is not RPC Proxy" % str(addr))
            keylen = struct.unpack("<i", base.recvall(sock, 4))[0]
            remote_key = py_str(base.recvall(sock, keylen))
            opts = _parse_server_opt(remote_key.split()[1:])
            logger.info("connected to %s", str(addr))
            process = multiprocessing.Process(
                target=_serve_loop, args=(sock, addr, load_library))
            process.deamon = True
            process.start()
            sock.close()
            process.join(opts.get("timeout", None))
            if process.is_alive():
                logger.info("Timeout in RPC session, kill..")
                process.terminate()
            retry_count = 0
        except (socket.error, IOError) as err:
            retry_count += 1
            logger.warning("Error encountered %s, retry in %g sec", str(err), retry_period)
            if retry_count > max_retry:
                raise RuntimeError("Maximum retry error: last error: %s" % str(err))
            time.sleep(retry_period)

def _popen(cmd):
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            env=os.environ)
    (out, _) = proc.communicate()
    if proc.returncode != 0:
        msg = "Server invoke error:\n"
        msg += out
        raise RuntimeError(msg)


class Server(object):
    """Start RPC server on a separate process.

    This is a simple python implementation based on multi-processing.
    It is also possible to implement a similar C based server with
    TVM runtime which does not depend on the python.

    Parameters
    ----------
    host : str
        The host url of the server.

    port : int
        The port to be bind to

    port_end : int, optional
        The end port to search

    is_proxy : bool, optional
        Whether the address specified is a proxy.
        If this is true, the host and port actually corresponds to the
        address of the proxy server.

    use_popen : bool, optional
        Whether to use Popen to start a fresh new process instead of fork.
        This is recommended to switch on if we want to do local RPC demonstration
        for GPU devices to avoid fork safety issues.

    tracker_addr: Tuple (str, int) , optional
        The address of RPC Tracker in tuple(host, ip) format.
        If is not None, the server will register itself to the tracker.

    key : str, optional
        The key used to identify the device type in tracker.

    load_library : str, optional
        List of additional libraries to be loaded during execution.

    custom_addr: str, optional
        Custom IP Address to Report to RPC Tracker

    silent: bool, optional
        Whether run this server in silent mode.
    """
    def __init__(self,
                 host,
                 port=9091,
                 port_end=9199,
                 is_proxy=False,
                 use_popen=False,
                 tracker_addr=None,
                 key="",
                 load_library=None,
                 custom_addr=None,
                 silent=False,
                 utvm_dev_id=None,
                 utvm_dev_config_args=None,
                 ):
        try:
            if _ffi_api.ServerLoop is None:
                raise RuntimeError("Please compile with USE_RPC=1")
        except NameError:
            raise RuntimeError("Please compile with USE_RPC=1")
        self.host = host
        self.port = port
        self.libs = []
        self.custom_addr = custom_addr
        self.use_popen = use_popen

        if silent:
            logger.setLevel(logging.ERROR)

        if use_popen:
            cmd = [sys.executable,
                   "-m", "tvm.exec.rpc_server",
                   "--host=%s" % host,
                   "--port=%s" % port]
            if tracker_addr:
                assert key
                cmd += ["--tracker=%s:%d" % tracker_addr,
                        "--key=%s" % key]
            if load_library:
                cmd += ["--load-library", load_library]
            if custom_addr:
                cmd += ["--custom-addr", custom_addr]
            if silent:
                cmd += ["--silent"]
            if utvm_dev_id is not None:
                assert utvm_dev_config_args is not None
                cmd += [f"--utvm-dev-id={utvm_dev_id}"]
                cmd += [f"--utvm-dev-config-args={utvm_dev_config_args}"]

            # prexec_fn is not thread safe and may result in deadlock.
            # python 3.2 introduced the start_new_session parameter as
            # an alternative to the common use case of
            # prexec_fn=os.setsid.  Once the minimum version of python
            # supported by TVM reaches python 3.2 this code can be
            # rewritten in favour of start_new_session.  In the
            # interim, stop the pylint diagnostic.
            #
            # pylint: disable=subprocess-popen-preexec-fn
            if platform.system() == "Windows":
                self.proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                self.proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
            time.sleep(0.5)
        elif not is_proxy:
            sock = socket.socket(base.get_addr_family((host, port)), socket.SOCK_STREAM)
            self.port = None
            for my_port in range(port, port_end):
                try:
                    sock.bind((host, my_port))
                    self.port = my_port
                    break
                except socket.error as sock_err:
                    if sock_err.errno in [98, 48]:
                        continue
                    raise sock_err
            if not self.port:
                raise ValueError("cannot bind to any port in [%d, %d)" % (port, port_end))
            logger.info("bind to %s:%d", host, self.port)
            sock.listen(1)
            self.sock = sock
            self.proc = multiprocessing.Process(
                target=_listen_loop, args=(
                    self.sock, self.port, key, tracker_addr, load_library,
                    self.custom_addr))
            self.proc.deamon = True
            self.proc.start()
        else:
            self.proc = multiprocessing.Process(
                target=_connect_proxy_loop, args=((host, port), key, load_library))
            self.proc.deamon = True
            self.proc.start()

    def terminate(self):
        """Terminate the server process"""
        if self.use_popen:
            if self.proc:
                if platform.system() == "Windows":
                    os.kill(self.proc.pid, signal.CTRL_C_EVENT)
                else:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc = None
        else:
            if self.proc:
                self.proc.terminate()
                self.proc = None

    def __del__(self):
        self.terminate()
