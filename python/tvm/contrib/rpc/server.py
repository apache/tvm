"""RPC server implementation.

Note
----
Server is TCP based with the following protocol:
- Initial handshake to the peer
  - [RPC_MAGIC, keysize(int32), key-bytes]
- The key is in format
   - {server|client}:device-type[:random-key] [-timeout=timeout]
"""
from __future__ import absolute_import

import os
import ctypes
import socket
import select
import struct
import logging
import multiprocessing
import subprocess
import time

from ..._ffi.function import register_func
from ..._ffi.base import py_str
from ..._ffi.libinfo import find_lib_path
from ...module import load as _load_module
from .. import util
from . import base
from . base import TrackerCode

def _server_env(load_library):
    """Server environment function return temp dir"""
    temp = util.tempdir()
    # pylint: disable=unused-variable
    @register_func("tvm.contrib.rpc.server.workpath")
    def get_workpath(path):
        return temp.relpath(path)

    @register_func("tvm.contrib.rpc.server.load_module", override=True)
    def load_module(file_name):
        """Load module from remote side."""
        path = temp.relpath(file_name)
        m = _load_module(path)
        logging.info("load_module %s", path)
        return m

    libs = []
    load_library = load_library.split(":") if load_library else []
    for file_name in load_library:
        file_name = find_lib_path(file_name)[0]
        libs.append(ctypes.CDLL(file_name, ctypes.RTLD_GLOBAL))
        logging.info("Load additional library %s", file_name)
    temp.libs = libs
    return temp


def _serve_loop(sock, addr, load_library):
    """Server loop"""
    sockfd = sock.fileno()
    temp = _server_env(load_library)
    base._ServerLoop(sockfd)
    temp.remove()
    logging.info("Finish serving %s", addr)


def _parse_server_opt(opts):
    # parse client options
    ret = {}
    for kv in opts:
        if kv.startswith("-timeout="):
            ret["timeout"] = float(kv[9:])
    return ret


def _listen_loop(sock, port, rpc_key, tracker_addr, load_library):
    """Lisenting loop of the server master."""
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
                          [TrackerCode.PUT, rpc_key, (port, matchkey)])
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
                    # it means the key is aqquired by a client but not used.
                    if matchkey not in pending_keys:
                        unmatch_period_count += 1
                    else:
                        unmatch_period_count = 0
                    # regenerate match key if key is aqquired but not used for a while
                    if unmatch_period_count * ping_period > unmatch_timeout + ping_period:
                        logging.info("RPCServer: no incoming connections, regenerate key ...")
                        matchkey = base.random_key(rpc_key + ":", old_keyset)
                        base.sendjson(tracker_conn,
                                      [TrackerCode.PUT, rpc_key, (port, matchkey)])
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
                logging.info("RPCServer: mismatch key from %s", addr)
                continue
            else:
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

        # step 3: serving
        logging.info("RPCServer: connection from %s", addr)
        server_proc = multiprocessing.Process(target=_serve_loop, args=(conn, addr, load_library))
        server_proc.deamon = True
        server_proc.start()
        # close from our side.
        conn.close()
        # wait until server process finish or timeout
        server_proc.join(opts.get("timeout", None))
        if server_proc.is_alive():
            logging.info("RPCServer: Timeout in RPC session, kill..")
            server_proc.terminate()


def _connect_proxy_loop(addr, key, load_library):
    key = "server:" + key
    retry_count = 0
    max_retry = 5
    retry_period = 5
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(addr)
            sock.sendall(struct.pack("<i", base.RPC_MAGIC))
            sock.sendall(struct.pack("<i", len(key)))
            sock.sendall(key.encode("utf-8"))
            magic = struct.unpack("<i", base.recvall(sock, 4))[0]
            if magic == base.RPC_CODE_DUPLICATE:
                raise RuntimeError("key: %s has already been used in proxy" % key)
            elif magic == base.RPC_CODE_MISMATCH:
                logging.info("RPCProxy do not have matching client key %s", key)
            elif magic != base.RPC_CODE_SUCCESS:
                raise RuntimeError("%s is not RPC Proxy" % str(addr))
            keylen = struct.unpack("<i", base.recvall(sock, 4))[0]
            remote_key = py_str(base.recvall(sock, keylen))
            opts = _parse_server_opt(remote_key.split()[1:])
            logging.info("RPCProxy connected to %s", str(addr))
            process = multiprocessing.Process(
                target=_serve_loop, args=(sock, addr, load_library))
            process.deamon = True
            process.start()
            sock.close()
            process.join(opts.get("timeout", None))
            if process.is_alive():
                logging.info("RPCProxyServer: Timeout in RPC session, kill..")
                process.terminate()
            retry_count = 0
        except (socket.error, IOError) as err:
            retry_count += 1
            logging.info("Error encountered %s, retry in %g sec", str(err), retry_period)
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
    """Start RPC server on a seperate process.

    This is a simple python implementation based on multi-processing.
    It is also possible to implement a similar C based sever with
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

    key : str, optional
        The key used to identify the server in Proxy connection.

    load_library : str, optional
        List of additional libraries to be loaded during execution.
    """
    def __init__(self,
                 host,
                 port=9091,
                 port_end=9199,
                 is_proxy=False,
                 use_popen=False,
                 tracker_addr=None,
                 key="",
                 load_library=None):
        try:
            if base._ServerLoop is None:
                raise RuntimeError("Please compile with USE_RPC=1")
        except NameError:
            raise RuntimeError("Please compile with USE_RPC=1")
        self.host = host
        self.port = port
        self.libs = []

        if use_popen:
            cmd = ["python",
                   "-m", "tvm.exec.rpc_server",
                   "--host=%s" % host,
                   "--port=%s" % port]
            if tracker_addr:
                assert key
                cmd += ["--tracker=%s:%d" % tracker_addr,
                        "--key=%s" % key]
            if load_library:
                cmd += ["--load-libary", load_library]
            self.proc = multiprocessing.Process(
                target=subprocess.check_call, args=(cmd,))
            self.proc.deamon = True
            self.proc.start()
            time.sleep(1)
        elif not is_proxy:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.port = None
            for my_port in range(port, port_end):
                try:
                    sock.bind((host, my_port))
                    self.port = my_port
                    break
                except socket.error as sock_err:
                    if sock_err.errno in [98, 48]:
                        continue
                    else:
                        raise sock_err
                if not self.port:
                    raise ValueError("cannot bind to any port in [%d, %d)" % (port, port_end))
            logging.info("RPCServer: bind to %s:%d", host, self.port)
            sock.listen(1)
            self.sock = sock
            self.proc = multiprocessing.Process(
                target=_listen_loop, args=(
                    self.sock, self.port, key, tracker_addr, load_library))
            self.proc.deamon = True
            self.proc.start()
        else:
            self.proc = multiprocessing.Process(
                target=_connect_proxy_loop, args=((host, port), key, load_library))
            self.proc.deamon = True
            self.proc.start()

    def terminate(self):
        """Terminate the server process"""
        if self.proc:
            self.proc.terminate()
            self.proc = None

    def __del__(self):
        self.terminate()
