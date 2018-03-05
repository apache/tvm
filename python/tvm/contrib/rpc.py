"""RPC interface for easy testing.

RPC enables connect to a remote server, upload and launch functions.
This is useful to for cross-compile and remote testing,
The compiler stack runs on local server, while we use RPC server
to run on remote runtime which don't have a compiler available.

The test program compiles the program on local server,
upload and run remote RPC server, get the result back to verify correctness.
"""
from __future__ import absolute_import

import os
import socket
import struct
import logging
import multiprocessing
import subprocess
import time
from . import util, cc, tar
from ..module import load as _load_module
from .._ffi.function import _init_api, register_func
from .._ffi.ndarray import context as _context
from .._ffi.base import py_str

RPC_MAGIC = 0xff271
RPC_SESS_MASK = 128

def _server_env():
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
        # Try create a shared library in remote
        if path.endswith(".o"):
            logging.info("Create shared library based on %s", path)
            cc.create_shared(path + ".so", path)
            path += ".so"
        elif path.endswith(".tar"):
            tar_temp = util.tempdir()
            tar.untar(path, tar_temp.temp_dir)
            files = [tar_temp.relpath(x) for x in tar_temp.listdir()]
            cc.create_shared(path + ".so", files)
            path += ".so"
        m = _load_module(path)
        logging.info("load_module %s", path)
        return m
    return temp


def _serve_loop(sock, addr):
    """Server loop"""
    sockfd = sock.fileno()
    temp = _server_env()
    _ServerLoop(sockfd)
    temp.remove()
    logging.info("Finish serving %s", addr)


def _recvall(sock, nbytes):
    res = []
    nread = 0
    while nread < nbytes:
        chunk = sock.recv(min(nbytes - nread, 1024))
        nread += len(chunk)
        res.append(chunk)
    return b"".join(res)


def _listen_loop(sock, exclusive):
    """Lisenting loop"""
    last_proc = None
    while True:
        conn, addr = sock.accept()

        if last_proc and last_proc.is_alive() and exclusive:
            logging.info("Kill last call")
            last_proc.terminate()

        logging.info("RPCServer: connection from %s", addr)
        magic = struct.unpack("@i", _recvall(conn, 4))[0]
        if magic != RPC_MAGIC:
            conn.close()
            continue
        keylen = struct.unpack("@i", _recvall(conn, 4))[0]
        key = py_str(_recvall(conn, keylen))
        if not key.startswith("client:"):
            conn.sendall(struct.pack("@i", RPC_MAGIC + 2))
        else:
            conn.sendall(struct.pack("@i", RPC_MAGIC))
        logging.info("Connection from %s", addr)

        process = multiprocessing.Process(target=_serve_loop, args=(conn, addr))
        process.deamon = True
        process.start()
        last_proc = process
        # close from our side.
        conn.close()


def _connect_proxy_loop(addr, key):
    key = "server:" + key
    while True:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(addr)
        sock.sendall(struct.pack("@i", RPC_MAGIC))
        sock.sendall(struct.pack("@i", len(key)))
        sock.sendall(key.encode("utf-8"))
        magic = struct.unpack("@i", _recvall(sock, 4))[0]
        if magic == RPC_MAGIC + 1:
            raise RuntimeError("key: %s has already been used in proxy" % key)
        elif magic == RPC_MAGIC + 2:
            logging.info("RPCProxy do not have matching client key %s", key)
        elif magic != RPC_MAGIC:
            raise RuntimeError("%s is not RPC Proxy" % str(addr))
        logging.info("RPCProxy connected to %s", str(addr))
        process = multiprocessing.Process(target=_serve_loop, args=(sock, addr))
        process.deamon = True
        process.start()
        process.join()


def _popen(cmd):
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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

    exclusive : bool, optional
        If this is enabled, the server will kill old connection
        when new connection comes. This can make sure the current call
        monopolize the hardware resource.

    key : str, optional
        The key used to identify the server in Proxy connection.
    """
    def __init__(self,
                 host,
                 port=9091,
                 port_end=9199,
                 is_proxy=False,
                 use_popen=False,
                 exclusive=False,
                 key=""):
        try:
            if _ServerLoop is None:
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
                target=_listen_loop, args=(self.sock, exclusive))
            self.proc.deamon = True
            self.proc.start()
        else:
            self.proc = multiprocessing.Process(
                target=_connect_proxy_loop, args=((host, port), key))
            self.proc.deamon = True
            self.proc.start()

    def terminate(self):
        """Terminate the server process"""
        if self.proc:
            self.proc.terminate()
            self.proc = None

    def __del__(self):
        self.terminate()


class RPCSession(object):
    """RPC Client session module

    Do not directly create the obhect, call connect
    """
    # pylint: disable=invalid-name
    def __init__(self, sess):
        self._sess = sess
        self._tbl_index = _SessTableIndex(sess)
        self._remote_funcs = {}

    def get_function(self, name):
        """Get function from the session.

        Parameters
        ----------
        name : str
            The name of the function

        Returns
        -------
        f : Function
            The result function.
        """
        return self._sess.get_function(name)

    def context(self, dev_type, dev_id=0):
        """Construct a remote context.

        Parameters
        ----------
        dev_type: int or str

        dev_id: int, optional

        Returns
        -------
        ctx: TVMContext
            The corresponding encoded remote context.
        """
        ctx = _context(dev_type, dev_id)
        encode = (self._tbl_index + 1) * RPC_SESS_MASK
        ctx.device_type += encode
        ctx._rpc_sess = self
        return ctx

    def cpu(self, dev_id=0):
        """Construct remote CPU device."""
        return self.context(1, dev_id)

    def gpu(self, dev_id=0):
        """Construct remote GPU device."""
        return self.context(2, dev_id)

    def cl(self, dev_id=0):
        """Construct remote OpenCL device."""
        return self.context(4, dev_id)

    def metal(self, dev_id=0):
        """Construct remote Metal device."""
        return self.context(8, dev_id)

    def opengl(self, dev_id=0):
        """Construct remote OpenGL device."""
        return self.context(11, dev_id)

    def ext_dev(self, dev_id=0):
        """Construct remote extension device."""
        return self.context(12, dev_id)

    def upload(self, data, target=None):
        """Upload file to remote runtime temp folder

        Parameters
        ----------
        data : str or bytearray
            The file name or binary in local to upload.

        target : str, optional
            The path in remote
        """
        if isinstance(data, bytearray):
            if not target:
                raise ValueError("target must present when file is a bytearray")
            blob = data
        else:
            blob = bytearray(open(data, "rb").read())
            if not target:
                target = os.path.basename(data)

        if "upload" not in self._remote_funcs:
            self._remote_funcs["upload"] = self.get_function(
                "tvm.contrib.rpc.server.upload")
        self._remote_funcs["upload"](target, blob)

    def download(self, path):
        """Download file from remote temp folder.

        Parameters
        ----------
        path : str
            The relative location to remote temp folder.

        Returns
        -------
        blob : bytearray
            The result blob from the file.
        """
        if "download" not in self._remote_funcs:
            self._remote_funcs["download"] = self.get_function(
                "tvm.contrib.rpc.server.download")
        return self._remote_funcs["download"](path)

    def load_module(self, path):
        """Load a remote module, the file need to be uploaded first.

        Parameters
        ----------
        path : str
            The relative location to remote temp folder.

        Returns
        -------
        m : Module
            The remote module containing remote function.
        """
        return _LoadRemoteModule(self._sess, path)


def connect(url, port, key=""):
    """Connect to RPC Server

    Parameters
    ----------
    url : str
        The url of the host

    port : int
        The port to connect to

    key : str, optional
        Additional key to match server

    Returns
    -------
    sess : RPCSession
        The connected session.
    """
    try:
        sess = _Connect(url, port, key)
    except NameError:
        raise RuntimeError("Please compile with USE_RPC=1")
    return RPCSession(sess)

_init_api("tvm.contrib.rpc")
