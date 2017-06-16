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
from . import util, cc_compiler
from ..module import load as _load_module
from .._ffi.function import _init_api, register_func
from .._ffi.ndarray import context as _context

RPC_MAGIC = 0xff271
RPC_SESS_MASK = 128

def _serve_loop(sock, addr):
    """Server loop"""
    sockfd = sock.fileno()
    temp = util.tempdir()
    # pylint: disable=unused-variable
    @register_func("tvm.contrib.rpc.server.upload")
    def upload(file_name, blob):
        """Upload the blob to remote temp file"""
        path = temp.relpath(file_name)
        with open(path, "wb") as out_file:
            out_file.write(blob)
        logging.info("upload %s", path)

    @register_func("tvm.contrib.rpc.server.download")
    def download(file_name):
        """Download file from remote"""
        path = temp.relpath(file_name)
        dat = bytearray(open(path, "rb").read())
        logging.info("download %s", path)
        return dat

    @register_func("tvm.contrib.rpc.server.load_module")
    def load_module(file_name):
        """Load module from remote side."""
        path = temp.relpath(file_name)
        # Try create a shared library in remote
        if path.endswith('.o'):
            logging.info('Create shared library based on %s', path)
            cc_compiler.create_shared(path + '.so', path)
            path += '.so'

        m = _load_module(path)
        logging.info("load_module %s", path)
        return m

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
    return b''.join(res)


def _listen_loop(sock):
    """Lisenting loop"""
    while True:
        conn, addr = sock.accept()
        logging.info("RPCServer: connection from %s", addr)
        conn.sendall(struct.pack('@i', RPC_MAGIC))
        magic = struct.unpack('@i', _recvall(conn, 4))[0]
        if magic != RPC_MAGIC:
            conn.close()
            continue
        logging.info("Connection from %s", addr)
        process = multiprocessing.Process(target=_serve_loop, args=(conn, addr))
        process.deamon = True
        process.start()
        # close from our side.
        conn.close()


class Server(object):
    """Start RPC server on a seperate process.

    This is a simple python implementation based on multi-processing.
    It is also possible to implement a similar C based sever with
    TVM runtime which does not depend on the python.

    Parameter
    ---------
    host : str
        The host url of the server.

    port : int
        The port to be bind to

    port_end : int, optional
        The end port to search
    """
    def __init__(self, host, port=9091, port_end=9199):
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
        self.host = host
        self.proc = multiprocessing.Process(target=_listen_loop, args=(self.sock,))
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
        self._upload_func = None
        self._download_func = None

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

        if not self._upload_func:
            self._upload_func = self.get_function(
                "tvm.contrib.rpc.server.upload")
        self._upload_func(target, blob)

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
        if not self._download_func:
            self._download_func = self.get_function(
                "tvm.contrib.rpc.server.download")
        return self._download_func(path)

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


def connect(url, port):
    """Connect to RPC Server

    Parameters
    ----------
    url : str
        The url of the host

    port : int
        The port to connect to

    Returns
    -------
    sess : RPCSession
        The connected session.
    """
    try:
        sess = _Connect(url, port)
    except NameError:
        import sys
        sys.exit('FATAL: Please compile with USE_RPC=1')
    return RPCSession(sess)

_init_api("tvm.contrib.rpc")
