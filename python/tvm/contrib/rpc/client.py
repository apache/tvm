"""RPC client tools"""
from __future__ import absolute_import

import os
import socket
import struct

from . import base
from ..._ffi.base import TVMError
from ..._ffi.ndarray import context as _context


class RPCSession(object):
    """RPC Client session module

    Do not directly create the obhect, call connect
    """
    # pylint: disable=invalid-name
    def __init__(self, sess):
        self._sess = sess
        self._tbl_index = base._SessTableIndex(sess)
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
        encode = (self._tbl_index + 1) * base.RPC_SESS_MASK
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
        return base._LoadRemoteModule(self._sess, path)


class TrackerSession(object):
    """Tracker client session.

    Parameters
    ----------
    addr : tuple
        The address tuple
    """
    def __init__(self, addr):
        self._addr = addr
        self._sock = None
        self._max_request_retry = 5
        self._connect()

    def __del__(self):
        self.close()

    def _connect(self):
        self._sock = base.connect_with_retry(self._addr)
        self._sock.sendall(struct.pack("@i", base.RPC_TRACKER_MAGIC))
        magic = struct.unpack("@i", base.recvall(self._sock, 4))[0]
        if magic != base.RPC_TRACKER_MAGIC:
            raise RuntimeError("%s is not RPC Tracker" % str(self._addr))

    def close(self):
        """Close the tracker connection."""
        if self._sock:
            self._sock.close()
            self._sock = None

    def request(self, key, priority=1, session_timeout=0):
        """Request a new connection from the tracker.

        Parameters
        ----------
        key : str
            The type key of the device.

        priority : int, optional
            The priority of the request.

        session_timeout : float, optional
            The duration of the session, allows server to kill
            the connection when duration is longer than this value.
            When duration is zero, it means the request must always be kept alive.
        """
        for _ in range(self._max_request_retry):
            try:
                if self._sock is None:
                    self._connect()
                base.sendjson(self._sock,
                              [base.TrackerCode.REQUEST, key, "", priority])
                value = base.recvjson(self._sock)
                if value[0] != base.TrackerCode.SUCCESS:
                    raise RuntimeError("Invalid return value %s" % str(value))
                url, port, matchkey = value[1]
                return connect(url, port, key + matchkey, session_timeout)
            except socket.error:
                self.close()
            except TVMError:
                pass


def connect(url, port, key="", session_timeout=0):
    """Connect to RPC Server

    Parameters
    ----------
    url : str
        The url of the host

    port : int
        The port to connect to

    key : str, optional
        Additional key to match server

    session_timeout : float, optional
        The duration of the session, allows server to kill
        the connection when duration is longer than this value.
        When duration is zero, it means the request must always be kept alive.

    Returns
    -------
    sess : RPCSession
        The connected session.
    """
    try:
        if session_timeout:
            key += " -timeout=%s" % str(session_timeout)
        sess = base._Connect(url, port, key)
    except NameError:
        raise RuntimeError("Please compile with USE_RPC=1")
    return RPCSession(sess)


def connect_tracker(url, port):
    """Connect to a RPC tracker

    Parameters
    ----------
    url : str
        The url of the host

    port : int
        The port to connect to

    Returns
    -------
    sess : TrackerSession
        The connected tracker session.
    """
    return TrackerSession((url, port))
