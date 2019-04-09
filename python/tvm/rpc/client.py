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
"""RPC client tools"""
from __future__ import absolute_import

import os
import socket
import struct
import time

from . import base
from ..contrib import util
from .._ffi.base import TVMError
from .._ffi import function
from .._ffi import ndarray as nd
from ..module import load as _load_module


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
        ctx = nd.context(dev_type, dev_id)
        encode = (self._tbl_index + 1) * base.RPC_SESS_MASK
        ctx.device_type += encode
        ctx._rpc_sess = self
        return ctx

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
                "tvm.rpc.server.upload")
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
                "tvm.rpc.server.download")
        return self._remote_funcs["download"](path)

    def remove(self, path):
        """Remove file from remote temp folder.

        Parameters
        ----------
        path: str
            The relative location to remote temp folder.
        """
        if "remove" not in self._remote_funcs:
            self._remote_funcs["remove"] = self.get_function(
                "tvm.rpc.server.remove")
        self._remote_funcs["remove"](path)

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

    def cpu(self, dev_id=0):
        """Construct CPU device."""
        return self.context(1, dev_id)

    def gpu(self, dev_id=0):
        """Construct GPU device."""
        return self.context(2, dev_id)

    def cl(self, dev_id=0):
        """Construct OpenCL device."""
        return self.context(4, dev_id)

    def vulkan(self, dev_id=0):
        """Construct Vulkan device."""
        return self.context(7, dev_id)

    def metal(self, dev_id=0):
        """Construct Metal device."""
        return self.context(8, dev_id)

    def opengl(self, dev_id=0):
        """Construct OpenGL device."""
        return self.context(11, dev_id)

    def ext_dev(self, dev_id=0):
        """Construct extension device."""
        return self.context(12, dev_id)


class LocalSession(RPCSession):
    """RPCSession interface backed by local environment.

    This class can be used to implement functions that
    need to be ran both locally and remotely.
    """
    def __init__(self):
        # pylint: disable=super-init-not-called
        self.context = nd.context
        self.get_function = function.get_global_func
        self._temp = util.tempdir()

    def upload(self, data, target=None):
        if isinstance(data, bytearray):
            if not target:
                raise ValueError("target must present when file is a bytearray")
            blob = data
        else:
            blob = bytearray(open(data, "rb").read())
            if not target:
                target = os.path.basename(data)
        with open(self._temp.relpath(target), "wb") as f:
            f.write(blob)

    def download(self, path):
        return bytearray(open(self._temp.relpath(path), "rb").read())

    def load_module(self, path):
        return _load_module(self._temp.relpath(path))


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
        self._connect()

    def __del__(self):
        self.close()

    def _connect(self):
        self._sock = base.connect_with_retry(self._addr)
        self._sock.sendall(struct.pack("<i", base.RPC_TRACKER_MAGIC))
        magic = struct.unpack("<i", base.recvall(self._sock, 4))[0]
        if magic != base.RPC_TRACKER_MAGIC:
            raise RuntimeError("%s is not RPC Tracker" % str(self._addr))

    def close(self):
        """Close the tracker connection."""
        if self._sock:
            self._sock.close()
            self._sock = None

    def summary(self):
        """Get the summary dict of the tracker."""
        base.sendjson(self._sock, [base.TrackerCode.SUMMARY])
        value = base.recvjson(self._sock)
        if value[0] != base.TrackerCode.SUCCESS:
            raise RuntimeError("Invalid return value %s" % str(value))
        return value[1]

    def text_summary(self):
        """Get a text summary of the tracker."""
        data = self.summary()

        total_ct = {}

        res = ""
        res += "Server List\n"
        res += "----------------------------\n"
        res += "server-address\tkey\n"
        res += "----------------------------\n"
        for item in data["server_info"]:
            addr = item["addr"]
            res += addr[0] + ":" + str(addr[1]) + "\t"
            res += item["key"] + "\n"
            key = item['key'].split(':')[1]   # 'server:rasp3b` -> 'rasp3b'
            if key not in total_ct:
                total_ct[key] = 0
            total_ct[key] += 1
        res += "----------------------------\n"
        res += "\n"

        # compute max length of device key
        queue_info = data['queue_info']
        keys = list(queue_info.keys())
        if keys:
            keys.sort()
            max_key_len = max([len(k) for k in keys])
        else:
            max_key_len = 0

        res += "Queue Status\n"
        title = ("%%-%ds" % max_key_len + "   total  free  pending\n") % 'key'
        separate_line = '-' * len(title) + '\n'
        res += separate_line + title + separate_line
        for k in keys:
            total = total_ct.get(k, 0)
            free, pending = queue_info[k]["free"], queue_info[k]["pending"]
            if total or pending:
                res += ("%%-%ds" % max_key_len + "   %-5d  %-4d  %-7d\n") % \
                       (k, total, free, pending)
        res += separate_line
        return res

    def request(self, key, priority=1, session_timeout=0, max_retry=5):
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

        max_retry : int, optional
            Maximum number of times to retry before give up.
        """
        last_err = None
        for _ in range(max_retry):
            try:
                if self._sock is None:
                    self._connect()
                base.sendjson(self._sock,
                              [base.TrackerCode.REQUEST, key, "", priority])
                value = base.recvjson(self._sock)
                if value[0] != base.TrackerCode.SUCCESS:
                    raise RuntimeError("Invalid return value %s" % str(value))
                url, port, matchkey = value[1]
                return connect(url, port, matchkey, session_timeout)
            except socket.error as err:
                self.close()
                last_err = err
            except TVMError as err:
                last_err = err
        raise RuntimeError(
            "Cannot request %s after %d retry, last_error:%s" % (
                key, max_retry, str(last_err)))

    def request_and_run(self,
                        key,
                        func,
                        priority=1,
                        session_timeout=0,
                        max_retry=2):
        """Request a resource from tracker and run the func.

        This function safe-guard rare server node dropout during execution.
        In such case, a new resource will be requested and func will be ran again.

        Parameters
        ----------
        key : str
            The type key of the device.

        func : function of session -> value
            A stateless function

        priority : int, optional
            The priority of the request.

        session_timeout : float, optional
            The duration of the session, allows server to kill
            the connection when duration is longer than this value.
            When duration is zero, it means the request must always be kept alive.

        max_retry : int, optional
            Maximum number of times to retry the function before give up.
        """
        last_err = None
        for _ in range(max_retry):
            try:
                sess = self.request(key,
                                    priority=priority,
                                    session_timeout=session_timeout)
                tstart = time.time()
                return func(sess)
            except TVMError as err:
                duration = time.time() - tstart
                # roughly estimate if the error is due to timeout termination
                if session_timeout and duration >= session_timeout * 0.95:
                    raise RuntimeError(
                        "Session timeout when running %s" % func.__name__)
                last_err = err
        raise RuntimeError(
            "Failed to run on %s after %d retry, last_error:%s" % (
                key, max_retry, str(last_err)))


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
