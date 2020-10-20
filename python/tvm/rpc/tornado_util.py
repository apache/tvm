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
"""Utilities used in tornado."""

import socket
import errno
from tornado import ioloop


class TCPHandler(object):
    """TCP socket handler backed tornado event loop.

    Parameters
    ----------
    sock : Socket
        The TCP socket, will set it to non-blocking mode.
    """

    def __init__(self, sock):
        self._sock = sock
        self._ioloop = ioloop.IOLoop.current()
        self._sock.setblocking(0)
        self._pending_write = []
        self._signal_close = False

        def _event_handler(_, events):
            self._event_handler(events)

        self._ioloop.add_handler(
            self._sock.fileno(), _event_handler, self._ioloop.READ | self._ioloop.ERROR
        )

    def signal_close(self):
        """Signal the handler to close.

        The handler will be closed after the existing
        pending message are sent to the peer.
        """
        if not self._pending_write:
            self.close()
        else:
            self._signal_close = True

    def close(self):
        """Close the socket"""
        if self._sock is not None:
            try:
                self._ioloop.remove_handler(self._sock.fileno())
                self._sock.close()
            except socket.error:
                pass
            self._sock = None
            self.on_close()

    def write_message(self, message, binary=True):
        assert binary
        if self._sock is None:
            raise IOError("socket is already closed")
        self._pending_write.append(message)
        self._update_write()

    def _event_handler(self, events):
        """centeral event handler"""
        if (events & self._ioloop.ERROR) or (events & self._ioloop.READ):
            if self._update_read() and (events & self._ioloop.WRITE):
                self._update_write()
        elif events & self._ioloop.WRITE:
            self._update_write()

    def _update_write(self):
        """Update the state on write"""
        while self._pending_write:
            try:
                msg = self._pending_write[0]
                if self._sock is None:
                    return
                nsend = self._sock.send(msg)
                if nsend != len(msg):
                    self._pending_write[0] = msg[nsend:]
                else:
                    self._pending_write.pop(0)
            except socket.error as err:
                if err.args[0] in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                self.on_error(err)

        if self._pending_write:
            self._ioloop.update_handler(
                self._sock.fileno(), self._ioloop.READ | self._ioloop.ERROR | self._ioloop.WRITE
            )
        else:
            if self._signal_close:
                self.close()
            else:
                self._ioloop.update_handler(
                    self._sock.fileno(), self._ioloop.READ | self._ioloop.ERROR
                )

    def _update_read(self):
        """Update state when there is read event"""
        try:
            msg = bytes(self._sock.recv(4096))
            if msg:
                self.on_message(msg)
                return True
            # normal close, remote is closed
            self.close()
        except socket.error as err:
            if err.args[0] in (errno.EAGAIN, errno.EWOULDBLOCK):
                pass
            else:
                self.on_error(err)
        return False
