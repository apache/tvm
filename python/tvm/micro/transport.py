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

"""Defines abstractions and implementations of the RPC transport used with micro TVM."""

import abc
import logging
import string
import typing

from .project_api.server import IoTimeoutError, TransportTimeouts
from .project_api.server import TransportClosedError


_ = TransportClosedError  # work around pylint unused-import error


_LOG = logging.getLogger(__name__)


def debug_transport_timeouts(session_start_retry_timeout_sec=0):
    return TransportTimeouts(
        session_start_retry_timeout_sec=session_start_retry_timeout_sec,
        session_start_timeout_sec=0,
        session_established_timeout_sec=0,
    )


class Transport(metaclass=abc.ABCMeta):
    """The abstract Transport class used for micro TVM."""

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    @abc.abstractmethod
    def timeouts(self):
        """Return TransportTimeouts suitable for use with this transport.

        See the TransportTimeouts documentation in python/tvm/micro/session.py.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def open(self):
        """Open any resources needed to send and receive RPC protocol data for a single session."""
        raise NotImplementedError()

    @abc.abstractmethod
    def close(self):
        """Release resources associated with this transport."""
        raise NotImplementedError()

    @abc.abstractmethod
    def read(self, n, timeout_sec):
        """Read up to n bytes from the transport.

        Parameters
        ----------
        n : int
            Maximum number of bytes to read from the transport.
        timeout_sec : Union[float, None]
            Number of seconds to wait for all `n` bytes to be received before timing out. The
            transport can wait additional time to account for transport latency or bandwidth
            limitations based on the selected configuration and number of bytes being received. If
            timeout_sec is 0, read should attempt to service the request in a non-blocking fashion.
            If timeout_sec is None, read should block until at least 1 byte of data can be returned.

        Returns
        -------
        bytes :
            Data read from the channel. Less than `n` bytes may be returned, but 0 bytes should
            never be returned. If returning less than `n` bytes, the full timeout_sec, plus any
            internally-added timeout, should be waited. If a timeout or transport error occurs,
            an exception should be raised rather than simply returning empty bytes.


        Raises
        ------
        TransportClosedError :
            When the transport layer determines that the transport can no longer send or receive
            data due to an underlying I/O problem (i.e. file descriptor closed, cable removed, etc).

        IoTimeoutError :
            When `timeout_sec` elapses without receiving any data.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def write(self, data, timeout_sec):
        """Write data to the transport channel.

        Parameters
        ----------
        data : bytes
            The data to write over the channel.
        timeout_sec : Union[float, None]
            Number of seconds to wait for at least one byte to be written before timing out. The
            transport can wait additional time to account for transport latency or bandwidth
            limitations based on the selected configuration and number of bytes being received. If
            timeout_sec is 0, write should attempt to service the request in a non-blocking fashion.
            If timeout_sec is None, write should block until at least 1 byte of data can be
            returned.

        Returns
        -------
        int :
            The number of bytes written to the underlying channel. This can be less than the length
            of `data`, but cannot be 0 (raise an exception instead).

        Raises
        ------
        TransportClosedError :
            When the transport layer determines that the transport can no longer send or receive
            data due to an underlying I/O problem (i.e. file descriptor closed, cable removed, etc).

        IoTimeoutError :
            When `timeout_sec` elapses without receiving any data.
        """
        raise NotImplementedError()


class TransportLogger(Transport):
    """Wraps a Transport implementation and logs traffic to the Python logging infrastructure."""

    def __init__(self, name, child, logger=None, level=logging.INFO):
        self.name = name
        self.child = child
        self.logger = logger or _LOG
        self.level = level

    # Construct PRINTABLE to exclude whitespace from string.printable.
    PRINTABLE = string.digits + string.ascii_letters + string.punctuation

    @classmethod
    def _to_hex(cls, data):
        lines = []
        if not data:
            lines.append("")
            return lines

        for i in range(0, (len(data) + 15) // 16):
            chunk = data[i * 16 : (i + 1) * 16]
            hex_chunk = " ".join(f"{c:02x}" for c in chunk)
            ascii_chunk = "".join((chr(c) if chr(c) in cls.PRINTABLE else ".") for c in chunk)
            lines.append(f"{i * 16:04x}  {hex_chunk:47}  {ascii_chunk}")

        if len(lines) == 1:
            lines[0] = lines[0][6:]

        return lines

    def timeouts(self):
        return self.child.timeouts()

    def open(self):
        self.logger.log(self.level, "%s: opening transport", self.name)
        self.child.open()

    def close(self):
        self.logger.log(self.level, "%s: closing transport", self.name)
        return self.child.close()

    def read(self, n, timeout_sec):
        timeout_str = f"{timeout_sec:5.2f}s" if timeout_sec is not None else " None "
        try:
            data = self.child.read(n, timeout_sec)
        except IoTimeoutError:
            self.logger.log(
                self.level,
                "%s: read {%s} %4d B -> [IoTimeoutError %s]",
                self.name,
                timeout_str,
                n,
                timeout_str,
            )
            raise
        except Exception as err:
            self.logger.log(
                self.level,
                "%s: read {%s} %4d B -> [err: %s]",
                self.name,
                timeout_str,
                n,
                err.__class__.__name__,
                exc_info=1,
            )
            raise err

        hex_lines = self._to_hex(data)
        if len(hex_lines) > 1:
            self.logger.log(
                self.level,
                "%s: read {%s} %4d B -> [%3d B]:\n%s",
                self.name,
                timeout_str,
                n,
                len(data),
                "\n".join(hex_lines),
            )
        else:
            self.logger.log(
                self.level,
                "%s: read {%s} %4d B -> [%3d B]: %s",
                self.name,
                timeout_str,
                n,
                len(data),
                hex_lines[0],
            )

        return data

    def write(self, data, timeout_sec):
        timeout_str = f"{timeout_sec:5.2f}s" if timeout_sec is not None else " None "
        try:
            self.child.write(data, timeout_sec)
        except IoTimeoutError:
            self.logger.log(
                self.level,
                "%s: write {%s}       <- [%3d B]: [IoTimeoutError %s]",
                self.name,
                timeout_str,
                len(data),
                timeout_str,
            )
            raise
        except Exception as err:
            self.logger.log(
                self.level,
                "%s: write {%s}       <- [%3d B]: [err: %s]",
                self.name,
                timeout_str,
                len(data),
                err.__class__.__name__,
                exc_info=1,
            )
            raise err

        hex_lines = self._to_hex(data)
        if len(hex_lines) > 1:
            self.logger.log(
                self.level,
                "%s: write {%s}        <- [%3d B]:\n%s",
                self.name,
                timeout_str,
                len(data),
                "\n".join(hex_lines),
            )
        else:
            self.logger.log(
                self.level,
                "%s: write {%s}        <- [%3d B]: %s",
                self.name,
                timeout_str,
                len(data),
                hex_lines[0],
            )


TransportContextManager = typing.ContextManager[Transport]
