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

"""Defines a Transport implementation using pyserial."""

import atexit
import time
import serial
import serial.tools.list_ports
from .base import IoTimeoutError, Transport, TransportTimeouts


_DEFAULT_SERIAL_TIMEOUTS = TransportTimeouts(
    session_start_retry_timeout_sec=5,
    session_start_timeout_sec=10.0,
    session_established_timeout_sec=30.0,
)


class SerialTransport(Transport):
    """A Transport implementation using pySerial."""

    _OPEN_PORTS = []

    @classmethod
    def close_atexit(cls):
        """Close all serial ports before exit.

        Some USB-UART kernel drivers are particularly sensitive to being left open (i.e. require
        unplugging and replugging of attached hardware or reboot of machine); try very hard to
        close all serial ports at exit.
        """
        for port in cls._OPEN_PORTS:
            try:
                port.close()
            except Exception:  # pylint: disable=broad-except
                _LOG.warn("exception closing port", exc_info=True)

        cls._OPEN_PORTS = []

    def __init__(self, grep=None, port_path=None, timeouts=None, **kw):
        self._port_path = port_path
        self._grep = grep
        self._timeouts = timeouts if timeouts is not None else _DEFAULT_SERIAL_TIMEOUTS
        self._kw = kw
        if self._port_path is None and self._grep is None:
            raise SerialPortNotFoundError("Must specify one of grep= or port_path=")

    def timeouts(self):
        return self._timeouts

    def open(self):
        if self._port_path is not None:
            port_path = self._port_path
        else:
            ports = list(serial.tools.list_ports.grep(self._grep))
            if len(ports) != 1:
                raise SerialPortNotFoundError(
                    f"grep expression should find 1 serial port; found {ports!r}"
                )

            port_path = ports[0].device

        self._port = serial.Serial(port_path, timeout=0.1, exclusive=True, **self._kw)
        self._port.cancel_read()
        self._port.reset_input_buffer()
        self._port.reset_output_buffer()
        self._OPEN_PORTS.append(self._port)

    def close(self):
        if self._port is None:
            return

        self._port.close()
        self._OPEN_PORTS.remove(self._port)
        self._port = None

    def read(self, n, timeout_sec):
        if timeout_sec is None:
            self._port.timeout = None
            in_waiting = self._port.in_waiting
            if in_waiting > 0:
                return self._port.read(min(n, in_waiting))
            return self._port.read(1)

        end_time = time.monotonic() + timeout_sec
        to_return = bytearray()
        while True:
            timeout_remaining = end_time - time.monotonic()
            if timeout_sec != 0 and timeout_remaining < 0:
                break

            # Read until *something* can be returned. If nothing is sent within 5 chars' time, stop.
            # 5 is an arbitrary number.
            self._port.timeout = 1 / self._port.baudrate * 5
            try:
                data = self._port.read(n if timeout_sec != 0 else 1)
                if not data and to_return:
                    break

                to_return.extend(data)
            except serial.SerialTimeoutException:
                if to_return:
                    break

        if not to_return:
            raise IoTimeoutError()

        return to_return

    def write(self, data, timeout_sec):
        self._port.write_timeout = timeout_sec
        try:
            to_return = self._port.write(data)
            self._port.flush()
            return to_return
        except serial.SerialTimeoutException:
            raise IoTimeoutError()


atexit.register(SerialTransport.close_atexit)
