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

# pylint: disable=invalid-name

"""
Serial Low Level or Hw Driver
"""

import time as t

from serial import Serial
import serial.tools.list_ports
from serial.serialutil import SerialException

from .ai_runner import AiHwDriver
from .ai_runner import HwIOError


def serial_device_discovery():
    """Scan for available serial ports. return a list of available ports"""

    comports = serial.tools.list_ports.comports()
    dev_list = []
    for com in comports:
        elem = {
            "type": "serial",
            "device": com.device,
            "desc": com.description,
            "hwid": com.hwid,
        }
        dev_list.append(elem)

    return dev_list


def serial_get_com_settings(desc):
    """
    Parse the desc parameter to retreive the COM id and the baudrate.
    Can be a "str" or directly an "int" if only the baudrate is passed.

    Example:

        'COM7:115200'      ->  'COM7'  115200
        '460800'           ->   None   460800
        ':921600           ->   None   921600
        'COM6'             ->   COM6   115200
        ':COM6             ->   COM6   115200

    """

    # default values
    port_ = None
    baud_ = int(115200)

    if desc is not None and isinstance(desc, int):
        return port_, int(desc)

    if desc is None or not isinstance(desc, str):
        return port_, baud_

    desc = desc.split(":")
    for _d in desc:
        if _d:
            try:
                _d = int(_d)
            except (ValueError, TypeError):
                port_ = _d
            else:
                baud_ = _d

    return port_, baud_


class SerialHwDriver(AiHwDriver):
    """Serial low-level IO driver"""

    def __init__(self, parent=None):
        """Constructor"""
        self._device = None
        self._baudrate = 0
        super().__init__(parent)

    def get_config(self):
        """"Return a dict with used configuration"""
        return {
            "device": self._device,
            "baudrate": self._baudrate,
        }

    def _open(self, device, baudrate, timeout, dry_run=False):
        """Open the COM port"""
        _RETRY = 4
        hdl = None
        while _RETRY:
            try:
                hdl = Serial(device, baudrate=baudrate, timeout=timeout)
            except SerialException as _e:
                _RETRY -= 1
                if not _RETRY:
                    if not dry_run:
                        raise HwIOError("{}".format(_e))
                    # else:
                    return None
                t.sleep(0.2)
            else:
                break
        return hdl

    def _discovery(self, device, baudrate, timeout):
        """Discover the possible COM port"""
        if device is None:
            devices = serial_device_discovery()
        else:
            devices = [{"device": device}]
        for dev in devices:
            dry_run = dev != devices[-1]
            hdl_ = self._open(dev["device"], baudrate, timeout, dry_run=dry_run)
            if hdl_:
                self._hdl = hdl_
                self._hdl.reset_input_buffer()
                self._hdl.reset_output_buffer()
                self._device = dev["device"]
                self._baudrate = baudrate
                cpt = 1000
                while self._read(10) and cpt:
                    cpt = cpt - 1
                t.sleep(0.4)
                if hasattr(self._parent, "is_alive") and self._parent.is_alive():
                    return
                self._hdl.close()
                self._hdl = None
                if not dry_run:
                    raise HwIOError(
                        "{} - {}:{}".format("Invalid firmware", dev["device"], baudrate)
                    )
        # return None

    def _connect(self, desc=None, **kwargs):
        """Open a connection"""

        dev_, baud_ = serial_get_com_settings(desc)
        baud_ = kwargs.get("baudrate", baud_)
        timeout_ = kwargs.get("timeout", 0.001)

        self._discovery(dev_, baud_, timeout_)

        return self.is_connected

    def _disconnect(self):
        """Close the connection"""
        self._hdl.reset_input_buffer()
        self._hdl.reset_output_buffer()
        self._hdl.close()
        self._hdl = None

    def _read(self, size, timeout=0):
        """Read data from the connected device"""
        return self._hdl.read(size)

    def _write(self, data, timeout=0):
        """Write data to the connected device"""
        return self._hdl.write(data)

    def short_desc(self):
        """Report a human description of the connection state"""
        desc = "SERIAL:" + str(self._device) + ":" + str(self._baudrate)
        desc += ":connected" if self.is_connected else ":not connected"
        return desc


if __name__ == "__main__":
    pass
