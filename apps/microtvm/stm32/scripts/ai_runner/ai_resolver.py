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

"""
Entry point to register a driver

Expected syntax for driver description/options for the connection

    [<domain>[:<option1>[:option2]]]

    domain:
        - None or 'serial' (default)
        - 'dll' or valid file_path/directory
        - 'socket'

"""

import os

_DEFAULT_DOMAIN = "serial"
_FILE_DOMAIN = "file"


def _default_resolver(domain, desc=None):  # pylint: disable=unused-argument
    """Default resolver function"""
    # if domain == _DEFAULT_DOMAIN or domain == None:
    if domain in (_DEFAULT_DOMAIN, None):
        return True
    return False


def _default_create(parent, desc):
    """Default create function"""
    from .serial_hw_drv import SerialHwDriver  # pylint: disable=import-outside-toplevel
    from .pb_mgr_drv import AiPbMsg  # pylint: disable=import-outside-toplevel

    return AiPbMsg(parent, SerialHwDriver()), desc


_DRIVERS = {
    _DEFAULT_DOMAIN: (_default_resolver, _default_create),  # default
}


def ai_runner_register(name, resolver, create):
    """Register a resolver"""
    _DRIVERS[name] = (resolver, create)


def ai_runner_resolver(parent, desc):
    """Return drv instance"""

    if desc is not None and isinstance(desc, str):
        desc = desc.strip()
        split_ = desc.split(":")
        nb_elem = len(split_)

        domain = None
        # only valid file or directory is passed
        if os.path.isfile(desc) or os.path.isdir(desc) or os.path.isdir(os.path.dirname(desc)):
            domain = _FILE_DOMAIN
        # domain is considered
        elif nb_elem >= 1 and len(split_[0]) > 0:
            domain = split_[0].lower()
            desc = desc[len(domain + ":") :]
        # ':' is passed as first character
        elif len(split_[0]) == 0:
            domain = _DEFAULT_DOMAIN
            desc = desc[1:]

        for drv_ in _DRIVERS:
            if _DRIVERS[drv_][0](domain, desc):
                return _DRIVERS[drv_][1](parent, desc)

        err_msg_ = f'invalid/unsupported "{domain}:{desc}" descriptor'
        return None, err_msg_

    # else:
    return _DRIVERS[_DEFAULT_DOMAIN][1](parent, desc)
