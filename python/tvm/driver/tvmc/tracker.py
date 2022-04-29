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
# specific language
"""
TVMC Remote Tracker
"""

import logging
from urllib.parse import urlparse

# pylint: disable=invalid-name
logger = logging.getLogger("TVMC")


def tracker_host_port_from_cli(rpc_tracker_str):
    """Extract hostname and (optional) port from strings
    like "1.2.3.4:9090" or "4.3.2.1".

    Used as a helper function to cover --rpc-tracker
    command line argument, in different subcommands.

    Parameters
    ----------
    rpc_tracker_str : str
        hostname (or IP address) and port of the RPC tracker,
        in the format 'hostname[:port]'.

    Returns
    -------
    rpc_hostname : str or None
        hostname or IP address, extracted from input.
    rpc_port : int or None
        port number extracted from input (9090 default).
    """

    rpc_hostname = rpc_port = None

    if rpc_tracker_str:
        parsed_url = urlparse("//%s" % rpc_tracker_str)
        rpc_hostname = parsed_url.hostname
        rpc_port = parsed_url.port or 9090
        logger.info("RPC tracker hostname: %s", rpc_hostname)
        logger.info("RPC tracker port: %s", rpc_port)

    return rpc_hostname, rpc_port
