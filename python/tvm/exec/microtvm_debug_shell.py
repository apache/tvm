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
# pylint: disable=redefined-outer-name, invalid-name
"""Start an RPC server intended for use as a microTVM debugger.

microTVM aims to be runtime-agnostic, and to that end, frameworks often define command-line tools
used to launch a debug flow. These tools often manage the process of connecting to an attached
device using a hardware debugger, exposing a GDB server, and launching GDB connected to that
server with a source file attached. It's also true that this debugger can typically not be executed
concurrently with any flash tool, so this integration point is provided to allow TVM to launch and
terminate any debuggers integrated with the larger microTVM compilation/autotuning flow.

To use this tool, first launch this script in a separate terminal window. Then, provide the hostport
to your compiler's Flasher instance.
"""

import argparse
import logging
import socket
import struct
import sys

import tvm.micro.debugger as _  # NOTE: imported to expose global PackedFuncs over RPC.

from .._ffi.base import py_str
from ..rpc import base
from ..rpc import _ffi_api


_LOG = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments to this script."""
    parser = argparse.ArgumentParser(description="microTVM debug-tool runner")
    parser.add_argument("--host", default="0.0.0.0", help="hostname to listen on")
    parser.add_argument("--port", type=int, default=9090, help="hostname to listen on")
    parser.add_argument(
        "--impl",
        help=(
            "If given, name of a module underneath tvm.micro.contrib "
            "which contains the Debugger implementation to use. For example, to enable a "
            "debugger named BarDebugger in python/tvm/micro/contrib/foo.py, specify either "
            "'tvm.micro.contrib.foo' or 'foo' here. To enable a debugger named BazDebugger in "
            "a third-party module ext_package.debugger, specify 'ext_package.debugger' here. "
            "NOTE: the module cannot be in a sub-package of tvm.micro.contrib."
        ),
    )

    return parser.parse_args()


class ConnectionClosedError(Exception):
    """Raised when the connection is closed."""


def handle_conn(conn, rpc_key):
    """Handle a single connection that has just been accept'd()."""

    def send(data):
        conn.sendall(data)
        return len(data)

    magic = struct.unpack("<i", base.recvall(conn, 4))[0]
    if magic != base.RPC_MAGIC:
        conn.close()
        return

    keylen = struct.unpack("<i", base.recvall(conn, 4))[0]
    key = py_str(base.recvall(conn, keylen))
    arr = key.split()
    expect_header = "client:"
    server_key = "server:" + rpc_key
    if arr[0] != expect_header:
        conn.sendall(struct.pack("<i", base.RPC_CODE_MISMATCH))
        _LOG.warning("mismatch key from %s", addr)
        return

    conn.sendall(struct.pack("<i", base.RPC_CODE_SUCCESS))
    conn.sendall(struct.pack("<i", len(server_key)))
    conn.sendall(server_key.encode("utf-8"))
    server = _ffi_api.CreateEventDrivenServer(send, "microtvm-rpc-debugger", key)

    def _readall(n):
        buf = bytearray()
        while len(buf) < n:
            x = conn.recv(n - len(buf))
            if not x:
                raise ConnectionClosedError()

            buf = buf + x

        return buf

    while True:
        packet_length_bytes = _readall(8)
        packet_length = struct.unpack("<q", packet_length_bytes)[0]
        if not packet_length:
            break

        status = server(packet_length_bytes, 3)
        if status == 0:
            break

        packet_body = _readall(packet_length)
        status = server(packet_body, 3)


def main():
    """Main entry point for microTVM debug shell."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.impl:
        package = None
        if "." not in args.impl:
            package = f"tvm.micro.contrib.{args.impl}"
        importlib.import_module(args.impl, package)

    sock = socket.socket(base.get_addr_family([args.host, args.port]), socket.SOCK_STREAM)
    # Never set socket SO_REUSEADDR on Windows. The SO_REUSEADDR flag allow reusing the
    # inactivate TIME_WATI state sockets on POSIX, but on Windows it will allow two or more
    # activate sockets to bind on the same address and port if they all set SO_REUSEADDR,
    # and result in indeterminate behavior.
    if reuse_addr and sys.platform != "win32":
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.listen(1)
    bind_addr, bind_port = sock.getsockname()
    _LOG.info("listening for connections on %s:%d", bind_addr, bind_port)

    while True:
        conn, peer = sock.accept()
        _LOG.info("accepted connection from %s", peer)
        try:
            handle_conn(conn, "")
        except ConnectionClosedError:
            pass
        finally:
            conn.close()
            _LOG.info("closed connection from %s", peer)


if __name__ == "__main__":
    main()
