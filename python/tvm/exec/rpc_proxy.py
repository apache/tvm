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
"""RPC web proxy, allows redirect to websocket based RPC servers(browsers)"""
import logging
import argparse
import os
import glob
from tvm.rpc.proxy import Proxy


def find_example_resource():
    """Find resource examples."""
    curr_path = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    base_path = os.path.abspath(os.path.join(curr_path, "..", "..", ".."))
    index_page = os.path.join(base_path, "web", "apps", "browser", "rpc_server.html")
    default_plugin_page = os.path.join(base_path, "web", "apps", "browser", "rpc_plugin.html")

    resource_files = [
        ("/", os.path.join(base_path, "web", "dist", "tvmjs.bundle.js")),
        ("/", os.path.join(base_path, "web", "dist", "wasm", "tvmjs_runtime.wasi.js")),
        ("/", index_page),
    ]
    allow_format = ("json", "bin", "js", "wasm", "html", "css", "model")

    # recursively apend things in www, up to two levels
    resource_bases = [
        os.path.join(base_path, "web", "dist", "www"),
        os.path.join(base_path, "web", ".ndarray_cache"),
    ]
    for base in resource_bases:
        if not os.path.isdir(base):
            continue
        for full_name in glob.glob("%s/**" % base, recursive=True):
            fname = os.path.relpath(full_name, base)
            dirname = os.path.dirname(fname)
            fmt = fname.rsplit(".", 1)[-1]
            if os.path.isfile(full_name) and fmt in allow_format:
                resource_files.append((dirname, full_name))

    for item in resource_files:
        fname = item[-1]
        if not os.path.exists(fname):
            raise RuntimeError("Cannot find %s" % fname)

    if not any(item[-1].endswith("rpc_plugin.html") for item in resource_files):
        resource_files.append(("/", default_plugin_page))

    return index_page, resource_files


def main(args):
    """Main function"""
    if args.tracker:
        url, port = args.tracker.split(":")
        port = int(port)
        tracker_addr = (url, port)
    else:
        tracker_addr = None

    if args.example_rpc:
        index, js_files = find_example_resource()
        prox = Proxy(
            args.host,
            port=args.port,
            port_end=args.port_end,
            web_port=args.web_port,
            index_page=index,
            resource_files=js_files,
            tracker_addr=tracker_addr,
        )
    else:
        prox = Proxy(
            args.host,
            port=args.port,
            port_end=args.port_end,
            web_port=args.web_port,
            tracker_addr=tracker_addr,
        )
    prox.proc.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="the hostname of the server")
    parser.add_argument("--port", type=int, default=9090, help="The port of the RPC")
    parser.add_argument("--port-end", type=int, default=9199, help="The end search port of the RPC")
    parser.add_argument(
        "--web-port", type=int, default=8888, help="The port of the http/websocket server"
    )
    parser.add_argument(
        "--example-rpc", type=bool, default=False, help="Whether to switch on example rpc mode"
    )
    parser.add_argument("--tracker", type=str, default="", help="Report to RPC tracker")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
