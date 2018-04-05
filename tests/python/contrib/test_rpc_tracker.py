import tvm
import logging
import numpy as np
import time
import multiprocessing
from tvm.contrib import rpc

def check_server_drop():
    """test when server drops"""
    try:
        from tvm.contrib.rpc import tracker, base
        from tvm.contrib.rpc.base import TrackerCode

        @tvm.register_func("rpc.test2.addone")
        def addone(x):
            return x + 1

        def _put(tclient, value):
            base.sendjson(tclient._sock, value)
            base.recvjson(tclient._sock)

        tserver = tracker.Tracker("localhost", 8888)
        tclient = rpc.connect_tracker("localhost", tserver.port)

        server1 = rpc.Server(
            "localhost", port=9099,
            tracker_addr=("localhost", tserver.port),
            key="xyz")
        server2 = rpc.Server(
            "localhost", port=9099,
            tracker_addr=("localhost", tserver.port),
            key="xyz")

        # Fault tolerence to stale worker value
        _put(tclient, [TrackerCode.PUT, "xyz", (server1.port, "abc")])
        _put(tclient, [TrackerCode.PUT, "xyz", (server1.port, "abcxxx")])
        _put(tclient, [TrackerCode.PUT, "xyz", (server2.port, "abcxxx11")])

        # Fault tolerence server timeout
        def check_timeout(timeout, sleeptime):
            try:
                remote = tclient.request("xyz", priority=0, session_timeout=timeout)
                remote2 = tclient.request("xyz", session_timeout=timeout)
                time.sleep(sleeptime)
                f1 = remote.get_function("rpc.test2.addone")
                assert f1(10) == 11
                f1 = remote2.get_function("rpc.test2.addone")
                assert f1(10) == 11
            except tvm.TVMError as e:
                pass
        check_timeout(0.01, 0.1)
        check_timeout(2, 0)
    except ImportError:
        print("Skip because tornado is not available")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_server_drop()
