import numpy as np

import tvm
from tvm import rpc
from tvm.autotvm.measure import request_remote
from tvm.auto_scheduler.utils import call_func_with_timeout
from tvm.contrib.popen_pool import PopenWorker, StatusKind
from tvm.rpc import tracker, proxy, server_ios_launcher


HOST_URL = "0.0.0.0"
HOST_PORT = 9190
DEVICE_KEY = "ios_mobile_device"
# TODO: When starting an application in pure_rpc mode, the address and port fields are ignored.
# The server starts on one of the following ports: 9090-9099.
# The port on which the server is running is printed in the application log window.
# This value is not broadcast anywhere else, it cannot be programmatically obtained from outside
# the application, so there may be connection problems if port 9090 is busy and the server starts
# on a different port.
HOST_PORT_PURE_RPC = 9090
TEST_FUNCTION_NAME = "rpc.test_ios_rpc.add_one"


@tvm.register_func(TEST_FUNCTION_NAME)
def add_one(x):
    return x + 1


def can_create_connection_without_deadlock(timeout, func, args=(), kwargs=None):
    def wrapper(*_args, **_kwargs):
        """
            This wrapper is needed because the cloudpicle
            cannot serialize objects that contain pointers (RPCSession)
        """
        func(*_args, **_kwargs)
        return StatusKind.COMPLETE

    worker = PopenWorker()
    ret = call_func_with_timeout(worker, timeout=timeout, func=wrapper, args=args, kwargs=kwargs)
    if isinstance(ret, Exception):
        raise ret
    return ret


def infra_pure_rpc(f):
    """
    Host  --  RPC server
    """
    def wrapper():
        device_server_launcher = server_ios_launcher.ServerIOSLauncher(mode=server_ios_launcher.RPCServerMode.pure_server.value,
                                                                       host=HOST_URL,
                                                                       port=HOST_PORT_PURE_RPC,
                                                                       key=DEVICE_KEY)
        f(host=device_server_launcher.host, port=device_server_launcher.port)
        device_server_launcher.terminate()
    return wrapper


def infra_rpc_proxy(f):
    """
    Host -- Proxy -- RPC server
    """
    def wrapper():
        proxy_server = proxy.Proxy(host=HOST_URL, port=HOST_PORT)
        device_server_launcher = server_ios_launcher.ServerIOSLauncher(mode=server_ios_launcher.RPCServerMode.proxy.value,
                                                                       host=proxy_server.host,
                                                                       port=proxy_server.port,
                                                                       key=DEVICE_KEY)
        f(host=proxy_server.host, port=proxy_server.port)
        device_server_launcher.terminate()
        proxy_server.terminate()
    return wrapper


def infra_rpc_tracker(f):
    """
         tracker
         /     \
    Host   --   RPC server
    """
    def wrapper():
        tracker_server = tracker.Tracker(host=HOST_URL, port=HOST_PORT, silent=True)
        device_server_launcher = server_ios_launcher.ServerIOSLauncher(mode=server_ios_launcher.RPCServerMode.tracker.value,
                                                                       host=tracker_server.host,
                                                                       port=tracker_server.port,
                                                                       key=DEVICE_KEY)
        f(host=tracker_server.host, port=tracker_server.port)
        device_server_launcher.terminate()
        tracker_server.terminate()
    return wrapper


def infra_rpc_tracker_via_proxy(f):
    """
         tracker
         /     \
    Host   --   Proxy -- RPC server
    """
    def wrapper():
        tracker_server = tracker.Tracker(host=HOST_URL, port=HOST_PORT, silent=True)
        proxy_server_tracker = proxy.Proxy(host=HOST_URL, port=8888, tracker_addr=(tracker_server.host, tracker_server.port))
        device_server_launcher = server_ios_launcher.ServerIOSLauncher(mode=server_ios_launcher.RPCServerMode.proxy.value,
                                                                       host=proxy_server_tracker.host,
                                                                       port=proxy_server_tracker.port,
                                                                       key=DEVICE_KEY)
        f(host=tracker_server.host, port=tracker_server.port)
        device_server_launcher.terminate()
        proxy_server_tracker.terminate()
        tracker_server.terminate()
    return wrapper


@infra_pure_rpc
def test_pure_rpc(host, port):
    try:
        results = []
        for _ in range(2):
            ret = can_create_connection_without_deadlock(timeout=10, func=rpc.connect,
                                                         args=(host, port, DEVICE_KEY))
            results.append(ret)
        if not np.all(np.array(results) == StatusKind.COMPLETE):
            raise ValueError("One or more sessions ended incorrectly.")
    except Exception as e:
        print(e)


@infra_rpc_proxy
def test_rpc_proxy(host, port):
    try:
        results = []
        for _ in range(2):
            ret = can_create_connection_without_deadlock(timeout=10, func=rpc.connect,
                                                         args=(host, port, DEVICE_KEY))
            results.append(ret)
        if not np.all(np.array(results) == StatusKind.COMPLETE):
            raise ValueError("One or more sessions ended incorrectly.")
    except Exception as e:
        print(e)


@infra_rpc_tracker
def test_rpc_tracker(host, port):
    try:
        results = []
        for _ in range(2):
            ret = can_create_connection_without_deadlock(timeout=10, func=request_remote,
                                                         args=(DEVICE_KEY, host, port))
            results.append(ret)
        if not np.all(np.array(results) == StatusKind.COMPLETE):
            raise ValueError("One or more sessions ended incorrectly.")
    except Exception as e:
        print(e)


@infra_rpc_tracker_via_proxy
def test_rpc_tracker_via_proxy(host, port):
    try:
        results = []
        for _ in range(2):
            ret = can_create_connection_without_deadlock(timeout=10, func=request_remote,
                                                         args=(DEVICE_KEY, host, port))
            results.append(ret)
        if not np.all(np.array(results) == StatusKind.COMPLETE):
            raise ValueError("One or more sessions ended incorrectly.")
    except Exception as e:
        print(e)


@infra_rpc_tracker
def test_can_call_remote_function_with_rpc_tracker(host, port):
    remote_session = request_remote(DEVICE_KEY, host, port)
    f = remote_session.get_function(TEST_FUNCTION_NAME)
    assert f(10) == 11


if __name__ == '__main__':
    test_pure_rpc()
    test_rpc_proxy()
    test_rpc_tracker()
    test_rpc_tracker_via_proxy()

    test_can_call_remote_function_with_rpc_tracker()

    server_ios_launcher.ServerIOSLauncher.shutdown_booted_devices()
