import numpy as np

from tvm import rpc
from tvm.autotvm.measure import request_remote
from tvm.auto_scheduler.utils import call_func_with_timeout
from tvm.contrib.popen_pool import PopenWorker, StatusKind
from tvm.rpc import tracker, proxy, server_ios_launcher


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


def test_rpc_proxy():
    """
    Host -- Proxy -- RPC server
    """
    proxy_server = proxy.Proxy(host=host_url, port=host_port)
    device_server_launcher = server_ios_launcher.ServerIOSLauncher(mode=server_ios_launcher.RPCServerMode.proxy.value,
                                                                   host=proxy_server.host,
                                                                   port=proxy_server.port,
                                                                   key=key)
    try:
        results = []
        for _ in range(2):
            ret = can_create_connection_without_deadlock(timeout=10, func=rpc.connect,
                                                         args=(proxy_server.host, proxy_server.port, key))
            results.append(ret)
        if not np.all(np.array(results) == StatusKind.COMPLETE):
            raise ValueError("One or more sessions ended incorrectly.")
    except Exception as e:
        print(e)

    device_server_launcher.terminate()
    proxy_server.terminate()


def test_rpc_tracker():
    """
         tracker
         /     \
    Host   --   RPC server
    """
    tracker_server = tracker.Tracker(host=host_url, port=host_port, silent=True)
    device_server_launcher = server_ios_launcher.ServerIOSLauncher(mode=server_ios_launcher.RPCServerMode.tracker.value,
                                                                   host=tracker_server.host,
                                                                   port=tracker_server.port,
                                                                   key=key)
    try:
        results = []
        for _ in range(2):
            ret = can_create_connection_without_deadlock(timeout=10, func=request_remote,
                                                         args=(key, tracker_server.host, tracker_server.port))
            results.append(ret)
        if not np.all(np.array(results) == StatusKind.COMPLETE):
            raise ValueError("One or more sessions ended incorrectly.")
    except Exception as e:
        print(e)

    device_server_launcher.terminate()
    tracker_server.terminate()


def test_rpc_tracker_via_proxy():
    """
         tracker
         /     \
    Host   --   Proxy -- RPC server
    """
    tracker_server = tracker.Tracker(host=host_url, port=host_port, silent=True)
    proxy_server_tracker = proxy.Proxy(host=host_url, port=8888, tracker_addr=(tracker_server.host, tracker_server.port))
    device_server_launcher = server_ios_launcher.ServerIOSLauncher(mode=server_ios_launcher.RPCServerMode.proxy.value,
                                                                   host=proxy_server_tracker.host,
                                                                   port=proxy_server_tracker.port,
                                                                   key=key)

    try:
        results = []
        for _ in range(2):
            ret = can_create_connection_without_deadlock(timeout=10, func=request_remote,
                                                         args=(key, tracker_server.host, tracker_server.port))
            results.append(ret)
        if not np.all(np.array(results) == StatusKind.COMPLETE):
            raise ValueError("One or more sessions ended incorrectly.")
    except Exception as e:
        print(e)

    device_server_launcher.terminate()
    proxy_server_tracker.terminate()
    tracker_server.terminate()


if __name__ == '__main__':
    host_url = "0.0.0.0"
    host_port = 9190
    key = "ios_mobile_device"

    test_rpc_proxy()
    test_rpc_tracker()
    test_rpc_tracker_via_proxy()

    server_ios_launcher.ServerIOSLauncher.shutdown_booted_devices()
