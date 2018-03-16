"""RPC proxy, allows both client/server to connect and match connection.

In normal RPC, client directly connect to server's IP address.
Sometimes this cannot be done when server do not have a static address.
RPCProxy allows both client and server connect to the proxy server,
the proxy server will forward the message between the client and server.
"""
# pylint: disable=unused-variable, unused-argument
from __future__ import absolute_import

import os
import logging
import socket
import multiprocessing
import errno
import struct
try:
    import tornado
    from tornado import gen
    from tornado import websocket
    from tornado import ioloop
    from tornado import websocket
except ImportError as error_msg:
    raise ImportError("RPCProxy module requires tornado package %s" % error_msg)
from . import rpc
from .rpc import RPC_MAGIC, _server_env
from .._ffi.base import py_str

class ForwardHandler(object):
    """Forward handler to forward the message."""
    def _init_handler(self):
        """Initialize handler."""
        self._init_message = bytes()
        self._init_req_nbytes = 4
        self.forward_proxy = None
        self._magic = None
        self.timeout = None
        self._rpc_key_length = None
        self.rpc_key = None
        self._done = False

    def __del__(self):
        logging.info("Delete %s...", self.name())

    def name(self):
        """Name of this connection."""
        return "RPCConnection"

    def _init_step(self, message):
        if self._magic is None:
            assert len(message) == 4
            self._magic = struct.unpack('@i', message)[0]
            if self._magic != RPC_MAGIC:
                logging.info("Invalid RPC magic from %s", self.name())
                self.close()
            self._init_req_nbytes = 4
        elif self._rpc_key_length is None:
            assert len(message) == 4
            self._rpc_key_length = struct.unpack('@i', message)[0]
            self._init_req_nbytes = self._rpc_key_length
        elif self.rpc_key is None:
            assert len(message) == self._rpc_key_length
            self.rpc_key = py_str(message)
            self.on_start()
        else:
            assert False

    def on_start(self):
        """Event when the initialization is completed"""
        ProxyServerHandler.current.handler_ready(self)

    def on_data(self, message):
        """on data"""
        assert isinstance(message, bytes)
        if self.forward_proxy:
            self.forward_proxy.send_data(message)
        else:
            while message and self._init_req_nbytes > len(self._init_message):
                nbytes = self._init_req_nbytes - len(self._init_message)
                self._init_message += message[:nbytes]
                message = message[nbytes:]
                if self._init_req_nbytes == len(self._init_message):
                    temp = self._init_message
                    self._init_req_nbytes = 0
                    self._init_message = bytes()
                    self._init_step(temp)
            if message:
                logging.info("Invalid RPC protocol, too many bytes %s", self.name())
                self.close()

    def on_error(self, err):
        logging.info("%s: Error in RPC %s", self.name(), err)
        self.close_pair()

    def close_pair(self):
        if self.forward_proxy:
            self.forward_proxy.signal_close()
            self.forward_proxy = None
        self.close()

    def on_close_event(self):
        """on close event"""
        assert not self._done
        logging.info("RPCProxy:on_close %s ...", self.name())
        self._done = True
        self.forward_proxy = None
        if self.rpc_key:
            key = self.rpc_key[6:]
            if ProxyServerHandler.current._client_pool.get(key, None) == self:
                ProxyServerHandler.current._client_pool.pop(key)
            if ProxyServerHandler.current._server_pool.get(key, None) == self:
                ProxyServerHandler.current._server_pool.pop(key)


class TCPHandler(ForwardHandler):
    """Event driven TCP handler."""
    def __init__(self, sock, addr):
        self._init_handler()
        self.sock = sock
        assert self.sock
        self.addr = addr
        self.loop = ioloop.IOLoop.current()
        self.sock.setblocking(0)
        self.pending_write = []
        self._signal_close = False
        def event_handler(_, events):
            self._on_event(events)
        ioloop.IOLoop.current().add_handler(
            self.sock.fileno(), event_handler, self.loop.READ | self.loop.ERROR)

    def name(self):
        return "TCPSocket: %s:%s"  % (str(self.addr), self.rpc_key)

    def send_data(self, message, binary=True):
        assert binary
        self.pending_write.append(message)
        self._on_write()

    def _on_write(self):
        while self.pending_write:
            try:
                msg = self.pending_write[0]
                nsend = self.sock.send(msg)
                if nsend != len(msg):
                    self.pending_write[0] = msg[nsend:]
                else:
                    del self.pending_write[0]
            except socket.error as err:
                if err.args[0] in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                else:
                    self.on_error(err)
        if self.pending_write:
            self.loop.update_handler(
                self.sock.fileno(), self.loop.READ | self.loop.ERROR | self.loop.WRITE)
        else:
            if self._signal_close:
                self.close()
            else:
                self.loop.update_handler(
                    self.sock.fileno(), self.loop.READ | self.loop.ERROR)

    def _on_read(self):
        try:
            msg = bytes(self.sock.recv(4096))
            if msg:
                self.on_data(msg)
                return True
            else:
                self.close_pair()
        except socket.error as err:
            if err.args[0] in (errno.EAGAIN, errno.EWOULDBLOCK):
                pass
            else:
                self.on_error(e)
        return False

    def _on_event(self, events):
        if (events & self.loop.ERROR) or (events & self.loop.READ):
            if self._on_read() and (events & self.loop.WRITE):
                self._on_write()
        elif events & self.loop.WRITE:
            self._on_write()

    def signal_close(self):
        if not self.pending_write:
            self.close()
        else:
            self._signal_close = True

    def close(self):
        if self.sock is not None:
            logging.info("%s Close socket..", self.name())
            try:
                ioloop.IOLoop.current().remove_handler(self.sock.fileno())
                self.sock.close()
            except socket.error:
                pass
            self.sock = None
            self.on_close_event()


class WebSocketHandler(websocket.WebSocketHandler, ForwardHandler):
    """Handler for websockets."""
    def __init__(self, *args, **kwargs):
        super(WebSocketHandler, self).__init__(*args, **kwargs)
        self._init_handler()

    def name(self):
        return "WebSocketProxy"

    def on_message(self, message):
        self.on_data(message)

    def data_received(self, _):
        raise NotImplementedError()

    def send_data(self, message):
        try:
            self.write_message(message, True)
        except websocket.WebSocketClosedError as err:
            self.on_error(err)

    def on_close(self):
        if self.forward_proxy:
            self.forward_proxy.signal_close()
            self.forward_proxy = None
        self.on_close_event()

    def signal_close(self):
        self.close()


class RequestHandler(tornado.web.RequestHandler):
    """Handles html request."""
    def __init__(self, *args, **kwargs):
        self.page = open(kwargs.pop("file_path")).read()
        web_port = kwargs.pop("rpc_web_port", None)
        if web_port:
            self.page = self.page.replace(
                "ws://localhost:9190/ws",
                "ws://localhost:%d/ws" % web_port)
        super(RequestHandler, self).__init__(*args, **kwargs)

    def data_received(self, _):
        pass

    def get(self, *args, **kwargs):
        self.write(self.page)



class ProxyServerHandler(object):
    """Internal proxy server handler class."""
    current = None
    def __init__(self,
                 sock,
                 web_port,
                 timeout_client,
                 timeout_server,
                 index_page=None,
                 resource_files=None):
        assert ProxyServerHandler.current is None
        ProxyServerHandler.current = self
        if web_port:
            handlers = [
                (r"/ws", WebSocketHandler),
            ]
            if index_page:
                handlers.append(
                    (r"/", RequestHandler, {"file_path": index_page, "rpc_web_port": web_port}))
                logging.info("Serving RPC index html page at http://localhost:%d", web_port)
            resource_files = resource_files if resource_files else []
            for fname in resource_files:
                basename = os.path.basename(fname)
                pair = (r"/%s" % basename, RequestHandler, {"file_path": fname})
                handlers.append(pair)
                logging.info(pair)
            self.app = tornado.web.Application(handlers)
            self.app.listen(web_port)
        self.sock = sock
        self.sock.setblocking(0)
        self.loop = ioloop.IOLoop.current()
        def event_handler(_, events):
            self._on_event(events)
        self.loop.add_handler(
            self.sock.fileno(), event_handler, self.loop.READ)
        self._client_pool = {}
        self._server_pool = {}
        self.timeout_client = timeout_client
        self.timeout_server = timeout_server
        logging.info("RPCProxy: Websock port bind to %d", web_port)

    def _on_event(self, _):
        while True:
            try:
                conn, addr = self.sock.accept()
                TCPHandler(conn, addr)
            except socket.error as err:
                if err.args[0] in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break

    def _pair_up(self, lhs, rhs):
        lhs.forward_proxy = rhs
        rhs.forward_proxy = lhs
        lhs.send_data(struct.pack('@i', RPC_MAGIC))
        rhs.send_data(struct.pack('@i', RPC_MAGIC))
        logging.info("Pairup connect %s  and %s", lhs.name(), rhs.name())

    def handler_ready(self, handler):
        """Report handler to be ready."""
        logging.info("Handler ready %s", handler.name())
        key = handler.rpc_key[6:]
        if handler.rpc_key.startswith("server:"):
            pool_src, pool_dst = self._client_pool, self._server_pool
            timeout = self.timeout_server
        else:
            pool_src, pool_dst = self._server_pool, self._client_pool
            timeout = self.timeout_client

        if key in pool_src:
            self._pair_up(pool_src.pop(key), handler)
            return
        elif key not in pool_dst:
            pool_dst[key] = handler
            def cleanup():
                """Cleanup client connection if timeout"""
                if pool_dst.get(key, None) == handler:
                    logging.info("Timeout client connection %s, cannot find match key=%s",
                                 handler.name(), key)
                    pool_dst.pop(key)
                    handler.send_data(struct.pack('@i', RPC_MAGIC + 2))
                    handler.signal_close()
            self.loop.call_later(timeout, cleanup)
        else:
            logging.info("Duplicate connection with same key=%s", key)
            handler.send_data(struct.pack('@i', RPC_MAGIC + 1))
            handler.signal_close()

    def run(self):
        """Run the proxy server"""
        ioloop.IOLoop.current().start()

def _proxy_server(listen_sock,
                  web_port,
                  timeout_client,
                  timeout_server,
                  index_page,
                  resource_files):
    handler = ProxyServerHandler(listen_sock,
                                 web_port,
                                 timeout_client,
                                 timeout_server,
                                 index_page,
                                 resource_files)
    handler.run()


class Proxy(object):
    """Start RPC proxy server on a seperate process.

    Python implementation based on multi-processing.

    Parameters
    ----------
    host : str
        The host url of the server.

    port : int
        The TCP port to be bind to

    port_end : int, optional
        The end TCP port to search

    web_port : int, optional
        The http/websocket port of the server.

    timeout_client : float, optional
        Timeout of client until it sees a matching connection.

    timeout_server : float, optional
        Timeout of server until it sees a matching connection.

    index_page : str, optional
        Path to an index page that can be used to display at proxy index.

    resource_files : str, optional
        Path to local resources that can be included in the http request
    """
    def __init__(self,
                 host,
                 port=9091,
                 port_end=9199,
                 web_port=0,
                 timeout_client=600,
                 timeout_server=600,
                 index_page=None,
                 resource_files=None):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.port = None
        for my_port in range(port, port_end):
            try:
                sock.bind((host, my_port))
                self.port = my_port
                break
            except socket.error as sock_err:
                if sock_err.errno in [98, 48]:
                    continue
                else:
                    raise sock_err
        if not self.port:
            raise ValueError("cannot bind to any port in [%d, %d)" % (port, port_end))
        logging.info("RPCProxy: client port bind to %s:%d", host, self.port)
        sock.listen(1)
        self.proc = multiprocessing.Process(
            target=_proxy_server, args=(sock, web_port,
                                        timeout_client, timeout_server,
                                        index_page, resource_files))
        self.proc.start()
        self.host = host

    def terminate(self):
        """Terminate the server process"""
        if self.proc:
            logging.info("Terminating Proxy Server...")
            self.proc.terminate()
            self.proc = None

    def __del__(self):
        self.terminate()


def websocket_proxy_server(url, key=""):
    """Create a RPC server that uses an websocket that connects to a proxy.

    Parameters
    ----------
    url : str
        The url to be connected.

    key : str
        The key to identify the server.
    """
    def create_on_message(conn):
        def _fsend(data):
            data = bytes(data)
            conn.write_message(data, binary=True)
            return len(data)
        on_message = rpc._CreateEventDrivenServer(_fsend, "WebSocketProxyServer")
        return on_message

    @gen.coroutine
    def _connect(key):
        conn = yield websocket.websocket_connect(url)
        on_message = create_on_message(conn)
        temp = _server_env()
        # Start connecton
        conn.write_message(struct.pack('@i', RPC_MAGIC), binary=True)
        key = "server:" + key
        conn.write_message(struct.pack('@i', len(key)), binary=True)
        conn.write_message(key.encode("utf-8"), binary=True)
        msg = yield conn.read_message()
        assert len(msg) >= 4
        magic = struct.unpack('@i', msg[:4])[0]
        if magic == RPC_MAGIC + 1:
            raise RuntimeError("key: %s has already been used in proxy" % key)
        elif magic == RPC_MAGIC + 2:
            logging.info("RPCProxy do not have matching client key %s", key)
        elif magic != RPC_MAGIC:
            raise RuntimeError("%s is not RPC Proxy" % url)
        logging.info("Connection established")
        msg = msg[4:]
        if msg:
            on_message(bytearray(msg), 3)

        while True:
            try:
                msg = yield conn.read_message()
                if msg is None:
                    break
                on_message(bytearray(msg), 3)
            except websocket.WebSocketClosedError as err:
                break
        logging.info("WebSocketProxyServer closed...")
        temp.remove()
        ioloop.IOLoop.current().stop()
    ioloop.IOLoop.current().spawn_callback(_connect, key)
    ioloop.IOLoop.current().start()
