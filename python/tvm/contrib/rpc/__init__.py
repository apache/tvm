"""Lightweight TVM RPC module.

RPC enables connect to a remote server, upload and launch functions.
This is useful to for cross-compile and remote testing,
The compiler stack runs on local server, while we use RPC server
to run on remote runtime which don't have a compiler available.

The test program compiles the program on local server,
upload and run remote RPC server, get the result back to verify correctness.
"""

from .server import Server
from .client import RPCSession, LocalSession, TrackerSession, connect, connect_tracker
