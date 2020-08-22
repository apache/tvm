import abc
import atexit
import contextlib
import logging
import os
import signal
import string
import subprocess
import time
import typing

_LOG = logging.getLogger(__name__)


class Transport(metaclass=abc.ABCMeta):

    def __enter__(self):
      self.open()
      return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
      self.close()

    @abc.abstractmethod
    def open(self):
      raise NotImplementedError()

    @abc.abstractmethod
    def close(self):
      raise NotImplementedError()

    @abc.abstractmethod
    def read(self, n):
      raise NotImplementedError()

    @abc.abstractmethod
    def write(self, data):
      raise NotImplementedError()


class TransportLogger(Transport):

  def __init__(self, name, child, logger=None, level=logging.INFO):
    self.name = name
    self.child = child
    self.logger = logger or _LOG
    self.level = level

  # Construct PRINTABLE to exclude whitespace from string.printable.
  PRINTABLE = (string.digits + string.ascii_letters + string.punctuation)

  @classmethod
  def _to_hex(cls, data):
    lines = []
    if not data:
      lines.append('')
      return lines

    for i in range(0, (len(data) + 15) // 16):
      chunk = data[i * 16:(i + 1) * 16]
      hex_chunk = ' '.join(f'{c:02x}' for c in chunk)
      ascii_chunk = ''.join((chr(c) if chr(c) in cls.PRINTABLE else '.') for c in chunk)
      lines.append(f'{i * 16:04x}  {hex_chunk:47}  {ascii_chunk}')

    if len(lines) == 1:
      lines[0] = lines[0][6:]

    return lines

  def open(self):
    self.logger.log(self.level, 'opening transport')
    self.child.open()

  def close(self):
    self.logger.log(self.level, 'closing transport')
    return self.child.close()

  def read(self, n):
    data = self.child.read(n)
    hex_lines = self._to_hex(data)
    if len(hex_lines) > 1:
      self.logger.log(self.level, '%s read %4d B -> [%d B]:\n%s',
                      self.name, n, len(data), '\n'.join(hex_lines))
    else:
      self.logger.log(self.level, '%s read %4d B -> [%d B]: %s', self.name, n, len(data), hex_lines[0])

    return data

  def write(self, data):
    bytes_written = self.child.write(data)
    hex_lines = self._to_hex(data[:bytes_written])
    if len(hex_lines) > 1:
      self.logger.log(self.level, '%s write      <- [%d B]:\n%s', self.name, bytes_written, '\n'.join(hex_lines))
    else:
      self.logger.log(self.level, '%s write      <- [%d B]: %s', self.name, bytes_written, hex_lines[0])

    return bytes_written


class SubprocessTransport(Transport):

  def __init__(self, args, **kw):
    self.args = args
    self.kw = kw
    self.popen = None

  def open(self):
    self.kw['stdout'] = subprocess.PIPE
    self.kw['stdin'] = subprocess.PIPE
    self.kw['bufsize'] = 0
    self.popen = subprocess.Popen(self.args, **self.kw)
    self.stdin = self.popen.stdin
    self.stdout = self.popen.stdout

  def write(self, data):
    to_return = self.stdin.write(data)
    self.stdin.flush()

    return to_return

  def read(self, n):
    return self.stdout.read(n)

  def close(self):
    self.stdin.close()
    self.stdout.close()
    self.popen.terminate()


class DebugWrapperTransport(Transport):

  def __init__(self, debugger, transport):
    self.debugger = debugger
    self.transport = transport
    self.debugger.on_terminate_callbacks.append(self.transport.close)

  def open(self):
    self.debugger.Start()

    try:
      self.transport.open()
    except Exception:
      self.debugger.Stop()
      raise

  def write(self, data):
    return self.transport.write(data)

  def read(self, n):
    return self.transport.read(n)

  def close(self):
    self.transport.close()
    self.debugger.Stop()


TransportContextManager = typing.ContextManager[Transport]
