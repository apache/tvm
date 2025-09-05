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

import os
import sys
import time

# Platform-specific imports for file locking
if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


class FileLock:
    """
    A cross-platform file locking mechanism using Python's standard library.
    This class implements an advisory lock, which must be respected by all
    cooperating processes.
    """

    def __init__(self, lock_file_path):
        self.lock_file_path = lock_file_path
        self._file_descriptor = None

    def __enter__(self):
        """
        Context manager protocol: acquire the lock upon entering the 'with' block.
        This method will block indefinitely until the lock is acquired.
        """
        self.blocking_acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager protocol: release the lock upon exiting the 'with' block.
        """
        self.release()
        return False  # Propagate exceptions, if any

    def acquire(self):
        """
        Acquires an exclusive, non-blocking lock on the file.
        Returns True if the lock was acquired, False otherwise.
        """
        try:
            if sys.platform == "win32":
                self._file_descriptor = os.open(
                    self.lock_file_path, os.O_RDWR | os.O_CREAT | os.O_BINARY
                )
                msvcrt.locking(self._file_descriptor, msvcrt.LK_NBLCK, 1)
            else:  # Unix-like systems
                self._file_descriptor = os.open(self.lock_file_path, os.O_WRONLY | os.O_CREAT)
                fcntl.flock(self._file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (IOError, BlockingIOError):
            if self._file_descriptor is not None:
                os.close(self._file_descriptor)
                self._file_descriptor = None
            return False
        except Exception as e:
            if self._file_descriptor is not None:
                os.close(self._file_descriptor)
                self._file_descriptor = None
            raise RuntimeError(f"An unexpected error occurred: {e}")

    def blocking_acquire(self, timeout=None, poll_interval=0.1):
        """
        Waits until an exclusive lock can be acquired, with an optional timeout.

        Args:
            timeout (float): The maximum time to wait for the lock in seconds.
                             A value of None means wait indefinitely.
            poll_interval (float): The time to wait between lock attempts in seconds.
        """
        start_time = time.time()
        while True:
            if self.acquire():
                return True

            # Check for timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Failed to acquire lock on '{self.lock_file_path}' after {timeout} seconds."
                )

            time.sleep(poll_interval)

    def release(self):
        """
        Releases the lock and closes the file descriptor.
        """
        if self._file_descriptor is not None:
            if sys.platform == "win32":
                msvcrt.locking(self._file_descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._file_descriptor, fcntl.LOCK_UN)
            os.close(self._file_descriptor)
            self._file_descriptor = None
