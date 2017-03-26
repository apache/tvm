"""Common utilities for test"""

class FIFODelayedReader(object):
    """Reader that have specified ready lag."""
    def __init__(self, read_data, read_valid, read_ready, lag):
        self.read_data = read_data
        self.read_valid = read_valid
        self.read_ready = read_ready
        self.read_ready.put_int(1)
        self.lag = list(reversed(lag))
        self.data = []
        self.wait_counter = 0
        self.wait_state = False

    def __call__(self):
        """Logic as if always at pos-edge"""
        if not self.wait_state:
            if (self.read_ready.get_int() and
                self.read_valid.get_int()):
                self.data.append(self.read_data.get_int())
                self.wait_counter = self.lag.pop() if self.lag else 0
                self.wait_state = True

        if self.wait_state:
            if self.wait_counter == 0:
                self.read_ready.put_int(1)
                self.wait_state = False
            else:
                self.wait_counter -= 1
                self.read_ready.put_int(0)


class FIFODelayedWriter(object):
    """Auxiliary class to write to FIFO """
    def __init__(self, write_data, write_valid, write_ready, data, lag):
        self.write_data = write_data
        self.write_valid = write_valid
        self.write_ready = write_ready
        self.write_valid.put_int(0)
        self.lag = list(reversed(lag))
        self.data = list(reversed(data))
        self.wait_counter = 0
        self.wait_state = True

    def __call__(self):
        """Logic as if always at pos-edge"""
        if not self.wait_state:
            if self.write_ready.get_int():
                self.wait_counter = self.lag.pop() if self.lag else 0
                self.wait_state = True

        if self.wait_state:
            if self.wait_counter == 0:
                if self.data:
                    self.write_valid.put_int(1)
                    self.write_data.put_int(self.data.pop())
                    self.wait_state = False
                else:
                    self.write_valid.put_int(0)
            else:
                self.write_valid.put_int(0)
                self.wait_counter -= 1
