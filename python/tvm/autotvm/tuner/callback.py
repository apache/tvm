# pylint: disable=consider-using-enumerate,invalid-name
"""Namespace of callback utilities of AutoTVM"""

import numpy as np

from .. import record

def log_to_file(file_out, protocol='json'):
    """Log the tuning records into file.
    The rows of the log are stored in the format of autotvm.record.encode.

    Parameters
    ----------
    file_out : File or str
        The file to log to.
    protocol: str, optional
        The log protocol. Can be 'json' or 'pickle'

    Returns
    -------
    callback : callable
        Callback function to do the logging.
    """

    def _callback(_, inputs, results):
        """Callback implementation"""
        if isinstance(file_out, str):
            with open(file_out, "a") as f:
                for inp, result in zip(inputs, results):
                    f.write(record.encode(inp, result, protocol) + "\n")
        else:
            for inp, result in zip(inputs, results):
                file_out.write(record.encode(inp, result, protocol) + "\n")
    return _callback


def save_tuner_state(prefix, save_every_sample=100):
    """Save the state of tuner

    Parameters
    ----------
    prefix : srt
        prefix of the filename to store state
    save_every_sample: int
        save the state every x samples

    Returns
    -------
    callback : function
        Callback function to do the auto saving.
    """
    def _callback(tuner, inputs, results):
        for _, __ in zip(inputs, results):
            try:
                ct = len(tuner.visited)
            except AttributeError:
                ct = 0
            if ct % save_every_sample == 0:
                tuner.save_state(prefix + "_%d.state" % ct)

    return _callback


def log_to_redis(host="localhost", port=6379, dbn=11):
    """Record the tuning record to a redis DB.

    Parameters
    ----------
    host: str, optional
        Host address of redis db
    port: int, optional
        Port of redis db
    dbn: int, optional
        which redis db to use, default 11
    """
    # import here so only depend on redis when necessary
    import redis
    red = redis.StrictRedis(host=host, port=port, db=dbn)

    def _callback(_, inputs, results):
        """Callback implementation"""
        for inp, result in zip(inputs, results):
            red.set(inp, result)
    return _callback

class Monitor(object):
    """A monitor to collect statistic during tuning"""
    def __init__(self):
        self.scores = []
        self.timestamps = []

    def __call__(self, tuner, inputs, results):
        for inp, res in zip(inputs, results):
            if res.error_no == 0:
                flops = inp.task.flop / np.mean(res.costs)
                self.scores.append(flops)
            else:
                self.scores.append(0)

            self.timestamps.append(res.timestamp)

    def reset(self):
        self.scores = []
        self.timestamps = []

    def trial_scores(self):
        """get scores (currently is flops) of all trials"""
        return np.array(self.scores)

    def trial_timestamps(self):
        """get wall clock time stamp of all trials"""
        return np.array(self.timestamps)
