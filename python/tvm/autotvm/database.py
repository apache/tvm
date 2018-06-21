# pylint: disable=consider-using-enumerate,invalid-name
"""
Database of MeasureInput/MeasureResult pair.
This can be used for replaying measurement.
"""
import os

from .record import encode, decode, measure_str_key


class Database(object):
    """
    Base class for a record database object.
    """
    def load(self, inp, get_all=False):
        """
        Load a result based on an input's string key

        Parameters
        ----------
        inp: MeasureInput
            to be translated into key for RedisDB
        get_all: bool, optional
            Whether the latest result (or all matching results) should be returned

        Returns
        -------
        rec: MeasureResult if previously saved, otherwise None
        """
        raise NotImplementedError()

    def save(self, inp, res, extend=False):
        """
        Save a result based on an input's string key

        Parameters
        ----------
        inp: MeasureInput
            to be translated into key for RedisDB
        res: MeasureResult
            to associate with key
        extend:
            Whether to extend existing MeasureResults if they exist
        """
        raise NotImplementedError()

    def filter_inputs(self, measure_inputs, retry=False):
        """
        Filter a measure_inputs batch based on saved db results

        Parameters
        ----------
        measure_inputs: Array of MeasureInput
            measure_inputs as expected in measure_batch
        retry: bool
            whether to retry if the saved result is a failure

        Returns
        -------
        partial_results: Array of MeasureResult
            a full list of result, where None denotes no corresponding saved result
        unsaved: Array of MeasureInput
            a list that only contains unsaved inputs
        """
        partial_results = [None] * len(measure_inputs)
        unsaved = list()
        for i in range(len(measure_inputs)):
            inp = measure_inputs[i]
            res = self.load(inp)
            skip = (res is not None and
                    (not retry or (retry and res.error_no == 0)))
            if skip:
                partial_results[i] = res
            else:
                unsaved.append(inp)
        return partial_results, unsaved


def unpack_existing(current):
    """
    Unpack existing MeasureResults (in str format)

    Parameters
    ----------
    current: str
        The current list of MeasureResults (in str format)
    """
    # pylint: disable = eval-used
    return eval(current)


def extend_existing(current, new):
    """
    Extend a list of MeasureResults (str-format in db) to add a new result.
    Duplicate results are ignored.

    Parameters
    ----------
    current: str
        The current MeasureResults (in str format)
    new: str
        The new MeasureResult (in str format)
    """
    current_list = unpack_existing(current)
    if new in current_list:
        return current_list
    current_list.append(new)
    return current_list


class RedisDatabase(Database):
    """
    Redis version of record database
    """

    REDIS_PROD = 15
    REDIS_LOCA = 14
    REDIS_TEST = 13        # for unit test
    REDIS_NIGHT_TEMP = 12  # for nightly report (will be flushed after every workload)

    def __init__(self, db_index=REDIS_PROD):
        import redis

        if db_index == RedisDatabase.REDIS_TEST:
            host = 'localhost'
        else:
            host = os.environ.get('TVM_FLEET_HOST')
        self.db = redis.StrictRedis(host=host, port=6379, db=db_index)
        self.db_index = db_index

    def load(self, inp, get_all=False):
        current = self.db.get(measure_str_key(inp))
        if current is not None:
            records = unpack_existing(current)
            results = [decode(rec)[1] for rec in records]
            if get_all:
                return results
            if len(results) < 1:
                return None
            return max(results, key=lambda result: result.timestamp)
        return current

    def save(self, inp, res, extend=False):
        current = self.db.get(measure_str_key(inp))
        if not extend or current is None:
            return self.db.set(measure_str_key(inp), [encode(inp, res)])
        return self.db.set(measure_str_key(inp), extend_existing(current,
                                                                 encode(inp, res)))

    def dump_target(self, target):
        """
        Dump all of the records for a particular target

        Parameters
        ----------
        target: tvm.target.Target
            The target to dump

        Returns
        -------
        list of records (inp, result) matching the target
        """
        matched_records = list()
        # may consider filtering in iterator in the future
        for key in self.db.scan_iter():
            current = self.db.get(key)
            records = unpack_existing(current)
            decoded = list()
            for rec in records:
                try:
                    decoded.append(decode(rec))
                except TypeError: # got a badly formatted/old format record
                    continue
            if len(decoded) < 1:
                continue

            inps, results = zip(*decoded)
            inp = inps[0]
            if inp.target.__repr__() != target.__repr__():
                continue
            result = max(results, key=lambda res: res.timestamp)
            matched_records.append((inp, result))
        return matched_records

    def flush(self):
        """Flush the database."""
        self.db.flushdb()
