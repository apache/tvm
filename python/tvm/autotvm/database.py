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


def filter_inputs(db, measure_inputs, retry=False):
    """
    Filter a measure_inputs batch based on saved db results

    Parameters
    ----------
    db: Database
        database object
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
    partial_results = list()
    unsaved = list()
    for inp in measure_inputs:
        res = db.load(inp)
        if res is None or (retry and res.error_no != 0):
            unsaved.append(inp)
            partial_results.append(None)
        else:
            partial_results.append(res)
    return partial_results, unsaved

class RedisDatabase(Database):
    """
    Redis version of record database
    """
    REDIS_PROD = 15
    REDIS_LOCA = 14
    REDIS_TEST = 13        # for unit test
    REDIS_NIGHT_TEMP = 12  # for nightly report (will be flushed after every workload)

    MAGIC_SPLIT = "$"

    def __init__(self, db_index=REDIS_PROD):
        import redis

        if db_index == RedisDatabase.REDIS_TEST:
            host = 'localhost'
        else:
            host = os.environ.get('TVM_FLEET_HOST')
        self.db = redis.StrictRedis(host=host, port=6379, db=db_index)
        self.db_index = db_index

    def set(self, key, value):
        self.db.set(key, value)

    def get(self, key):
        return self.db.get(key)

    def load(self, inp, get_all=False):
        current = self.get(measure_str_key(inp))
        if current is not None:
            current = str(current)
            records = [decode(x) for x in current.split(RedisDatabase.MAGIC_SPLIT)]
            results = [rec[1] for rec in records]
            if get_all:
                return results
            return max(results, key=lambda result: result.timestamp)
        return current

    def save(self, inp, res, extend=False):
        current = self.get(measure_str_key(inp))
        if not extend or current is None:
            self.set(measure_str_key(inp),
                     RedisDatabase.MAGIC_SPLIT.join([encode(inp, res)]))
        else:
            current = current.split(RedisDatabase.MAGIC_SPLIT)
            self.set(measure_str_key(inp),
                     RedisDatabase.MAGIC_SPLIT.join(current + [encode(inp, res)]))

    def filter(self, func):
        """
        Dump all of the records for a particular target

        Parameters
        ----------
        func: callable
            The signature of the function is bool (MeasureInput, Array of MeasureResult)

        Returns
        -------
        list of records (inp, result) matching the target

        Examples
        --------
        get records for a target
        >>> db.filter(lambda inp, resulst: "cuda" in inp.target.keys)
        """
        matched_records = list()
        # may consider filtering in iterator in the future
        for key in self.db:
            current = self.get(key)
            try:
                records = [decode(x) for x in current.spilt(RedisDatabase.MAGIC_SPLIT)]
            except TypeError:  # got a badly formatted/old format record
                continue

            inps, results = zip(*records)
            inp = inps[0]
            if not func(inp, results):
                continue
            result = max(results, key=lambda res: res.timestamp)
            matched_records.append((inp, result))
        return matched_records

    def flush(self):
        self.db.flushdb()

class DummyDatabase(RedisDatabase):
    """
    A database based on python dictionary for testing.
    """

    def __init__(self):
        # pylint: disable=super-init-not-called
        self.db = {}

    def set(self, key, value):
        self.db[key] = value

    def get(self, key):
        return self.db.get(key)

    def flush(self):
        self.db = {}
