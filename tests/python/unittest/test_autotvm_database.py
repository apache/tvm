"""Test database"""
import copy
import logging

from tvm.autotvm import database
from tvm.autotvm.record import encode, MeasureResult

from test_autotvm_common import get_sample_records

def test_save_load():
    logging.info("test basic db load/save ...")
    records = get_sample_records(3)
    inp1, res1 = records[0]
    inp2, res2 = records[1]
    inp3, _ = records[2]

    _db = database.DummyDatabase()
    _db.flush()
    _db.save(inp1, res1)
    _db.save(inp2, res2)

    load1 = _db.load(inp1)
    load2 = _db.load(inp2)
    load3 = _db.load(inp3)
    assert load1 == res1
    assert load2 == res2
    assert load3 is None
    assert load1 != load2

TRIAL_LIMIT = 2

def test_db_hash():
    logging.info("test db hash check ...")
    inp1, res1 = get_sample_records(1)[0]
    inp2 = copy.deepcopy(inp1)
    inp1.config.code_hash = 'cafecafe'
    inp2.config.code_hash = 'dbffdbff'
    res2l = list(tuple(res1))

    # set timestamp
    res2l[-1] = -1
    res2 = MeasureResult(*res2l)
    _db = database.DummyDatabase()
    _db.flush()
    _db.save(inp1, res1, extend=True)
    _db.save(inp2, res2, extend=True)

    load1 = _db.load(inp1)
    load2 = _db.load(inp2)
    assert load1 != load2
    assert load1.timestamp != -1
    assert load2.timestamp == -1

def test_db_latest_all():
    logging.info("test db load w/ multiple results ...")
    inp1, res1 = get_sample_records(1)[0]
    lis1 = list(tuple(res1))
    lis2 = list(tuple(res1))
    lis3 = list(tuple(res1))

    # set timestamp
    lis1[-1] = 0.0
    lis2[-1] = 1.1
    lis3[-1] = 9999.9999
    res1 = MeasureResult(*lis1)
    res2 = MeasureResult(*lis2)
    res3 = MeasureResult(*lis3)

    _db = database.DummyDatabase()
    _db.flush()
    _db.save(inp1, res1, extend=True)
    load1 = _db.load(inp1)
    assert load1.timestamp == 0.0
    _db.save(inp1, res2, extend=True)
    load2 = _db.load(inp1)
    assert load2.timestamp == 1.1
    _db.save(inp1, res3, extend=True)
    load3 = _db.load(inp1)
    assert load3.timestamp == 9999.9999

    load4 = _db.load(inp1, get_all=True)
    assert encode(inp1, load4[0]) == encode(inp1, res1)
    assert encode(inp1, load4[1]) == encode(inp1, res2)
    assert encode(inp1, load4[2]) == encode(inp1, res3)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_save_load()
    test_db_hash()
    test_db_latest_all()
