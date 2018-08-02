"""Test database"""
import copy
import logging
import time

import numpy as np
import tvm

from tvm import autotvm
from tvm.autotvm import database
from tvm.autotvm.measure.measure_methods import HashMismatchError
from tvm.autotvm.record import encode, MeasureInput, MeasureResult

from test_autotvm_common import get_sample_task, get_sample_records

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

def test_db_filter():
    logging.info("test db filter ...")

    # Pick a GPU target because there are more likely to be failures/invalid configs
    task, target = get_sample_task()

    ctx = tvm.context(str(target))
    if not ctx.exist:
        logging.warning("Skip this test because there is no supported device for test")

    batch_size = 2

    measure_option = autotvm.measure_option('local', do_fork=False, timeout=2)
    measure_batch = autotvm.measure.create_measure_batch(task, measure_option)

    ct = 0
    all_inputs = list()
    all_results = list()
    batches = list()
    tuner = autotvm.tuner.RandomTuner(task)
    while ct < TRIAL_LIMIT:
        inputs = list()
        for i in range(batch_size):
            cfg = tuner.next_batch(1)[0]
            inputs.append((MeasureInput(target, task, cfg)))
            all_inputs.append(inputs[-1])
        batches.append(inputs)
        results = measure_batch(inputs)
        all_results += results
        ct += 1

    del measure_batch

    db = database.DummyDatabase()
    db.flush()

    # First setting, memoize one input at a time, check that each is saved and replayed
    measure_option = autotvm.measure_option('local', do_fork=False, timeout=2, replay_db=db)
    measure_batch = autotvm.measure.create_measure_batch(task, measure_option)

    for i in range(len(all_inputs)+1):
        db.flush()
        for j in range(i):
            db.save(all_inputs[j], all_results[j])

        for k in range(len(batches)):
            batch = batches[k]
            batch_result = measure_batch(batch)
            for l in range(batch_size):
                all_idx = k*batch_size + l
                assert batch_result[l] is not None
                if all_idx < i:
                    assert encode(batch[l], batch_result[l]) == encode(batch[l], all_results[all_idx]), \
                        "(no retry) EXPECTED MATCH, GOT MISMATCH"
                else:
                    assert encode(batch[l], batch_result[l]) != encode(batch[l], all_results[all_idx]), \
                        "(no retry) EXPECTED MISMATCH, GOT MATCH"

    del measure_batch

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

def test_db_save_replay():
    logging.info("test db save (from measure_batch) and replay ...")
    _db = database.DummyDatabase()
    _db.flush()

    task, target = get_sample_task()

    ctx = tvm.context(str(target))
    if not ctx.exist:
        logging.warning("Skip this test because there is no supported device for test")

    measure_option = autotvm.measure_option('local',
                                            do_fork=False,
                                            timeout=2,
                                            replay_db=_db)
    measure_batch = autotvm.measure.create_measure_batch(task, measure_option)

    batch_size = 2

    ct = 0
    all_inputs = list()
    all_results = list()
    batches = list()
    tuner = autotvm.tuner.RandomTuner(task)
    while ct < TRIAL_LIMIT:
        inputs = list()
        for i in range(batch_size):
            cfg = tuner.next_batch(1)[0]
            inputs.append((MeasureInput(target, task, cfg)))
            all_inputs.append(inputs[-1])
        batches.append(inputs)
        results = measure_batch(inputs)
        all_results += results
        ct += 1
    callback = autotvm.callback.log_to_database(_db)
    callback(None, all_inputs, all_results)

    assert len(_db.db.keys()) == batch_size * TRIAL_LIMIT, \
        "%d vs %d" % (len(_db.db.keys()), batch_size * TRIAL_LIMIT)

    all_results_2 = measure_batch(all_inputs)
    all_results_3 = measure_batch(all_inputs)

    for i in range(len(all_results)):
        encr1 = encode(all_inputs[i], all_results[i])
        encr2 = encode(all_inputs[i], all_results_2[i])
        encr3 = encode(all_inputs[i], all_results_3[i])
        assert encr1 == encr2, "EXPECTED MATCH WITH SAVE REPLAY (first replay), got MISMATCH"
        assert encr2 == encr3, "EXPECTED MATCH WITH SAVE REPLAY (second replay), got MISMATCH"

    del measure_batch

def test_check_hashmismatch():
    logging.info("test hash mismatch check")

    task, target = get_sample_task()

    ctx = tvm.context(str(target))
    if not ctx.exist:
        logging.warning("Skip this test because there is no supported device for test")

    measure_option = autotvm.measure_option('local', do_fork=False)
    measure_batch = autotvm.measure.create_measure_batch(task, measure_option)

    inputs = list()
    cfg = task.config_space.get(np.random.randint(len(task.config_space)))
    # notvalidh is not a valid CRC32 hash (not hex)
    cfg.code_hash = 'notvalidh'
    inputs.append((MeasureInput(target, task, cfg)))

    try:
        results = measure_batch(inputs)
        assert False, "HashMismatchError should be raised"
    except HashMismatchError:
        pass

    del measure_batch

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_save_load()
    test_db_filter()
    test_db_hash()
    test_db_latest_all()
    test_db_save_replay()
    test_check_hashmismatch()
