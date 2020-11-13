from __future__ import absolute_import

import json
from json import JSONEncoder
from enum import IntEnum

HAGO_LOG_VERSION = 0.1

class Strategy(object):
    def __init__(self, model_hash, bits, thresholds):
        self.model_hash = model_hash
        self.bits = bits
        self.thresholds = [float(th) for th in thresholds]

    def to_json(self):
        json_dict = self.__dict__.copy()
        return json_dict

    def __str__(self):
        return 'Strategy(model_hash=' + str(self.model_hash) + \
                ', bits=' + str(self.bits) + \
                ', thresholds=' + str(self.thresholds) + ')'


class MeasureKind(object):
    MEASURE_KEYS = ['accuracy', 'kl_distance']
    Accuracy = 0
    # KLDistance = 1
    @staticmethod
    def enum_to_str(kind):
        assert kind < len(MeasureKind.MEASURE_KEYS)
        return MeasureKind.MEASURE_KEYS[kind]

    @staticmethod
    def str_to_enum(key): 
        assert key in MeasureKind.MEASURE_KEYS
        return MeasureKind.MEASURE_KEYS.index(key)


# TODO(ziheng): consider multiple measure metric in the future: latency, energy, etc 
class MeasureResult(object):
    def __init__(self, accuracy=None, kl_distance=None):
        self.accuracy = accuracy
        self.kl_distance = kl_distance

    def to_json(self):
        json_dict = self.__dict__.copy()
        return json_dict

    def __str__(self):
        keys = self.__dict__.keys()
        pairs = [key + '=' + str(getattr(self, key)) for key in keys]
        return 'MeasureResult(' +  ', '.join(pairs) + ')'

class Measure(object):
    def __init__(self, strategy, result):
        self.version = HAGO_LOG_VERSION
        self.strategy = strategy
        self.result = result

    def to_json(self):
        json_dict = self.__dict__.copy()
        return json_dict

    def __str__(self):
        return 'Measure(version=' + str(self.version) + \
                ', strategy=' + str(self.strategy) + \
                ', result=' + str(self.result) + \
                ')'


def best_measure(measures, kind):
    def compare_key(m):
        key = MeasureKind.enum_to_str(kind)
        attr = getattr(m.result, key)
        nbit = sum(m.strategy.bits)
        return (attr, -nbit)
    return max(measures, key=compare_key)


def serialize(obj):
    class Encoder(JSONEncoder):
        def default(self, obj):
            if hasattr(obj, 'to_json'):
                return obj.to_json()
            return json.JSONEncoder.default(self, obj)
    return json.dumps(obj, cls=Encoder, sort_keys=True)


def deserialize(json_str):
    def decode_strategy(obj):
        model_hash = obj['model_hash']
        bits = obj['bits']
        thresholds = obj['thresholds']
        return Strategy(model_hash, bits, thresholds)
    
    def decode_result(obj):
        accuracy = obj['accuracy']
        kl_distance = obj['kl_distance']
        return MeasureResult(accuracy, kl_distance)
    
    json_data = json.loads(json_str)
    strategy = decode_strategy(json_data['strategy'])
    result = decode_result(json_data['result'])
    return Measure(strategy, result)


def load_from_file(fname):
    records = []
    with open(fname) as fin:
        for json_str in fin:
            record = deserialize(json_str)
            records.append(record)
    return records


# FIXME(ziheng))
def pick_best(fname, key):
    records = load_from_file(fname)
    records.sort(key=lambda rec: getattr(rec['result'], key))
    if key in ['accuracy']:
        return records[-1]
    elif key in ['kl_divergence']:
        return records[0]
    else:
        raise ValueError
