import importlib
import json
import typing


JsonSerializable = typing.Union[int, float, str, None, bool]


class SerializedFactoryError(Exception):
  """Raised when ClassFactory.from_json is invoked with an invalid JSON blob."""


class ClassFactory:
  """Describes a JSON-serializable class instantiation, for use with the RPC server."""

  # When not None, the superclass from which all cls must derive.
  SUPERCLASS = None

  def __init__(self, cls : typing.Callable, init_args : typing.List[JsonSerializable],
               init_kw : typing.Dict[str, JsonSerializable]):
    self.cls = cls
    self.init_args = init_args
    self.init_kw = init_kw

  def override_kw(self, **kw_overrides):
    kw = self.init_kw
    if kw_overrides:
      kw = dict(kw)
      for k, v in kw_overrides.items():
        kw[k] = v

    return self.__class__(self.cls, self.init_args, kw)

  def instantiate(self):
    return self.cls(*self.init_args, **self.init_kw)

  @property
  def to_json(self):
    return json.dumps({
      'cls': '.'.join([self.cls.__module__, self.cls.__name__]),
      'init_args': self.init_args,
      'init_kw': self.init_kw,
    })

  EXPECTED_KEYS = ('cls', 'init_args', 'init_kw')

  @classmethod
  def from_json(cls, data):
    obj = json.loads(data)
    if not isinstance(obj, dict):
      raise SerializedFactoryError(f'deserialized json payload: want dict, got: {obj!r}')

    for key in cls.EXPECTED_KEYS:
      if key not in obj:
        raise SerializedFactoryError(f'deserialized json payload: expect key {key}, got: {obj!r}')

    cls_package_name, cls_name = obj['cls'].rsplit('.', 1)
    cls_package = importlib.import_module(cls_package_name)
    cls_obj = getattr(cls_package, cls_name)
    return cls(cls_obj, obj['init_args'], obj['init_kw'])
