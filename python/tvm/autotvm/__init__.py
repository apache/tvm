"""The auto-tuning module of tvm

This module includes:

* Tuning space definition API

* Efficient auto-tuners

* Tuning result and database support

* Distributed measurement to scale up tuning
"""

from . import database
from . import feature
from . import measure
from . import record
from . import task
from . import tuner
from . import util
from . import env
from . import tophub

# some shortcuts
from .measure import measure_option, MeasureInput, MeasureResult, MeasureErrorNo, \
    LocalBuilder, LocalRunner, RPCRunner
from .tuner import callback
from .task import template, get_config, create, ConfigSpace, ConfigEntity, \
    register_topi_compute, register_topi_schedule, \
    DispatchContext, FallbackContext, ApplyHistoryBest as apply_history_best
from .env import GLOBAL_SCOPE
