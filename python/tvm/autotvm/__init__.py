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

# some shortcuts
from .measure import measure_option, MeasureInput, MeasureResult, MeasureErrorNo
from .tuner import callback, tune_tasks
from .task import template, get_config, create, ConfigSpace, ConfigEntity
from .record import ApplyHistoryBest as apply_history_best, load_op_param
