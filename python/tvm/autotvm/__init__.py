"""The auto-tuning module of tvm

This module includes
* tuning space definition API
* efficient auto-tuners
* distributed measurement to scale up tuning
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
from .tuner import callback
from .task import template, get_config, create
from .record import ApplyHistoryBest as apply_history_best
