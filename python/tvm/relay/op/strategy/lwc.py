import logging
import re

from tvm import tir, topi
from tvm.auto_scheduler import is_auto_scheduler_enabled
from tvm.meta_schedule import is_meta_schedule_enabled
from tvm.relay.ty import is_dynamic
from tvm.target import Target
from tvm.te import SpecializedCondition

from .. import op as _op
from .generic import *

logger = logging.getLogger("strategy")
@schedule_pool.register("x330")
def schedule_pool_lwc(attrs, outs, target):
    with target:
        return topi.lwc.schedule_pool(outs, attrs.layout)
