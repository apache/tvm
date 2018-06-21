"""
Template function module.

Template functions are basic components of tuning.
For tuning, we can compose up template functions to
build a task that can be consumed by the tuner.

Template is a function that defines 1) the configuration space of a schedule
and 2) how to schedule according to the parameters in these space.
You can regard template as a parametrized version of vanilla schedule.

"""

from .space import ConfigSpace, Axis
from .dispatcher import dispatcher, DispatchContext, ApplyConfig
from .code_hash import attach_code_hash, attach_code_hash_to_arg
