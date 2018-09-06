"""Task is a tunable composition of template functions.

Tuner takes a tunable task and optimizes the joint configuration
space of all the template functions in the task.
This module defines the task data structure, as well as a collection(zoo)
of typical tasks of interest.
"""

from .task import Task, create, register, template, get_config, args_to_workload
from .space import ConfigSpace, ConfigEntity
from .code_hash import attach_code_hash, attach_code_hash_to_arg
from .dispatcher import dispatcher, DispatchContext, ApplyConfig, ApplyHistoryBest, \
    FallbackContext, clear_fallback_cache

from .topi_integration import register_topi_compute, register_topi_schedule
from .nnvm_integration import extract_from_graph, extract_from_multiple_graph
