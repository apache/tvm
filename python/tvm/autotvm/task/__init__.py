"""Task is a tunable composition of template functions.

Tuner takes a tunable task and optimizes the joint configuration
space of all the template functions in the task.
This module defines the task data structure, as well as a collection(zoo)
of typical tasks of interest.
"""

from .task import Task, create, register, simple_template, get_config, create_task
