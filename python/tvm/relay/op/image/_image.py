#pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import
from ..op import  register_schedule, schedule_injective

# resize
register_schedule("image.resize", schedule_injective)
