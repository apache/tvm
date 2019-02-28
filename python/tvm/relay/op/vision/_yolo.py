#pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import
from ..op import  register_schedule, register_pattern
from ..op import schedule_injective, OpPattern

# reorg
register_pattern("vision.yolo_reorg", OpPattern.INJECTIVE)
register_schedule("vision.yolo_reorg", schedule_injective)
