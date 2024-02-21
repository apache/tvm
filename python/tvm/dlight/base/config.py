from dataclasses import dataclass

class ScheduleConfig:
    """Configuration for dlight schedule"""
    def __init__(self):
        self._config = {}
        self.block_factors = []
        self.thread_factors = []
        self.rstep = []
        self.reduce_thread = []
        self.pipeline_stage = 1
        self.vectorize = {}
        
    def __getattr__(self, name):
        return self._config[name]

    def __setattr__(self, name, value):
        self._config[name] = value
        
    def from_roller(self, roller_config):
        self.block = roller_config.block
        self.thread = roller_config.thread
        self.rstep = roller_config.rstep
        self.reduce_thread = roller_config.reduce_thread
        self.pipeline_stage = roller_config.pipeline_stage
        self.vectorize = roller_config.vectorize
