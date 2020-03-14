from attrdict import AttrDict
import numpy as np 


class Tuner:
    def __init__(self, config, runs, maximize=None):
        self.config_ = config
        self.runs_ = runs
        self.maximize_ = maximize
        self.run_id_ = 0
    
    @property
    def in_progress(self):
        return self.run_id_ < self.runs_

    def next(self, score):
        self.run_id_ += 1
        return _next(score)

    def _next(self, score):
        return NotImplementedError


class RandomSearchTuner(Tuner):

    def _next(self, score):
        return 
    
    @staticmethod
    def get_random_config(tunning_config):
        config_run = {}
        for tunable_name in tunning_config.keys():
            



