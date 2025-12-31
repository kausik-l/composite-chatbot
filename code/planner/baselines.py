import random
import numpy as np

class ChatbotRandomPipelinePlanner:
    """
    Baseline: Randomly selects a valid component for the current stage.
    """
    def __init__(self, stage_map, action_name="select_component"):
        self.stage_map = stage_map
        self.action_name = action_name

    def sample_action(self, state):
        current_stage = self._get_stage(state)
        
        if current_stage and current_stage in self.stage_map:
            choice = random.choice(self.stage_map[current_stage])
            return {f"{self.action_name}___{choice}": 1}
        
        return {} 

    def _get_stage(self, state):
        for k, v in state.items():
            if "current_stage" in k and (v == True or v == 1):
                return k.split("___")[-1]
        return None

    def update(self, s, a, r, ns): 
        pass


class ChatbotFixedPipelinePlanner:
    """
    Baseline: always selects the N-th option in the list for each stage.
    """
    def __init__(self, stage_map, selection_index=0, name="Fixed", action_name="select_component"):
        self.stage_map = stage_map
        self.selection_index = selection_index
        self.name = name
        self.action_name = action_name

    def sample_action(self, state):
        current_stage = self._get_stage(state)
        
        if current_stage and current_stage in self.stage_map:
            options = self.stage_map[current_stage]
            # Ensure index is within bounds
            idx = min(self.selection_index, len(options) - 1)
            choice = options[idx]
            return {f"{self.action_name}___{choice}": 1}
        
        return {}

    def _get_stage(self, state):
        for k, v in state.items():
            if "current_stage" in k and (v == True or v == 1):
                return k.split("___")[-1]
        return None

    def update(self, s, a, r, ns): 
        pass